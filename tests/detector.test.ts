import assert from 'node:assert/strict';
import test from 'node:test';
import { DETECTION_CONFIG } from '../src/services/satellite/config.ts';
import {
  computeLuminanceAndLaplacian,
  estimateDownsampledDoGResponse,
} from '../src/services/satellite/imageProcessing.ts';
import { countTreesInPolygon } from '../src/services/satellite/polygonCounter.ts';
import { worldPxToLonLat } from '../src/services/satellite/tileFetcher.ts';
import {
  fuseDetectionsWBF,
  mergeDetectionsNMS,
  runMorphologicalFourGateFallback,
  type RawCandidate,
} from '../src/services/satellite/treeDetector.ts';
import type { TreeDetectionBox } from '../src/types/qa.ts';

function makeSolidCropTile(w: number, h: number): Uint8ClampedArray {
  // Uniform bright-green alfalfa/berseem crop field -> must produce 0 trees!
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    rgba[o] = 48;
    rgba[o + 1] = 132;
    rgba[o + 2] = 42;
    rgba[o + 3] = 255;
  }
  return rgba;
}

function makeOrchardTileWithTwoTrees(w: number, h: number): Uint8ClampedArray {
  // Sandy loam background (soil/dry ground) + 2 circular green crowns with directional SE cast shadows
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    rgba[o] = 175;
    rgba[o + 1] = 152;
    rgba[o + 2] = 122;
    rgba[o + 3] = 255;
  }

  const centers = [
    { cx: 45, cy: 45, r: 7 },
    { cx: 95, cy: 75, r: 7 },
  ];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      for (const c of centers) {
        // Directional dark cast shadow offset SE (+7, +7)
        const ds = Math.hypot(x - (c.cx + 7), y - (c.cy + 7));
        if (ds <= c.r * 0.95) {
          rgba[o] = 22;
          rgba[o + 1] = 24;
          rgba[o + 2] = 20;
        }
        // Bright sunlit crown with slight texture modulation
        const dc = Math.hypot(x - c.cx, y - c.cy);
        if (dc <= c.r) {
          const tex = ((x * 3 + y * 5) % 18) - 9;
          rgba[o] = Math.max(0, Math.min(255, 52 + tex));
          rgba[o + 1] = Math.max(0, Math.min(255, 165 + tex));
          rgba[o + 2] = Math.max(0, Math.min(255, 46 + tex));
        }
      }
    }
  }
  return rgba;
}

test('Class-aware WBF/NMS preserves overlapping distinct tree classes and fuses seam duplicates', () => {
  const candidates: RawCandidate[] = [
    { xmin: 10, ymin: 10, xmax: 30, ymax: 30, score: 0.9, classId: 0 }, // orchard_tree
    { xmin: 12, ymin: 12, xmax: 32, ymax: 32, score: 0.78, classId: 0 }, // overlapping seam orchard_tree
    { xmin: 13, ymin: 13, xmax: 33, ymax: 33, score: 0.85, classId: 1 }, // overlapping palm_tree (IoU < 0.85)
  ];

  const merged = mergeDetectionsNMS(candidates, DETECTION_CONFIG.nmsIouThresh, 8);
  assert.equal(merged.length, 2, 'Expected distinct orchard_tree and palm_tree classes to be preserved');
  assert.deepEqual(
    merged.map((m) => m.classId).sort(),
    [0, 1]
  );

  // Verify Weighted Box Fusion blended the two orchard_tree boxes instead of discarding the 2nd box
  const orchard = merged.find((m) => m.classId === 0)!;
  assert.ok(orchard.xmin > 10.2 && orchard.xmin < 11.8, `Expected confidence-weighted xmin blend, got ${orchard.xmin}`);
  assert.equal(fuseDetectionsWBF(candidates, DETECTION_CONFIG.wbfIouThresh, 8).length, 2);
});

test('Polygon counter produces genuine multi-threshold lo/mid/hi interval and unit aiFactor=1.0', () => {
  const w = 100;
  const h = 100;
  const strictMask = new Uint8Array(w * h).fill(1);
  const boundaryMask = new Uint8Array(w * h).fill(1);

  const candidates: RawCandidate[] = [
    { xmin: 10, ymin: 10, xmax: 20, ymax: 20, score: 0.88, classId: 0 },
    { xmin: 30, ymin: 10, xmax: 40, ymax: 20, score: 0.72, classId: 0 },
    { xmin: 50, ymin: 10, xmax: 60, ymax: 20, score: 0.58, classId: 0 },
    { xmin: 70, ymin: 10, xmax: 80, ymax: 20, score: 0.48, classId: 0 }, // below 0.55, included in conf-0.10=0.45
  ];

  const primaryBoxes: TreeDetectionBox[] = candidates.slice(0, 3).map((c, idx) => ({
    id: idx + 1,
    xmin: c.xmin,
    ymin: c.ymin,
    xmax: c.xmax,
    ymax: c.ymax,
    cx: (c.xmin + c.xmax) / 2,
    cy: (c.ymin + c.ymax) / 2,
    radiusPx: 5,
    score: c.score,
    classId: c.classId,
    className: 'orchard_tree',
    health: 'healthy',
    onBoundary: true,
    shadowDrop: 0.4,
    laplacianVar: 35,
  }));

  const summary = countTreesInPolygon({
    boxes: primaryBoxes,
    allThresholdCandidates: candidates,
    strictMask,
    boundaryMask,
    width: w,
    height: h,
    mpp: 0.26,
    inwardBufferM: -1.5,
  });

  assert.equal(summary.aiFactor, 1.0, 'aiFactor must remain strictly 1.0 (no unvalidated multipliers)');
  assert.equal(summary.aiCount, 3);
  assert.equal(summary.aiLow, 2, 'lo should reflect high-threshold conf+0.10=0.65 count');
  assert.equal(summary.aiHigh, 4, 'hi should reflect low-threshold conf-0.10=0.45 count');
  assert.ok(summary.aiLow < summary.aiCount && summary.aiCount < summary.aiHigh);

  // Also test custom lowered confThresh (0.45) capturing all 4 trees
  const sensitiveBoxes: TreeDetectionBox[] = candidates.map((c, idx) => ({
    id: idx + 1,
    xmin: c.xmin,
    ymin: c.ymin,
    xmax: c.xmax,
    ymax: c.ymax,
    cx: (c.xmin + c.xmax) / 2,
    cy: (c.ymin + c.ymax) / 2,
    radiusPx: 5,
    score: c.score,
    classId: c.classId,
    className: 'orchard_tree',
    health: 'healthy',
    onBoundary: true,
    shadowDrop: 0.4,
    laplacianVar: 35,
  }));
  const sensitiveSummary = countTreesInPolygon({
    boxes: sensitiveBoxes,
    allThresholdCandidates: candidates,
    strictMask,
    boundaryMask,
    width: w,
    height: h,
    mpp: 0.26,
    inwardBufferM: -1.5,
    confThresh: 0.45,
  });
  assert.equal(sensitiveSummary.aiCount, 4);
  assert.equal(sensitiveSummary.aiLow, 3); // >= 0.55
});

test('4-Gate morphological fallback rejects uniform green crop sheet (>85 m2) to 0 trees', () => {
  const w = 140;
  const h = 140;
  const rgba = makeSolidCropTile(w, h);
  const strictMask = new Uint8Array(w * h).fill(1);
  const boundaryMask = new Uint8Array(w * h).fill(1);
  const maps = computeLuminanceAndLaplacian(rgba, w, h, strictMask);

  const res = runMorphologicalFourGateFallback({
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26, // ~1324 m^2 total solid green crop basin
    species: 'citrus',
    maps,
  });

  assert.equal(res.boxes.length, 0, 'Flat crop basin must return 0 trees');
  assert.ok(res.gate2ComponentsRejected >= 1, 'Gate 2 should reject the continuous crop sheet');
});

test('4-Gate morphological fallback detects spaced tree crowns paired with directional cast shadows', () => {
  const w = 140;
  const h = 140;
  const rgba = makeOrchardTileWithTwoTrees(w, h);
  const strictMask = new Uint8Array(w * h).fill(1);
  const boundaryMask = new Uint8Array(w * h).fill(1);
  const maps = computeLuminanceAndLaplacian(rgba, w, h, strictMask);

  const res = runMorphologicalFourGateFallback({
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26,
    species: 'citrus',
    maps,
  });

  assert.equal(res.boxes.length, 2, 'Expected exact detection of the 2 shadowed orchard crowns');
  assert.ok(res.boxes.every((b) => b.shadowDrop >= DETECTION_CONFIG.minCastShadowDrop));
});

test('Downsampled DoG crown response and Mercator worldPxToLonLat behave deterministically', () => {
  const w = 140;
  const h = 140;
  const rgba = makeOrchardTileWithTwoTrees(w, h);
  const strictMask = new Uint8Array(w * h).fill(1);
  const maps = computeLuminanceAndLaplacian(rgba, w, h, strictMask);

  const dogOut = estimateDownsampledDoGResponse(maps.lum, w, h, 2);
  assert.equal(dogOut.dw, 70);
  assert.equal(dogOut.dh, 70);
  const soilIdx = Math.floor(15 / 2) * dogOut.dw + Math.floor(15 / 2);
  const treeIdx = Math.floor(45 / 2) * dogOut.dw + Math.floor(45 / 2);
  assert.ok(Math.abs(dogOut.dog[soilIdx]) < 1e-2, 'Flat uniform soil should have near-zero DoG response');
  assert.ok(Math.abs(dogOut.dog[treeIdx]) > Math.abs(dogOut.dog[soilIdx]), 'Crown should trigger stronger DoG than soil');

  const [lon, lat] = worldPxToLonLat(256 * Math.pow(2, 19) * 0.58, 256 * Math.pow(2, 19) * 0.41, 19);
  assert.ok(lon > 20 && lon < 40 && lat > 20 && lat < 40, 'Egypt-band Mercator projection check');
});
