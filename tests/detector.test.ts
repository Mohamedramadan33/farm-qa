import assert from 'node:assert/strict';
import test from 'node:test';
import { DETECTION_CONFIG } from '../src/services/satellite/config.ts';
import { computeSpatialOrchardMetrics } from '../src/services/satellite/auditorIntelligence.ts';
import {
  computeLuminanceAndLaplacian,
  estimateDownsampledDoGResponse,
  verifyAndScoreAiCandidate,
} from '../src/services/satellite/imageProcessing.ts';
import { countTreesInPolygon } from '../src/services/satellite/polygonCounter.ts';
import {
  lonLatToWorldPx,
  tileIntersectsPolygon,
  worldPxToLonLat,
} from '../src/services/satellite/tileFetcher.ts';
import {
  computeSquareTilePadding,
  fuseDetectionsWBF,
  generateSahiSlices,
  mergeDetectionsNMS,
  runMorphologicalFourGateFallback,
  type RawCandidate,
} from '../src/services/satellite/treeDetector.ts';
import {
  buildConsensusEnsemble,
  runMultiScaleDoGDetector,
  runShadowTexturePairDetector,
  summarizeLibraryResult,
} from '../src/services/satellite/multiLibraryEngine.ts';
import type { PointLonLat, TreeDetectionBox } from '../src/types/qa.ts';

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

  // Verify Weighted Box Fusion blended the two orchard_tree boxes and boosted multi-tile consensus confidence
  const orchard = merged.find((m) => m.classId === 0)!;
  assert.ok(orchard.xmin > 10.2 && orchard.xmin < 11.8, `Expected confidence-weighted xmin blend, got ${orchard.xmin}`);
  assert.ok(orchard.score > 0.9, `Expected multi-tile consensus score > 0.90, got ${orchard.score}`);
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

test('AI physical/spectral guard rejects concrete roofs & flat crop sheets while retaining shadowed/olive crowns', () => {
  const w = 140;
  const h = 140;
  const strictMask = new Uint8Array(w * h).fill(1);

  // 1. Flat uniform green crop sheet -> reject low-conf AI hallucination
  const cropRgba = makeSolidCropTile(w, h);
  const cropMaps = computeLuminanceAndLaplacian(cropRgba, w, h, strictMask);
  const flatCheck = verifyAndScoreAiCandidate({
    xmin: 40,
    ymin: 40,
    xmax: 58,
    ymax: 58,
    aiScore: 0.62,
    width: w,
    height: h,
    rgba: cropRgba,
    strictMask,
    maps: cropMaps,
  });
  assert.equal(flatCheck.keep, false);
  assert.equal(flatCheck.reason, 'flat_uniform_crop');

  // 2. Bright grey concrete/metal roof -> reject
  const roofRgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    roofRgba[o] = 205;
    roofRgba[o + 1] = 200;
    roofRgba[o + 2] = 198;
    roofRgba[o + 3] = 255;
  }
  const roofMaps = computeLuminanceAndLaplacian(roofRgba, w, h, strictMask);
  const roofCheck = verifyAndScoreAiCandidate({
    xmin: 30,
    ymin: 30,
    xmax: 52,
    ymax: 52,
    aiScore: 0.72,
    width: w,
    height: h,
    rgba: roofRgba,
    strictMask,
    maps: roofMaps,
  });
  assert.equal(roofCheck.keep, false);
  assert.equal(roofCheck.reason, 'achromatic_roof_or_sand');

  // 3. True shadowed orchard tree crown -> kept with healthy/calibrated score
  const orchardRgba = makeOrchardTileWithTwoTrees(w, h);
  const orchardMaps = computeLuminanceAndLaplacian(orchardRgba, w, h, strictMask);
  const treeCheck = verifyAndScoreAiCandidate({
    xmin: 38,
    ymin: 38,
    xmax: 52,
    ymax: 52,
    aiScore: 0.74,
    width: w,
    height: h,
    rgba: orchardRgba,
    strictMask,
    maps: orchardMaps,
  });
  assert.equal(treeCheck.keep, true);
  assert.equal(treeCheck.reason, 'ok');
  assert.ok(treeCheck.adjustedScore >= 0.68, `Expected calibrated score >= 0.68, got ${treeCheck.adjustedScore}`);
});

test('Spatial intelligence detects regular orchard row grid & flags oversized double-crowns', () => {
  const mpp = 0.25; // 4 px = 1m, so 20 px = 5m spacing
  const boxes: TreeDetectionBox[] = [];
  let id = 1;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const cx = 30 + col * 20;
      const cy = 30 + row * 20;
      const isOversized = id === 5; // 7m double crown vs 2.5m peers
      const rPx = isOversized ? 14 : 5;
      boxes.push({
        id,
        xmin: cx - rPx,
        ymin: cy - rPx,
        xmax: cx + rPx,
        ymax: cy + rPx,
        cx,
        cy,
        radiusPx: rPx,
        score: 0.88,
        classId: 0,
        className: 'orchard_tree',
        health: 'healthy',
        onBoundary: false,
        shadowDrop: 0.4,
        laplacianVar: 30,
        crownDiameterM: rPx * 2 * mpp,
      });
      id++;
    }
  }

  const metrics = computeSpatialOrchardMetrics({
    boxes,
    mpp,
    vegPct: 24,
    engineerCount: 10,
    aiCount: 9,
    forecast: 9,
    inwardBufferM: -1.5,
  });

  assert.equal(metrics.pattern, 'row_grid');
  assert.equal(metrics.medianSpacingM, 5);
  assert.equal(metrics.spacingRegularityPct, 100);
  assert.deepEqual(metrics.oversizedCrownIds, [5]);
  assert.equal(metrics.denseFlatCropDetected, false);
});

test('Spatial intelligence explains dense flat crop understory when vegPct is high and trees are few', () => {
  const metrics = computeSpatialOrchardMetrics({
    boxes: [],
    mpp: 0.26,
    vegPct: 68.5,
    engineerCount: 250,
    aiCount: 0,
    forecast: 80,
    inwardBufferM: -1.5,
  });

  assert.equal(metrics.pattern, 'sparse_or_none');
  assert.equal(metrics.denseFlatCropDetected, true);
  assert.ok(metrics.diagnosticsAr.some((d) => d.includes('غطاء نباتي أرضي كثيف')));
});

test('High-Res mode: tileIntersectsPolygon skips off-farm diagonal corner tiles while keeping farm tiles', () => {
  const z = 20;
  // Build a narrow diagonal polygon spanning across ~3 tile widths horizontally and vertically
  const [cx, cy] = lonLatToWorldPx(30.92, 30.65, z);
  const baseTx = Math.floor(cx / 256);
  const baseTy = Math.floor(cy / 256);
  const toPt = (wx: number, wy: number): PointLonLat => worldPxToLonLat(wx, wy, z);

  // Diagonal corridor from (baseTx, baseTy) to (baseTx+2, baseTy+2) -> corners (baseTx+2, baseTy) & (baseTx, baseTy+2) are empty
  const diagPoly: PointLonLat[] = [
    toPt(baseTx * 256 + 40, baseTy * 256 + 10),
    toPt(baseTx * 256 + 10, baseTy * 256 + 40),
    toPt((baseTx + 2) * 256 + 216, (baseTy + 2) * 256 + 246),
    toPt((baseTx + 2) * 256 + 246, (baseTy + 2) * 256 + 216),
  ];

  assert.equal(tileIntersectsPolygon(baseTx, baseTy, z, diagPoly, 0), true, 'Start corner intersects');
  assert.equal(tileIntersectsPolygon(baseTx + 1, baseTy + 1, z, diagPoly, 0), true, 'Middle tile intersects');
  assert.equal(tileIntersectsPolygon(baseTx + 2, baseTy + 2, z, diagPoly, 0), true, 'End corner intersects');
  assert.equal(tileIntersectsPolygon(baseTx + 2, baseTy, z, diagPoly, 0), false, 'Far top-right corner skipped');
  assert.equal(tileIntersectsPolygon(baseTx, baseTy + 2, z, diagPoly, 0), false, 'Far bottom-left corner skipped');
});

test('High-Res mode: 288px 40%-overlap SAHI slicing & zero-stretch square padding preserve 1:1 crop aspect', () => {
  const w = 640;
  const h = 420;
  const mask = new Uint8Array(w * h).fill(1);

  const stdSlices = generateSahiSlices(
    w,
    h,
    mask,
    DETECTION_CONFIG.sahiTileSize,
    DETECTION_CONFIG.sahiOverlap,
    0.05
  );
  const hiResSlices = generateSahiSlices(
    w,
    h,
    mask,
    DETECTION_CONFIG.highResTileSize,
    DETECTION_CONFIG.highResOverlap,
    0.03
  );

  assert.equal(DETECTION_CONFIG.highResTileSize, 288);
  assert.equal(DETECTION_CONFIG.highResOverlap, 0.4);
  assert.ok(hiResSlices.length > stdSlices.length, 'High-Res 288px 40%-overlap produces finer overlapping grid');

  // Verify non-square small parcel edge slice pads cleanly to 288x288 without distorting scale
  const smallEdgeSlices = generateSahiSlices(190, 140, new Uint8Array(190 * 140).fill(1), 288, 0.4, 0.03);
  assert.equal(smallEdgeSlices.length, 1);
  const padSpec = computeSquareTilePadding(smallEdgeSlices[0], DETECTION_CONFIG.highResTileSize);
  assert.equal(padSpec.targetSize, 288);
  assert.equal(padSpec.copyW, 190);
  assert.equal(padSpec.copyH, 140);
  assert.equal(padSpec.padRight, 98);
  assert.equal(padSpec.padBottom, 148);
});

test('High-Res tighter WBF separation preserves closely spaced ~2.0m young saplings at Z20', () => {
  const mpp = 0.13; // Zoom 20 resolution (~0.13 m/px)
  const minCenterDistPx = Math.max(3, Math.round(1.3 / Math.max(mpp, 0.1))); // ~10px (1.3m)
  // Two adjacent saplings 2.0m (~15.4px) apart center-to-center
  const saplings: RawCandidate[] = [
    { xmin: 40, ymin: 40, xmax: 52, ymax: 52, score: 0.86, classId: 0 }, // center (46, 46)
    { xmin: 56, ymin: 40, xmax: 68, ymax: 52, score: 0.84, classId: 0 }, // center (62, 46) -> 16px apart
  ];

  const fused = fuseDetectionsWBF(saplings, DETECTION_CONFIG.wbfIouThresh, minCenterDistPx);
  assert.equal(fused.length, 2, 'Closely spaced ~2m saplings must remain distinct under High-Res WBF');
});

test('Multi-Scale DoG & Shadow-Texture Pair detectors reject solid crop fields and detect true shadowed crowns', () => {
  const w = 140;
  const h = 140;
  const strictMask = new Uint8Array(w * h).fill(1);
  const boundaryMask = new Uint8Array(w * h).fill(1);

  // 1. Flat uniform crop field -> both extra CV libraries must output 0 trees
  const cropRgba = makeSolidCropTile(w, h);
  const cropMaps = computeLuminanceAndLaplacian(cropRgba, w, h, strictMask);

  const dogCrop = runMultiScaleDoGDetector({
    rgba: cropRgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26,
    species: 'citrus',
    maps: cropMaps,
  });
  assert.equal(dogCrop.boxes.length, 0, 'Multi-Scale DoG must return 0 trees on uniform crop');

  const shadowCrop = runShadowTexturePairDetector({
    rgba: cropRgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26,
    species: 'citrus',
    maps: cropMaps,
  });
  assert.equal(shadowCrop.boxes.length, 0, 'Shadow-Texture Pair must return 0 trees on uniform crop');

  // 2. Orchard tile with 2 shadowed trees -> both extra CV libraries detect them accurately
  const orchardRgba = makeOrchardTileWithTwoTrees(w, h);
  const orchardMaps = computeLuminanceAndLaplacian(orchardRgba, w, h, strictMask);

  const dogOrchard = runMultiScaleDoGDetector({
    rgba: orchardRgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26,
    species: 'citrus',
    maps: orchardMaps,
  });
  assert.equal(dogOrchard.boxes.length, 2, 'Multi-Scale DoG should detect both shadowed orchard crowns');

  const shadowOrchard = runShadowTexturePairDetector({
    rgba: orchardRgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp: 0.26,
    species: 'citrus',
    maps: orchardMaps,
  });
  assert.equal(shadowOrchard.boxes.length, 2, 'Shadow-Texture Pair should detect both shadowed orchard crowns');
});

test('Geo-DBSCAN Consensus Ensemble fuses multi-library detections and rejects isolated low-confidence noise', () => {
  const w = 160;
  const h = 160;
  const strictMask = new Uint8Array(w * h).fill(1);
  const boundaryMask = new Uint8Array(w * h).fill(1);
  const mpp = 0.26;

  const makeBox = (
    id: number,
    cx: number,
    cy: number,
    score: number,
    shadowDrop = 0.36,
    laplacianVar = 24
  ): TreeDetectionBox => ({
    id,
    xmin: cx - 6,
    ymin: cy - 6,
    xmax: cx + 6,
    ymax: cy + 6,
    cx,
    cy,
    radiusPx: 6,
    score,
    classId: 0,
    className: 'orchard_tree',
    health: 'healthy',
    onBoundary: false,
    shadowDrop,
    laplacianVar,
    crownDiameterM: 3.1,
    source: 'ai',
  });

  // Tree A at (40,40) seen by Gemini + Watershed + DoG
  // Tree B at (95,95) missed by Gemini, but strongly confirmed by BOTH Watershed + DoG -> recovered!
  // Noise C at (130,30) hallucinated by ONLY 1 classical library with moderate score -> filtered out!
  const geminiSummary = summarizeLibraryResult({
    id: 'gemini_vision',
    name: 'Gemini 4 Hydrogen Spatial Vision',
    nameAr: 'رؤية الذكاء الاصطناعي',
    libraryBadge: '@google/genai',
    category: 'ai_vision',
    status: 'ran',
    boxes: [makeBox(1, 40, 40, 0.88)],
    inwardBufferM: -1.5,
    mpp,
    durationMs: 120,
    notesAr: '',
  });

  const morphSummary = summarizeLibraryResult({
    id: 'morphological_watershed',
    name: '4-Gate Morphological EDT Watershed',
    nameAr: 'التحليل المورفولوجي',
    libraryBadge: 'Meijster EDT',
    category: 'cv_classical',
    status: 'ran',
    boxes: [makeBox(1, 41, 40, 0.78), makeBox(2, 95, 95, 0.74), makeBox(3, 130, 30, 0.62, 0.25, 10)],
    inwardBufferM: -1.5,
    mpp,
    durationMs: 12,
    notesAr: '',
  });

  const dogSummary = summarizeLibraryResult({
    id: 'multiscale_dog',
    name: 'Multi-Scale DoG Scale-Space',
    nameAr: 'كاشف القمم',
    libraryBadge: 'Gaussian Pyramid DoG',
    category: 'cv_classical',
    status: 'ran',
    boxes: [makeBox(1, 40, 41, 0.76), makeBox(2, 96, 95, 0.71)],
    inwardBufferM: -1.5,
    mpp,
    durationMs: 8,
    notesAr: '',
  });

  const { ensembleSummary, report } = buildConsensusEnsemble({
    baseLibraries: [geminiSummary, morphSummary, dogSummary],
    mpp,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    inwardBufferM: -1.5,
    confThresh: 0.55,
    highRes: false,
  });

  assert.equal(ensembleSummary.count, 2, 'Ensemble should keep Tree A + multi-CV recovered Tree B, and drop Noise C');
  assert.equal(report.consensusCount, 2);
  assert.equal(report.recommendedEngineId, 'turf_consensus_ensemble');
  assert.ok(report.agreementPct >= 50 && report.agreementPct <= 100);
});



