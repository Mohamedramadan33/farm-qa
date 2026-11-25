import { clustersDbscan } from '@turf/clusters-dbscan';
import { featureCollection, point } from '@turf/helpers';
import { normAr, present } from '../../lib/arabicNorm';
import { DETECTION_CONFIG, getDetectionAuthHeaders } from './config';
import {
  computeStage1VegetationMask,
  evaluateCrownShadowAndHealth,
  type SpectralMaps,
} from './imageProcessing';
import { worldPxToLonLat } from './tileFetcher';
import type {
  AiLibraryArbitration,
  LibraryDetectionSummary,
  LibraryEngineId,
  MultiLibraryComparisonReport,
  TreeClassName,
  TreeDetectionBox,
  TreeHealthStatus,
} from '../../types/qa';

function inferFallbackTreeClass(
  species: string,
  onBoundary: boolean,
  diameterM: number
): {
  classId: 0 | 1 | 2;
  className: TreeClassName;
} {
  const s = present(species) ? normAr(species) : '';
  const isPalm = ['نخيل', 'نخل', 'بلح', 'تمر', 'palm', 'date'].some((k) => s.includes(normAr(k)));
  const isWindbreak = [
    'جازورين',
    'كازورينا',
    'كافور',
    'سرو',
    'فيكس',
    'ماهوجني',
    'casuarina',
    'eucalyptus',
    'مصدات',
  ].some((k) => s.includes(normAr(k)));

  if (isPalm || (!onBoundary && diameterM >= 5.2)) {
    return { classId: 1, className: 'palm_tree' };
  }
  if (onBoundary || isWindbreak) {
    return { classId: 2, className: 'windbreak_tree' };
  }
  return { classId: 0, className: 'orchard_tree' };
}

// ---------------------------------------------------------------------------
// Separable Gaussian Blur Helper (for Scale-Space DoG Canopy Detection)
// ---------------------------------------------------------------------------

function separableGaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const rad = Math.max(1, Math.min(18, Math.ceil(sigma * 2.2)));
  const kLen = rad * 2 + 1;
  const kernel = new Float32Array(kLen);
  const denom = 2 * sigma * sigma;
  let kSum = 0;
  for (let i = -rad; i <= rad; i++) {
    const val = Math.exp(-(i * i) / denom);
    kernel[i + rad] = val;
    kSum += val;
  }
  for (let i = 0; i < kLen; i++) kernel[i] /= kSum;

  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const xx = x + k < 0 ? 0 : x + k >= w ? w - 1 : x + k;
        acc += src[row + xx] * kernel[k + rad];
      }
      tmp[row + x] = acc;
    }
  }

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const yy = y + k < 0 ? 0 : y + k >= h ? h - 1 : y + k;
        acc += tmp[yy * w + x] * kernel[k + rad];
      }
      out[row + x] = acc;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Engine 3: Multi-Scale DoG / Scale-Space Blob Canopy Detector
// ---------------------------------------------------------------------------

export function runMultiScaleDoGDetector(params: {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  mpp: number;
  species: string;
  maps: SpectralMaps;
  stage1MaskResult?: ReturnType<typeof computeStage1VegetationMask>;
  worldOriginX?: number;
  worldOriginY?: number;
  zoom?: number;
}): { boxes: TreeDetectionBox[]; durationMs: number; scalesM: number[] } {
  const t0 = performance.now();
  const {
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp,
    species,
    maps,
    worldOriginX,
    worldOriginY,
    zoom,
  } = params;
  const { lum, hsvV, hsvS, exg, lapMag } = maps;
  const n = w * h;

  const stage1 =
    params.stage1MaskResult ??
    computeStage1VegetationMask({ rgba, width: w, height: h, strictMask, mpp, maps });
  const { vegMask, gate2KeptPixels } = stage1;

  const scalesM = [0.75, 1.25, 1.95]; // maps roughly to ~1.9m, ~3.2m, ~5.0m crown diameters

  if (gate2KeptPixels === 0 || w < 16 || h < 16) {
    return { boxes: [], durationMs: Math.round(performance.now() - t0), scalesM };
  }

  // Green canopy prominence scalar field (zeroed outside Gate-2 clean vegetation mask)
  const signal = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (!vegMask[i]) continue;
    const ePos = Math.max(0, exg[i]);
    const lumFactor = Math.max(0, Math.min(1, lum[i] / 210));
    signal[i] = ePos * (0.4 + 0.6 * hsvS[i]) * lumFactor;
  }

  const safeMpp = Math.max(mpp, 0.1);
  const dogMaps: Float32Array[] = [];
  const scaleRadiiPx: number[] = [];

  for (const radM of scalesM) {
    const sigma = Math.max(1.15, Math.min(14, (radM / safeMpp) * 0.62));
    const gFine = separableGaussianBlur(signal, w, h, sigma);
    const gCoarse = separableGaussianBlur(signal, w, h, sigma * 1.85);
    const dog = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      dog[i] = Math.max(0, (gFine[i] - gCoarse[i]) * 2.2);
    }
    dogMaps.push(dog);
    scaleRadiiPx.push(Math.max(4, Math.round(radM / safeMpp)));
  }

  interface DogPeak {
    x: number;
    y: number;
    scaleIdx: number;
    resp: number;
    radiusPx: number;
  }

  const peaks: DogPeak[] = [];
  for (let y = 3; y < h - 3; y++) {
    const row = y * w;
    for (let x = 3; x < w - 3; x++) {
      const idx = row + x;
      if (!vegMask[idx]) continue;

      let bestS = -1;
      let bestResp = 0.018;
      for (let s = 0; s < dogMaps.length; s++) {
        const val = dogMaps[s][idx];
        if (val > bestResp) {
          bestResp = val;
          bestS = s;
        }
      }
      if (bestS < 0) continue;

      const curMap = dogMaps[bestS];
      const isLocalMax =
        bestResp >= curMap[idx - 1] &&
        bestResp >= curMap[idx + 1] &&
        bestResp >= curMap[idx - w] &&
        bestResp >= curMap[idx + w] &&
        bestResp >= curMap[idx - w - 1] &&
        bestResp >= curMap[idx - w + 1] &&
        bestResp >= curMap[idx + w - 1] &&
        bestResp >= curMap[idx + w + 1];

      if (isLocalMax) {
        peaks.push({
          x,
          y,
          scaleIdx: bestS,
          resp: bestResp,
          radiusPx: scaleRadiiPx[bestS],
        });
      }
    }
  }

  peaks.sort((a, b) => b.resp - a.resp);

  const verifiedBoxes: TreeDetectionBox[] = [];
  let nextId = 1;
  const minGlobalSpacingPx = Math.max(4, Math.round(2.0 / safeMpp));

  for (const p of peaks) {
    const minSep = Math.max(minGlobalSpacingPx, Math.round(p.radiusPx * 1.15));
    if (verifiedBoxes.some((b) => Math.hypot(b.cx - p.x, b.cy - p.y) < minSep)) {
      continue;
    }

    const r = p.radiusPx;
    const x1 = Math.max(1, p.x - r);
    const y1 = Math.max(1, p.y - r);
    const x2 = Math.min(w - 2, p.x + r);
    const y2 = Math.min(h - 2, p.y + r);

    const evalRes = evaluateCrownShadowAndHealth(
      p.x,
      p.y,
      r,
      x1,
      y1,
      x2,
      y2,
      w,
      h,
      rgba,
      strictMask,
      lum,
      hsvV,
      exg,
      lapMag
    );
    if (!evalRes || !evalRes.hasPairedCastShadow) continue;
    if (evalRes.laplacianVar < 8.0) continue;

    const diamM = Number((r * 2 * mpp).toFixed(2));
    if (diamM < DETECTION_CONFIG.minCrownDiamM || diamM > DETECTION_CONFIG.maxCrownDiamM) {
      continue;
    }

    const onBoundary = Boolean(boundaryMask[p.y * w + p.x]);
    const cls = inferFallbackTreeClass(species, onBoundary, diamM);
    const score = Math.min(0.94, 0.5 + Math.min(0.24, p.resp * 2.5) + evalRes.relShadowDrop * 0.25);

    let lat: number | undefined;
    let lon: number | undefined;
    if (worldOriginX !== undefined && worldOriginY !== undefined && zoom !== undefined) {
      [lon, lat] = worldPxToLonLat(worldOriginX + p.x, worldOriginY + p.y, zoom);
    }

    verifiedBoxes.push({
      id: nextId++,
      xmin: x1,
      ymin: y1,
      xmax: x2,
      ymax: y2,
      cx: p.x,
      cy: p.y,
      radiusPx: r,
      score: Number(score.toFixed(3)),
      classId: cls.classId,
      className: cls.className,
      health: evalRes.health,
      onBoundary,
      shadowDrop: Number(evalRes.relShadowDrop.toFixed(3)),
      laplacianVar: Number(evalRes.laplacianVar.toFixed(1)),
      crownDiameterM: diamM,
      lat,
      lon,
      source: 'ai',
    });

    if (verifiedBoxes.length >= 1200) break;
  }

  return {
    boxes: verifiedBoxes,
    durationMs: Math.round(performance.now() - t0),
    scalesM,
  };
}

// ---------------------------------------------------------------------------
// Engine 5: Cast-Shadow Azimuth Pair + Texture Crown Detector
// ---------------------------------------------------------------------------

export function runShadowTexturePairDetector(params: {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  mpp: number;
  species: string;
  maps: SpectralMaps;
  worldOriginX?: number;
  worldOriginY?: number;
  zoom?: number;
}): { boxes: TreeDetectionBox[]; shadowClustersFound: number; durationMs: number } {
  const t0 = performance.now();
  const {
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp,
    species,
    maps,
    worldOriginX,
    worldOriginY,
    zoom,
  } = params;
  const { lum, hsvV, exg, lapMag } = maps;
  const n = w * h;
  const safeMpp = Math.max(mpp, 0.1);
  const pxAreaM2 = safeMpp * safeMpp;

  // Step 1: Extract isolated dark shadow pockets inside strictMask (exclude deep canals/roofs via area bounds)
  const shadowMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!strictMask[i]) continue;
    const o = i * 4;
    const b = rgba[o + 2];
    const g = rgba[o + 1];
    // Dark cast shadow: low HSV Value and low luminance, not bright blue water
    if (hsvV[i] <= 0.25 && lum[i] <= 74 && b <= g * 1.25 + 14) {
      shadowMask[i] = 1;
    }
  }

  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);
  const shadowBlobs: Array<{ sx: number; sy: number; eqRadPx: number; areaM2: number }> = [];

  for (let seed = 0; seed < n; seed++) {
    if (!shadowMask[seed] || visited[seed]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    visited[seed] = 1;
    let sumX = 0;
    let sumY = 0;

    while (head < tail) {
      const cur = queue[head++];
      const cy = Math.floor(cur / w);
      const cx = cur - cy * w;
      sumX += cx;
      sumY += cy;

      if (cx + 1 < w && shadowMask[cur + 1] && !visited[cur + 1]) {
        visited[cur + 1] = 1;
        queue[tail++] = cur + 1;
      }
      if (cx - 1 >= 0 && shadowMask[cur - 1] && !visited[cur - 1]) {
        visited[cur - 1] = 1;
        queue[tail++] = cur - 1;
      }
      if (cy + 1 < h && shadowMask[cur + w] && !visited[cur + w]) {
        visited[cur + w] = 1;
        queue[tail++] = cur + w;
      }
      if (cy - 1 >= 0 && shadowMask[cur - w] && !visited[cur - w]) {
        visited[cur - w] = 1;
        queue[tail++] = cur - w;
      }
    }

    const areaM2 = tail * pxAreaM2;
    if (areaM2 < 0.55 || areaM2 > 26.0) continue;

    shadowBlobs.push({
      sx: sumX / tail,
      sy: sumY / tail,
      eqRadPx: Math.max(2, Math.sqrt(tail / Math.PI)),
      areaM2,
    });
  }

  // Step 2: For each shadow pocket, probe 8 radial directions for adjacent lit green leafy crown center
  const candidateBoxes: TreeDetectionBox[] = [];
  const minSepPx = Math.max(4, Math.round(2.0 / safeMpp));
  let nextId = 1;

  for (const sb of shadowBlobs) {
    const probeDistMin = Math.max(3, sb.eqRadPx * 1.15);
    const probeDistMax = Math.max(probeDistMin + 3, Math.min(24, sb.eqRadPx * 2.5));

    let bestCandidate: {
      cx: number;
      cy: number;
      radiusPx: number;
      score: number;
      health: TreeHealthStatus;
      shadowDrop: number;
      lapVar: number;
    } | null = null;

    for (let dir = 0; dir < 8; dir++) {
      const ang = (dir * Math.PI) / 4;
      const dx = Math.cos(ang);
      const dy = Math.sin(ang);

      for (const frac of [0.35, 0.7]) {
        const dist = probeDistMin + (probeDistMax - probeDistMin) * frac;
        const px = Math.round(sb.sx + dx * dist);
        const py = Math.round(sb.sy + dy * dist);
        if (px < 4 || px >= w - 4 || py < 4 || py >= h - 4 || !strictMask[py * w + px]) {
          continue;
        }
        if (exg[py * w + px] <= 0.035 || lum[py * w + px] < 65) continue;

        const estRadPx = Math.max(4, Math.min(18, Math.round(Math.max(sb.eqRadPx * 1.05, dist * 0.55))));
        const x1 = Math.max(1, px - estRadPx);
        const y1 = Math.max(1, py - estRadPx);
        const x2 = Math.min(w - 2, px + estRadPx);
        const y2 = Math.min(h - 2, py + estRadPx);

        const evalRes = evaluateCrownShadowAndHealth(
          px,
          py,
          estRadPx,
          x1,
          y1,
          x2,
          y2,
          w,
          h,
          rgba,
          strictMask,
          lum,
          hsvV,
          exg,
          lapMag
        );
        if (!evalRes || !evalRes.hasPairedCastShadow) continue;
        if (evalRes.meanExg <= 0.04 || evalRes.laplacianVar < 9.0) continue;

        const score = Math.min(
          0.93,
          0.5 + evalRes.relShadowDrop * 0.34 + Math.min(0.18, evalRes.meanExg * 0.65)
        );
        if (!bestCandidate || score > bestCandidate.score) {
          bestCandidate = {
            cx: px,
            cy: py,
            radiusPx: estRadPx,
            score,
            health: evalRes.health,
            shadowDrop: evalRes.relShadowDrop,
            lapVar: evalRes.laplacianVar,
          };
        }
      }
    }

    if (!bestCandidate) continue;
    const diamM = Number((bestCandidate.radiusPx * 2 * mpp).toFixed(2));
    if (diamM < DETECTION_CONFIG.minCrownDiamM || diamM > DETECTION_CONFIG.maxCrownDiamM) {
      continue;
    }

    if (
      candidateBoxes.some(
        (b) => Math.hypot(b.cx - bestCandidate!.cx, b.cy - bestCandidate!.cy) < minSepPx
      )
    ) {
      continue;
    }

    const onBoundary = Boolean(boundaryMask[bestCandidate.cy * w + bestCandidate.cx]);
    const cls = inferFallbackTreeClass(species, onBoundary, diamM);

    let lat: number | undefined;
    let lon: number | undefined;
    if (worldOriginX !== undefined && worldOriginY !== undefined && zoom !== undefined) {
      [lon, lat] = worldPxToLonLat(
        worldOriginX + bestCandidate.cx,
        worldOriginY + bestCandidate.cy,
        zoom
      );
    }

    candidateBoxes.push({
      id: nextId++,
      xmin: Math.max(1, bestCandidate.cx - bestCandidate.radiusPx),
      ymin: Math.max(1, bestCandidate.cy - bestCandidate.radiusPx),
      xmax: Math.min(w - 2, bestCandidate.cx + bestCandidate.radiusPx),
      ymax: Math.min(h - 2, bestCandidate.cy + bestCandidate.radiusPx),
      cx: bestCandidate.cx,
      cy: bestCandidate.cy,
      radiusPx: bestCandidate.radiusPx,
      score: Number(bestCandidate.score.toFixed(3)),
      classId: cls.classId,
      className: cls.className,
      health: bestCandidate.health,
      onBoundary,
      shadowDrop: Number(bestCandidate.shadowDrop.toFixed(3)),
      laplacianVar: Number(bestCandidate.lapVar.toFixed(1)),
      crownDiameterM: diamM,
      lat,
      lon,
      source: 'ai',
    });

    if (candidateBoxes.length >= 1200) break;
  }

  return {
    boxes: candidateBoxes,
    shadowClustersFound: shadowBlobs.length,
    durationMs: Math.round(performance.now() - t0),
  };
}

// ---------------------------------------------------------------------------
// Summary Builder & Turf Geo-DBSCAN Multi-Library Consensus Ensemble
// ---------------------------------------------------------------------------

export function summarizeLibraryResult(params: {
  id: LibraryEngineId;
  name: string;
  nameAr: string;
  libraryBadge: string;
  category: LibraryDetectionSummary['category'];
  status: LibraryDetectionSummary['status'];
  unavailableReason?: string;
  boxes: TreeDetectionBox[];
  inwardBufferM: number;
  mpp: number;
  durationMs: number;
  notesAr: string;
}): LibraryDetectionSummary {
  const {
    id,
    name,
    nameAr,
    libraryBadge,
    category,
    status,
    unavailableReason,
    boxes,
    inwardBufferM,
    mpp,
    durationMs,
    notesAr,
  } = params;

  const boundaryCount = boxes.filter((b) => b.onBoundary).length;
  const interiorCount = boxes.length - boundaryCount;
  const activeBoxes = inwardBufferM < -0.5 ? boxes : boxes.filter((b) => b.onBoundary);

  const avgConfidence =
    boxes.length > 0
      ? Number((boxes.reduce((acc, b) => acc + b.score, 0) / boxes.length).toFixed(3))
      : 0;

  let medianCrownM: number | null = null;
  if (activeBoxes.length > 0) {
    const diams = activeBoxes
      .map((b) => b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * mpp)
      .sort((a, b) => a - b);
    medianCrownM = Number(diams[Math.floor(diams.length / 2)].toFixed(2));
  }

  return {
    id,
    name,
    nameAr,
    libraryBadge,
    category,
    status,
    unavailableReason,
    count: activeBoxes.length,
    interiorCount,
    boundaryCount,
    avgConfidence,
    medianCrownM,
    durationMs,
    boxes,
    notesAr,
  };
}

const ENGINE_WEIGHTS: Record<LibraryEngineId, number> = {
  gemini_vision: 2.3,
  yolov8_onnx: 2.1,
  morphological_watershed: 1.0,
  multiscale_dog: 1.0,
  shadow_texture_pair: 1.0,
  turf_consensus_ensemble: 2.5,
};

function syntheticLonLat(cx: number, cy: number, mpp: number): [number, number] {
  const originLat = 30.05;
  const originLon = 31.2;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos((originLat * Math.PI) / 180);
  return [originLon + (cx * mpp) / mPerDegLon, originLat - (cy * mpp) / mPerDegLat];
}

export function buildConsensusEnsemble(params: {
  baseLibraries: LibraryDetectionSummary[];
  mpp: number;
  width: number;
  height: number;
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  inwardBufferM: number;
  confThresh: number;
  highRes: boolean;
  worldOriginX?: number;
  worldOriginY?: number;
  zoom?: number;
}): {
  ensembleSummary: LibraryDetectionSummary;
  report: MultiLibraryComparisonReport;
} {
  const t0 = performance.now();
  const {
    baseLibraries,
    mpp,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    inwardBufferM,
    confThresh,
    highRes,
    worldOriginX,
    worldOriginY,
    zoom,
  } = params;

  const ranBase = baseLibraries.filter((l) => l.status === 'ran');
  const hasGemini = ranBase.some((l) => l.id === 'gemini_vision');
  const hasYolo = ranBase.some((l) => l.id === 'yolov8_onnx');
  const hasSemanticAi = hasGemini || hasYolo;

  interface TaggedBox {
    engineId: LibraryEngineId;
    weight: number;
    box: TreeDetectionBox;
    lon: number;
    lat: number;
  }

  const allTagged: TaggedBox[] = [];
  for (const lib of ranBase) {
    const wt = ENGINE_WEIGHTS[lib.id] ?? 1.0;
    for (const box of lib.boxes) {
      const [lon, lat] =
        box.lon !== undefined && box.lat !== undefined
          ? [box.lon, box.lat]
          : worldOriginX !== undefined && worldOriginY !== undefined && zoom !== undefined
            ? worldPxToLonLat(worldOriginX + box.cx, worldOriginY + box.cy, zoom)
            : syntheticLonLat(box.cx, box.cy, mpp);
      allTagged.push({ engineId: lib.id, weight: wt, box, lon, lat });
    }
  }

  const clusterEpsMeters = highRes ? 1.45 : 1.75;
  const ensembleBoxes: TreeDetectionBox[] = [];

  if (allTagged.length > 0) {
    const fc = featureCollection(
      allTagged.map((tb, idx) =>
        point([tb.lon, tb.lat], {
          idx,
          engineId: tb.engineId,
        })
      )
    );

    const clustered = clustersDbscan(fc, clusterEpsMeters, {
      units: 'meters',
      minPoints: 1,
    });

    const clusterBuckets = new Map<number, TaggedBox[]>();
    for (const feat of clustered.features) {
      const cid = typeof feat.properties?.cluster === 'number' ? feat.properties.cluster : -1;
      const idx = Number(feat.properties?.idx ?? -1);
      if (cid < 0 || idx < 0 || idx >= allTagged.length) continue;
      const arr = clusterBuckets.get(cid);
      if (arr) arr.push(allTagged[idx]);
      else clusterBuckets.set(cid, [allTagged[idx]]);
    }

    const minCenterSepPx = Math.max(4, Math.round((highRes ? 1.35 : 1.65) / Math.max(mpp, 0.1)));
    const candidateFused: TreeDetectionBox[] = [];

    for (const members of clusterBuckets.values()) {
      // Keep highest-score representative per engine in each cluster so one engine cannot self-vote twice
      const bestPerEngine = new Map<LibraryEngineId, TaggedBox>();
      for (const m of members) {
        const cur = bestPerEngine.get(m.engineId);
        if (!cur || m.box.score > cur.box.score) {
          bestPerEngine.set(m.engineId, m);
        }
      }
      const uniqueVotes = Array.from(bestPerEngine.values());

      const geminiVote = bestPerEngine.get('gemini_vision');
      const yoloVote = bestPerEngine.get('yolov8_onnx');
      const classicalVotes = uniqueVotes.filter(
        (v) => v.engineId !== 'gemini_vision' && v.engineId !== 'yolov8_onnx'
      );

      let totalWeight = 0;
      let wCx = 0;
      let wCy = 0;
      let wXmin = 0;
      let wYmin = 0;
      let wXmax = 0;
      let wYmax = 0;
      let bestScore = 0;
      let bestShadow = 0;
      let bestLap = 0;

      const classScores: Record<TreeClassName, number> = {
        orchard_tree: 0,
        palm_tree: 0,
        windbreak_tree: 0,
      };
      const healthScores: Record<TreeHealthStatus, number> = {
        healthy: 0,
        stressed: 0,
        dead: 0,
      };

      for (const v of uniqueVotes) {
        const effW = v.weight * Math.max(0.35, v.box.score);
        totalWeight += effW;
        wCx += v.box.cx * effW;
        wCy += v.box.cy * effW;
        wXmin += v.box.xmin * effW;
        wYmin += v.box.ymin * effW;
        wXmax += v.box.xmax * effW;
        wYmax += v.box.ymax * effW;
        bestScore = Math.max(bestScore, v.box.score);
        bestShadow = Math.max(bestShadow, v.box.shadowDrop);
        bestLap = Math.max(bestLap, v.box.laplacianVar);
        classScores[v.box.className] = (classScores[v.box.className] || 0) + effW;
        const hStat = v.box.health || 'healthy';
        healthScores[hStat] = (healthScores[hStat] || 0) + effW;
      }

      // Consensus Acceptance Rule:
      // 1) If Semantic AI (Gemini / YOLO) ran: accept if supported by Semantic AI above threshold
      //    OR corroborated by >=2 independent classical CV libraries with real cast shadow & texture.
      // 2) If offline (no AI Vision): accept if >=2 independent classical CV libraries agree, OR
      //    a single engine has very strong physical shadow+texture crown evidence.
      let keep = false;
      if (hasSemanticAi) {
        const aiScore = Math.max(geminiVote?.box.score ?? 0, yoloVote?.box.score ?? 0);
        const aiDirectKeep =
          Boolean(geminiVote || yoloVote) &&
          (aiScore >= Math.max(0.45, confThresh - 0.03) ||
            (uniqueVotes.length >= 2 && aiScore >= Math.max(0.38, confThresh - 0.12)));
        const multiCvRecovery =
          classicalVotes.length >= 2 &&
          bestScore >= 0.64 &&
          bestShadow >= 0.28 &&
          bestLap >= 12.0;
        keep = aiDirectKeep || multiCvRecovery;
      } else {
        const multiCvAgree = classicalVotes.length >= 2 && bestScore >= 0.54;
        const strongSoloCrown =
          classicalVotes.length === 1 &&
          bestScore >= 0.76 &&
          bestShadow >= 0.34 &&
          bestLap >= 18.0;
        keep = multiCvAgree || strongSoloCrown;
      }

      if (!keep || totalWeight <= 0) continue;

      const cx = Math.max(1, Math.min(w - 2, Math.round(wCx / totalWeight)));
      const cy = Math.max(1, Math.min(h - 2, Math.round(wCy / totalWeight)));
      if (!strictMask[cy * w + cx]) continue;

      const xmin = Math.max(0, Math.round(wXmin / totalWeight));
      const ymin = Math.max(0, Math.round(wYmin / totalWeight));
      const xmax = Math.min(w - 1, Math.round(wXmax / totalWeight));
      const ymax = Math.min(h - 1, Math.round(wYmax / totalWeight));
      const radiusPx = Math.max(4, Math.round((xmax - xmin + (ymax - ymin)) / 4));

      // Prefer semantic AI label when available, otherwise weighted majority vote
      const className: TreeClassName =
        geminiVote?.box.className ??
        yoloVote?.box.className ??
        (Object.entries(classScores).sort((a, b) => b[1] - a[1])[0][0] as TreeClassName);
      const classId: 0 | 1 | 2 =
        className === 'palm_tree' ? 1 : className === 'windbreak_tree' ? 2 : 0;

      const health: TreeHealthStatus =
        geminiVote?.box.health ??
        (Object.entries(healthScores).sort((a, b) => b[1] - a[1])[0][0] as TreeHealthStatus);

      const multiAgreementBonus = Math.min(0.1, Math.max(0, uniqueVotes.length - 1) * 0.045);
      const fusedScore = Number(Math.min(0.98, bestScore + multiAgreementBonus).toFixed(3));
      const onBoundary = Boolean(boundaryMask[cy * w + cx]);

      const [lon, lat] =
        worldOriginX !== undefined && worldOriginY !== undefined && zoom !== undefined
          ? worldPxToLonLat(worldOriginX + cx, worldOriginY + cy, zoom)
          : syntheticLonLat(cx, cy, mpp);

      candidateFused.push({
        id: 0,
        xmin,
        ymin,
        xmax,
        ymax,
        cx,
        cy,
        radiusPx,
        score: fusedScore,
        classId,
        className,
        health,
        onBoundary,
        shadowDrop: Number(bestShadow.toFixed(3)),
        laplacianVar: Number(bestLap.toFixed(1)),
        crownDiameterM: Number((radiusPx * 2 * mpp).toFixed(2)),
        lat,
        lon,
        source: 'ai',
      });
    }

    // Deduplicate any adjacent fused peaks closer than physical minCenterSepPx
    candidateFused.sort((a, b) => b.score - a.score);
    for (const c of candidateFused) {
      if (ensembleBoxes.some((k) => Math.hypot(k.cx - c.cx, k.cy - c.cy) < minCenterSepPx)) {
        continue;
      }
      ensembleBoxes.push({ ...c, id: ensembleBoxes.length + 1 });
      if (ensembleBoxes.length >= 1200) break;
    }
  }

  const ensembleSummary = summarizeLibraryResult({
    id: 'turf_consensus_ensemble',
    name: 'Geo-DBSCAN Multi-Library Ensemble',
    nameAr: 'التوافق المكاني لجميع المكتبات (Geo-DBSCAN)',
    libraryBadge: '@turf/clusters-dbscan',
    category: 'geospatial_ensemble',
    status: 'ran',
    boxes: ensembleBoxes,
    inwardBufferM,
    mpp,
    durationMs: Math.round(performance.now() - t0),
    notesAr: hasSemanticAi
      ? 'يدمج توافق الذكاء الاصطناعي ومكتبات الرؤية الحاسوبية مكانياً عبر Geo-DBSCAN مع استرداد الأشجار المؤكدة بمكتبتين فأكثر.'
      : 'يدمج نتائج المكتبات البصرية الثلاث عبر Geo-DBSCAN لتأكيد الأشجار الحقيقية واستبعاد التداخل العشوائي.',
  });

  // Compute median & agreement across active engines
  const ranCounts = ranBase.map((l) => l.count).sort((a, b) => a - b);
  const medianCount =
    ranCounts.length > 0 ? ranCounts[Math.floor(ranCounts.length / 2)] : ensembleSummary.count;

  const maxCount = Math.max(1, ...ranCounts, ensembleSummary.count);
  const meanAbsDev =
    ranCounts.length > 0
      ? ranCounts.reduce((acc, c) => acc + Math.abs(c - medianCount), 0) / ranCounts.length
      : 0;
  const agreementPct =
    ranCounts.length === 0
      ? 100
      : ranCounts.every((c) => c === 0) && ensembleSummary.count === 0
        ? 100
        : Math.max(15, Math.min(100, Math.round((1 - meanAbsDev / Math.max(4, maxCount)) * 100)));

  // Pick recommended engine: Turf Consensus Ensemble reconciles AI Vision + multi-CV evidence cleanly
  const recommendedEngineId: LibraryEngineId =
    ranBase.length >= 2
      ? 'turf_consensus_ensemble'
      : ranBase[0]?.id ?? 'morphological_watershed';

  const libraries: LibraryDetectionSummary[] = [ensembleSummary, ...baseLibraries];
  const recSummary = libraries.find((l) => l.id === recommendedEngineId) ?? ensembleSummary;

  return {
    ensembleSummary,
    report: {
      libraries,
      activeEngineId: recommendedEngineId,
      recommendedEngineId,
      recommendedCount: recSummary.count,
      consensusCount: ensembleSummary.count,
      medianCount,
      agreementPct,
    },
  };
}

// ---------------------------------------------------------------------------
// Backend Gemini AI Arbitration across All Libraries
// ---------------------------------------------------------------------------

export async function requestAiLibraryArbitration(params: {
  farmId: string;
  species: string;
  cropType: string;
  vegPct: number;
  mpp: number;
  engineerCount: number | null;
  forecast: number | null;
  inwardBufferM: number;
  highResMode: boolean;
  provider: string;
  comparison: MultiLibraryComparisonReport;
  signal?: AbortSignal;
}): Promise<AiLibraryArbitration | null> {
  const ranLibs = params.comparison.libraries.filter((l) => l.status === 'ran');
  if (ranLibs.length < 2) return null;

  try {
    const resp = await fetch('/api/detect-trees/arbitrate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDetectionAuthHeaders(),
      },
      signal: params.signal,
      body: JSON.stringify({
        farmId: params.farmId,
        species: params.species,
        cropType: params.cropType,
        vegPct: params.vegPct,
        mpp: params.mpp,
        engineerCount: params.engineerCount,
        forecast: params.forecast,
        inwardBufferM: params.inwardBufferM,
        highResMode: params.highResMode,
        provider: params.provider,
        consensusCount: params.comparison.consensusCount,
        medianCount: params.comparison.medianCount,
        agreementPct: params.comparison.agreementPct,
        libraries: params.comparison.libraries.map((l) => ({
          id: l.id,
          name: l.name,
          libraryBadge: l.libraryBadge,
          category: l.category,
          status: l.status,
          count: l.count,
          interiorCount: l.interiorCount,
          boundaryCount: l.boundaryCount,
          avgConfidence: l.avgConfidence,
          medianCrownM: l.medianCrownM,
          durationMs: l.durationMs,
        })),
      }),
    });

    if (!resp.ok) return null;
    const data = (await resp.json().catch(() => ({}))) as {
      ok?: boolean;
      arbitration?: AiLibraryArbitration;
    };
    if (!data.ok || !data.arbitration) return null;
    return data.arbitration;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    return null;
  }
}
