import * as ort from 'onnxruntime-web';
import { normAr, present } from '../../lib/arabicNorm';
import { DETECTION_CONFIG, TREE_CLASSES, getDetectionAuthHeaders } from './config';
import {
  computeEuclideanDistanceTransform,
  computeStage1VegetationMask,
  evaluateCrownShadowAndHealth,
  type SpectralMaps,
} from './imageProcessing';
import { worldPxToLonLat } from './tileFetcher';
import type {
  DetectorMode,
  SatelliteProgress,
  TreeClassName,
  TreeDetectionBox,
  TreeHealthStatus,
} from '../../types/qa';

export interface SahiSlice {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface RawCandidate {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  score: number;
  classId: 0 | 1 | 2;
  health?: TreeHealthStatus;
  circularity?: number;
  crownDiameterM?: number;
  lat?: number;
  lon?: number;
}

export interface ModelStatusInfo {
  geminiReady: boolean;
  onnxReady: boolean;
  onnxUrl: string | null;
  backendYoloReady: boolean;
}

export interface MorphologicalFallbackStats {
  boxes: TreeDetectionBox[];
  gate1VegPixels: number;
  gate2ComponentsRejected: number;
  gate2ActivePct: number;
  gate3BasinsProposed: number;
}

/**
 * Fallback-only heuristic class inference. Never used on trained AI paths (YOLO / Gemini).
 */
export function inferFallbackTreeClass(
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

export function generateSahiSlices(
  imgW: number,
  imgH: number,
  mask: Uint8Array,
  sliceSize = DETECTION_CONFIG.sahiTileSize,
  overlap = DETECTION_CONFIG.sahiOverlap,
  minActiveRatio = 0.05
): SahiSlice[] {
  const sW = Math.min(sliceSize, imgW);
  const sH = Math.min(sliceSize, imgH);
  const strideX = Math.max(64, Math.round(sliceSize * (1 - overlap)));
  const strideY = Math.max(64, Math.round(sliceSize * (1 - overlap)));

  const xs: number[] = [];
  for (let x = 0; x < imgW; x += strideX) {
    const clamped = Math.max(0, Math.min(x, imgW - sW));
    if (xs.length === 0 || xs[xs.length - 1] !== clamped) xs.push(clamped);
    if (clamped + sW >= imgW) break;
  }
  const ys: number[] = [];
  for (let y = 0; y < imgH; y += strideY) {
    const clamped = Math.max(0, Math.min(y, imgH - sH));
    if (ys.length === 0 || ys[ys.length - 1] !== clamped) ys.push(clamped);
    if (clamped + sH >= imgH) break;
  }

  const slices: SahiSlice[] = [];
  for (const sy of ys) {
    for (const sx of xs) {
      let activeSampled = 0;
      let totalSampled = 0;
      for (let py = sy; py < sy + sH; py += 2) {
        const rowOff = py * imgW;
        for (let px = sx; px < sx + sW; px += 2) {
          totalSampled++;
          if (mask[rowOff + px]) activeSampled++;
        }
      }
      if (totalSampled > 0 && activeSampled / totalSampled >= minActiveRatio) {
        slices.push({ x: sx, y: sy, w: sW, h: sH });
      }
    }
  }
  return slices;
}

export function boxIoU(a: RawCandidate, b: RawCandidate): number {
  const ix1 = Math.max(a.xmin, b.xmin);
  const iy1 = Math.max(a.ymin, b.ymin);
  const ix2 = Math.min(a.xmax, b.xmax);
  const iy2 = Math.min(a.ymax, b.ymax);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const areaA = Math.max(1, (a.xmax - a.xmin) * (a.ymax - a.ymin));
  const areaB = Math.max(1, (b.xmax - b.xmin) * (b.ymax - b.ymin));
  return inter / (areaA + areaB - inter);
}

/**
 * Class-aware Non-Maximum Suppression.
 * Suppresses same-class overlaps at `iouThresh` and `minCenterDistPx`, while permitting distinct
 * classes (e.g., palm_tree vs orchard_tree) unless cross-class overlap exceeds `crossClassIou`.
 */
export function mergeDetectionsNMS(
  candidates: RawCandidate[],
  iouThresh = DETECTION_CONFIG.nmsIouThresh,
  minCenterDistPx = 6,
  classAware = DETECTION_CONFIG.classAwareNms,
  crossClassIou = DETECTION_CONFIG.crossClassSuppressionIou
): RawCandidate[] {
  const ordered = [...candidates].sort((a, b) => b.score - a.score);
  const kept: RawCandidate[] = [];
  for (const cand of ordered) {
    const cx = (cand.xmin + cand.xmax) / 2;
    const cy = (cand.ymin + cand.ymax) / 2;
    const overlaps = kept.some((prev) => {
      const iou = boxIoU(cand, prev);
      const pcx = (prev.xmin + prev.xmax) / 2;
      const pcy = (prev.ymin + prev.ymax) / 2;
      const dist = Math.hypot(cx - pcx, cy - pcy);

      if (!classAware || cand.classId === prev.classId) {
        return iou >= iouThresh || dist < minCenterDistPx;
      }
      return iou >= crossClassIou;
    });
    if (!overlaps) {
      kept.push(cand);
      if (kept.length >= 900) break;
    }
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Tier 1: Gemini Spatial Vision (/api/detect-trees)
// ---------------------------------------------------------------------------

async function runGeminiVisionDetection(params: {
  maskedCanvas: HTMLCanvasElement;
  slices: SahiSlice[];
  polygonRingPx: Array<[number, number]>;
  worldOriginX: number;
  worldOriginY: number;
  zoom: number;
  mpp: number;
  highAccuracy: boolean;
  signal?: AbortSignal;
  onProgress?: (p: SatelliteProgress) => void;
}): Promise<{
  ok: boolean;
  engine: string;
  candidates: RawCandidate[];
  errorNotice?: string;
}> {
  const {
    maskedCanvas,
    slices,
    polygonRingPx,
    worldOriginX,
    worldOriginY,
    zoom,
    mpp,
    highAccuracy,
    signal,
    onProgress,
  } = params;

  if (slices.length === 0) {
    return {
      ok: false,
      engine: '',
      candidates: [],
      errorNotice: 'No active polygon tiles to analyze.',
    };
  }

  const tilePayloads = slices.map((s) => {
    const sub = document.createElement('canvas');
    sub.width = s.w;
    sub.height = s.h;
    sub.getContext('2d')!.drawImage(maskedCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
    return {
      imageBase64: sub.toDataURL('image/png'),
      offsetX: s.x,
      offsetY: s.y,
      width: s.w,
      height: s.h,
    };
  });

  const batchSize = 8;
  const allCands: RawCandidate[] = [];
  let usedEngine = highAccuracy
    ? 'AI Vision (Gemini Pro High-Accuracy Spatial)'
    : 'AI Vision (Gemini Flash Spatial)';

  for (let i = 0; i < tilePayloads.length; i += batchSize) {
    if (signal?.aborted) {
      throw new DOMException('Detection aborted by user.', 'AbortError');
    }
    const batch = tilePayloads.slice(i, i + batchSize);
    onProgress?.({
      stage: 'detect',
      loaded: i,
      total: tilePayloads.length,
      message: `AI Vision scanning tiles ${i + 1}-${Math.min(i + batch.length, tilePayloads.length)}/${tilePayloads.length}`,
    });

    try {
      const resp = await fetch('/api/detect-trees', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getDetectionAuthHeaders(),
        },
        signal,
        body: JSON.stringify({
          tiles: batch,
          polygonRingPx,
          geoTransform: { worldOriginX, worldOriginY, zoom, mpp },
          highAccuracy,
        }),
      });
      const data = (await resp.json().catch(() => ({}))) as {
        ok?: boolean;
        engine?: string;
        error?: string;
        detections?: Array<{
          xmin: number;
          ymin: number;
          xmax: number;
          ymax: number;
          label: TreeClassName;
          confidence: number;
          health: TreeHealthStatus;
          lat?: number;
          lon?: number;
        }>;
      };

      if (!resp.ok || !data.ok) {
        return {
          ok: false,
          engine: '',
          candidates: [],
          errorNotice: data.error || `AI Vision endpoint returned HTTP ${resp.status}`,
        };
      }

      if (data.engine) usedEngine = data.engine;
      for (const d of data.detections || []) {
        const classId: 0 | 1 | 2 =
          d.label === 'palm_tree' ? 1 : d.label === 'windbreak_tree' ? 2 : 0;
        const avgDiamM = (((d.xmax - d.xmin) + (d.ymax - d.ymin)) / 2) * mpp;
        allCands.push({
          xmin: d.xmin,
          ymin: d.ymin,
          xmax: d.xmax,
          ymax: d.ymax,
          score: d.confidence,
          classId,
          health: d.health,
          crownDiameterM: Number(avgDiamM.toFixed(2)),
          lat: d.lat,
          lon: d.lon,
        });
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err;
      return {
        ok: false,
        engine: '',
        candidates: [],
        errorNotice: err instanceof Error ? err.message : 'Network error calling AI Vision',
      };
    }
  }

  return { ok: true, engine: usedEngine, candidates: allCands };
}

// ---------------------------------------------------------------------------
// Tier 2: ONNX Runtime Web (WebGPU/WebGL/WASM) & FastAPI Batch YOLO
// ---------------------------------------------------------------------------

let cachedModelStatus: ModelStatusInfo | null = null;
let cachedOrtSession: ort.InferenceSession | null = null;
let ortLoadAttempted = false;

export async function getModelStatus(): Promise<ModelStatusInfo> {
  if (cachedModelStatus) return cachedModelStatus;
  try {
    const resp = await fetch('/api/satellite/model-status');
    if (resp.ok) {
      const data = (await resp.json()) as Partial<ModelStatusInfo>;
      cachedModelStatus = {
        geminiReady: Boolean(data.geminiReady),
        onnxReady: Boolean(data.onnxReady),
        onnxUrl: data.onnxUrl || null,
        backendYoloReady: Boolean(data.backendYoloReady),
      };
      return cachedModelStatus;
    }
  } catch {
    // ignore
  }
  cachedModelStatus = {
    geminiReady: false,
    onnxReady: false,
    onnxUrl: null,
    backendYoloReady: false,
  };
  return cachedModelStatus;
}

async function getOnnxSession(onnxUrl: string): Promise<ort.InferenceSession | null> {
  if (cachedOrtSession) return cachedOrtSession;
  if (ortLoadAttempted) return null;
  ortLoadAttempted = true;

  try {
    const isIsolated = typeof self !== 'undefined' && Boolean((self as { crossOriginIsolated?: boolean }).crossOriginIsolated);
    const hwThreads = typeof navigator !== 'undefined' && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 2;
    ort.env.wasm.numThreads = isIsolated ? Math.min(4, hwThreads) : 1;
  } catch {
    // ignore WASM env setup failure
  }

  const providerCandidates: string[][] = [];
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    providerCandidates.push(['webgpu', 'wasm']);
  }
  providerCandidates.push(['webgl', 'wasm'], ['wasm']);

  for (const providers of providerCandidates) {
    try {
      cachedOrtSession = await ort.InferenceSession.create(onnxUrl, {
        executionProviders: providers,
      });
      return cachedOrtSession;
    } catch {
      // try next execution provider chain
    }
  }
  return null;
}

async function runOnnxTileInference(
  session: ort.InferenceSession,
  tileCanvas: HTMLCanvasElement,
  sliceX: number,
  sliceY: number,
  confThresh: number
): Promise<RawCandidate[]> {
  const inputSize = DETECTION_CONFIG.sahiTileSize;
  const resized = document.createElement('canvas');
  resized.width = inputSize;
  resized.height = inputSize;
  const rCtx = resized.getContext('2d')!;
  rCtx.drawImage(tileCanvas, 0, 0, inputSize, inputSize);
  const rgba = rCtx.getImageData(0, 0, inputSize, inputSize).data;

  const floatData = new Float32Array(3 * inputSize * inputSize);
  const plane = inputSize * inputSize;
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    floatData[i] = rgba[o] / 255.0;
    floatData[plane + i] = rgba[o + 1] / 255.0;
    floatData[2 * plane + i] = rgba[o + 2] / 255.0;
  }

  const inputName = session.inputNames[0] || 'images';
  const tensor = new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]);
  const outputs = await session.run({ [inputName]: tensor });
  const outName = session.outputNames[0];
  const outTensor = outputs[outName];
  const raw = outTensor.data as Float32Array;
  const dims = outTensor.dims;

  const scaleX = tileCanvas.width / inputSize;
  const scaleY = tileCanvas.height / inputSize;
  const cands: RawCandidate[] = [];

  if (dims.length === 3 && dims[1] >= 5 && dims[2] > dims[1]) {
    const channels = dims[1];
    const numBoxes = dims[2];
    const numClasses = channels - 4;
    for (let i = 0; i < numBoxes; i++) {
      let bestCls = 0;
      let bestConf = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = raw[(4 + c) * numBoxes + i];
        if (score > bestConf) {
          bestConf = score;
          bestCls = c;
        }
      }
      if (bestConf < confThresh) continue;
      const cx = raw[0 * numBoxes + i] * scaleX;
      const cy = raw[1 * numBoxes + i] * scaleY;
      const bw = raw[2 * numBoxes + i] * scaleX;
      const bh = raw[3 * numBoxes + i] * scaleY;
      const clsId = (bestCls <= 2 ? bestCls : 0) as 0 | 1 | 2;
      cands.push({
        xmin: sliceX + cx - bw / 2,
        ymin: sliceY + cy - bh / 2,
        xmax: sliceX + cx + bw / 2,
        ymax: sliceY + cy + bh / 2,
        score: bestConf,
        classId: clsId,
      });
    }
  } else if (dims.length === 3 && dims[2] >= 5) {
    const numBoxes = dims[1];
    const channels = dims[2];
    const numClasses = channels - 4;
    for (let i = 0; i < numBoxes; i++) {
      const base = i * channels;
      let bestCls = 0;
      let bestConf = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = raw[base + 4 + c];
        if (score > bestConf) {
          bestConf = score;
          bestCls = c;
        }
      }
      if (bestConf < confThresh) continue;
      const cx = raw[base] * scaleX;
      const cy = raw[base + 1] * scaleY;
      const bw = raw[base + 2] * scaleX;
      const bh = raw[base + 3] * scaleY;
      const clsId = (bestCls <= 2 ? bestCls : 0) as 0 | 1 | 2;
      cands.push({
        xmin: sliceX + cx - bw / 2,
        ymin: sliceY + cy - bh / 2,
        xmax: sliceX + cx + bw / 2,
        ymax: sliceY + cy + bh / 2,
        score: bestConf,
        classId: clsId,
      });
    }
  }
  return cands;
}

async function runBackendYoloBatch(
  rawCanvas: HTMLCanvasElement,
  slices: SahiSlice[],
  confThresh: number,
  signal?: AbortSignal
): Promise<{ ok: boolean; engine: string; cands: RawCandidate[] }> {
  try {
    const tiles = slices.map((s) => {
      const sub = document.createElement('canvas');
      sub.width = s.w;
      sub.height = s.h;
      // Domain alignment: pass raw unmasked tile matching YOLO training imagery
      sub.getContext('2d')!.drawImage(rawCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
      return {
        imageBase64: sub.toDataURL('image/png'),
        offsetX: s.x,
        offsetY: s.y,
      };
    });

    const resp = await fetch('/api/satellite/detect-batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDetectionAuthHeaders(),
      },
      signal,
      body: JSON.stringify({
        tiles,
        conf: confThresh,
        iou: DETECTION_CONFIG.nmsIouThresh,
      }),
    });
    if (!resp.ok) return { ok: false, engine: 'unavailable', cands: [] };
    const data = (await resp.json()) as {
      ok?: boolean;
      engine?: string;
      boxes?: Array<{
        xmin: number;
        ymin: number;
        xmax: number;
        ymax: number;
        classId?: number;
        score?: number;
      }>;
    };
    if (!data.ok) return { ok: false, engine: data.engine || 'unavailable', cands: [] };
    const cands: RawCandidate[] = (data.boxes || [])
      .filter((b) => Number(b.score ?? 0) >= confThresh)
      .map((b) => ({
        xmin: b.xmin,
        ymin: b.ymin,
        xmax: b.xmax,
        ymax: b.ymax,
        score: Number(b.score ?? 0.55),
        classId: (b.classId === 1 || b.classId === 2 ? b.classId : 0) as 0 | 1 | 2,
      }));
    return { ok: true, engine: data.engine || 'YOLOv8 FastAPI Batch', cands };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    return { ok: false, engine: 'unavailable', cands: [] };
  }
}

// ---------------------------------------------------------------------------
// Tier 3: 4-Gate Canopy Watershed Algorithm (Classical Fallback)
// ---------------------------------------------------------------------------

const MARCHING_PERIM = [
  0,
  Math.SQRT1_2,
  Math.SQRT1_2,
  1.0,
  Math.SQRT1_2,
  1.0,
  Math.SQRT2,
  Math.SQRT1_2,
  Math.SQRT1_2,
  Math.SQRT2,
  1.0,
  Math.SQRT1_2,
  1.0,
  Math.SQRT1_2,
  Math.SQRT1_2,
  0,
];

export function runMorphologicalFourGateFallback(params: {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  mpp: number;
  species: string;
  maps: SpectralMaps;
  stage1MaskResult?: ReturnType<typeof computeStage1VegetationMask>;
}): MorphologicalFallbackStats {
  const { rgba, width: w, height: h, strictMask, boundaryMask, mpp, species, maps } = params;
  const { lum, hsvV, exg, lapMag } = maps;
  const n = w * h;

  const stage1 =
    params.stage1MaskResult ??
    computeStage1VegetationMask({ rgba, width: w, height: h, strictMask, mpp, maps });
  const { vegMask, gate1VegPixels, gate2ComponentsRejected, gate2KeptPixels, gate2ActivePct } = stage1;

  if (gate2KeptPixels === 0) {
    return {
      boxes: [],
      gate1VegPixels,
      gate2ComponentsRejected,
      gate2ActivePct,
      gate3BasinsProposed: 0,
    };
  }

  const dt = computeEuclideanDistanceTransform(vegMask, w, h);
  const minPeakDtPx = Math.max(1.8, 0.72 / Math.max(mpp, 0.12));
  const minPeakSpacingPx = Math.max(
    4,
    Math.round(DETECTION_CONFIG.minCrownSpacingM / Math.max(mpp, 0.12))
  );

  const rawPeaks: Array<{ idx: number; x: number; y: number; dist: number }> = [];
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (!vegMask[idx]) continue;
      const d = dt[idx];
      if (d < minPeakDtPx) continue;
      if (
        d >= dt[idx - 1] &&
        d >= dt[idx + 1] &&
        d >= dt[idx - w] &&
        d >= dt[idx + w] &&
        d >= dt[idx - w - 1] &&
        d >= dt[idx - w + 1] &&
        d >= dt[idx + w - 1] &&
        d >= dt[idx + w + 1]
      ) {
        rawPeaks.push({ idx, x, y, dist: d });
      }
    }
  }

  rawPeaks.sort((a, b) => b.dist - a.dist);
  const seeds: Array<{ id: number; idx: number; x: number; y: number; dist: number }> = [];
  for (const p of rawPeaks) {
    const tooClose = seeds.some((s) => Math.hypot(p.x - s.x, p.y - s.y) < minPeakSpacingPx);
    if (!tooClose) {
      seeds.push({ id: seeds.length + 1, ...p });
      if (seeds.length >= 1200) break;
    }
  }

  const basinLabel = new Int32Array(n);
  const fgPixels: number[] = [];
  for (let i = 0; i < n; i++) {
    if (vegMask[i]) fgPixels.push(i);
  }
  for (const s of seeds) {
    basinLabel[s.idx] = s.id;
  }

  fgPixels.sort((a, b) => dt[b] - dt[a]);
  for (let pass = 0; pass < 3; pass++) {
    for (const idx of fgPixels) {
      if (basinLabel[idx] !== 0) continue;
      const y = Math.floor(idx / w);
      const x = idx - y * w;
      let bestLbl = 0;
      let bestDt = -1;
      const nbs = [
        x > 0 ? idx - 1 : -1,
        x + 1 < w ? idx + 1 : -1,
        y > 0 ? idx - w : -1,
        y + 1 < h ? idx + w : -1,
      ];
      for (const nb of nbs) {
        if (nb >= 0 && basinLabel[nb] > 0 && dt[nb] > bestDt) {
          bestDt = dt[nb];
          bestLbl = basinLabel[nb];
        }
      }
      if (bestLbl > 0) basinLabel[idx] = bestLbl;
    }
  }

  interface BasinStats {
    id: number;
    pixels: number[];
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    sumX: number;
    sumY: number;
    sumXX: number;
    sumYY: number;
    sumXY: number;
  }

  const basinMap = new Map<number, BasinStats>();
  for (const idx of fgPixels) {
    const lbl = basinLabel[idx];
    if (lbl <= 0) continue;
    const y = Math.floor(idx / w);
    const x = idx - y * w;
    let st = basinMap.get(lbl);
    if (!st) {
      st = {
        id: lbl,
        pixels: [],
        minX: x,
        minY: y,
        maxX: x,
        maxY: y,
        sumX: 0,
        sumY: 0,
        sumXX: 0,
        sumYY: 0,
        sumXY: 0,
      };
      basinMap.set(lbl, st);
    }
    st.pixels.push(idx);
    if (x < st.minX) st.minX = x;
    if (x > st.maxX) st.maxX = x;
    if (y < st.minY) st.minY = y;
    if (y > st.maxY) st.maxY = y;
    st.sumX += x;
    st.sumY += y;
    st.sumXX += x * x;
    st.sumYY += y * y;
    st.sumXY += x * y;
  }

  const gate3BasinsProposed = basinMap.size;
  const verifiedBoxes: TreeDetectionBox[] = [];
  let nextBoxId = 1;

  for (const st of basinMap.values()) {
    const areaPx = st.pixels.length;
    if (areaPx < 10) continue;

    const eqDiamM = 2 * Math.sqrt(areaPx / Math.PI) * mpp;
    if (eqDiamM < DETECTION_CONFIG.minCrownDiamM || eqDiamM > DETECTION_CONFIG.maxCrownDiamM) {
      continue;
    }

    const bw = st.maxX - st.minX + 3;
    const bh = st.maxY - st.minY + 3;
    const localMask = new Uint8Array(bw * bh);
    for (const idx of st.pixels) {
      const y = Math.floor(idx / w);
      const x = idx - y * w;
      localMask[(y - st.minY + 1) * bw + (x - st.minX + 1)] = 1;
    }

    let perimeterPx = 0;
    for (let ly = 0; ly < bh - 1; ly++) {
      const r0 = ly * bw;
      const r1 = (ly + 1) * bw;
      for (let lx = 0; lx < bw - 1; lx++) {
        const state =
          (localMask[r0 + lx] ? 8 : 0) |
          (localMask[r0 + lx + 1] ? 4 : 0) |
          (localMask[r1 + lx + 1] ? 2 : 0) |
          (localMask[r1 + lx] ? 1 : 0);
        perimeterPx += MARCHING_PERIM[state];
      }
    }
    if (perimeterPx <= 1) continue;

    const circularity = (4 * Math.PI * areaPx) / (perimeterPx * perimeterPx);
    if (circularity < DETECTION_CONFIG.minWatershedCircularity) continue;

    const mx = st.sumX / areaPx;
    const my = st.sumY / areaPx;
    const cxx = st.sumXX / areaPx - mx * mx;
    const cyy = st.sumYY / areaPx - my * my;
    const cxy = st.sumXY / areaPx - mx * my;
    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
    const lam1 = trace / 2 + disc;
    const lam2 = Math.max(0.15, trace / 2 - disc);
    const aspectRatio = Math.sqrt(lam1 / lam2);
    if (aspectRatio > 2.2) continue;

    const cx = Math.round(mx);
    const cy = Math.round(my);
    if (cx < 2 || cx >= w - 2 || cy < 2 || cy >= h - 2 || !strictMask[cy * w + cx]) continue;

    const radiusPx = Math.max(4, Math.round(eqDiamM / 2 / Math.max(mpp, 0.12)));

    // Gate 4: strict cast-shadow verification applies ONLY to the classical CV fallback
    const evalRes = evaluateCrownShadowAndHealth(
      cx,
      cy,
      radiusPx,
      st.minX,
      st.minY,
      st.maxX,
      st.maxY,
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

    const onBoundary = Boolean(boundaryMask[cy * w + cx]);
    const cls = inferFallbackTreeClass(species, onBoundary, eqDiamM);
    const score = Math.min(0.95, 0.52 + circularity * 0.25 + evalRes.relShadowDrop * 0.25);

    verifiedBoxes.push({
      id: nextBoxId++,
      xmin: Math.max(1, st.minX),
      ymin: Math.max(1, st.minY),
      xmax: Math.min(w - 2, st.maxX),
      ymax: Math.min(h - 2, st.maxY),
      cx,
      cy,
      radiusPx,
      score: Number(score.toFixed(3)),
      classId: cls.classId,
      className: cls.className,
      health: evalRes.health,
      onBoundary,
      shadowDrop: Number(evalRes.relShadowDrop.toFixed(3)),
      laplacianVar: Number(evalRes.laplacianVar.toFixed(1)),
      circularity: Number(circularity.toFixed(2)),
      crownDiameterM: Number(eqDiamM.toFixed(2)),
      source: 'ai',
    });
  }

  return {
    boxes: verifiedBoxes,
    gate1VegPixels,
    gate2ComponentsRejected,
    gate2ActivePct,
    gate3BasinsProposed,
  };
}

// ---------------------------------------------------------------------------
// Hybrid 3-Tier Orchestrator
// ---------------------------------------------------------------------------

export async function detectTreesHybrid(params: {
  rawCanvas: HTMLCanvasElement;
  strictMaskedCanvas: HTMLCanvasElement;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  clippedRing: Array<[number, number]>;
  worldOriginX: number;
  worldOriginY: number;
  zoom: number;
  mpp: number;
  species: string;
  highAccuracy: boolean;
  maps: SpectralMaps;
  signal?: AbortSignal;
  onProgress?: (p: SatelliteProgress) => void;
}): Promise<{
  detectorMode: DetectorMode;
  detectorEngine: string;
  aiVisionNotice?: string;
  boxes: TreeDetectionBox[];
  allThresholdCandidates: RawCandidate[];
  stage1ActivePct?: number;
  stage1AuditRejectedAi?: number;
}> {
  const {
    rawCanvas,
    strictMaskedCanvas,
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    clippedRing,
    worldOriginX,
    worldOriginY,
    zoom,
    mpp,
    species,
    highAccuracy,
    maps,
    signal,
    onProgress,
  } = params;

  const slices = generateSahiSlices(
    w,
    h,
    strictMask,
    DETECTION_CONFIG.sahiTileSize,
    DETECTION_CONFIG.sahiOverlap,
    0.05
  );
  const modelStatus = await getModelStatus();
  const stage1Diag = computeStage1VegetationMask({
    rgba,
    width: w,
    height: h,
    strictMask,
    mpp,
    maps,
  });

  let detectorMode: DetectorMode = 'morphological';
  let detectorEngine = '4-Gate Canopy Watershed Fallback';
  let aiVisionNotice: string | undefined;
  let allThresholdCandidates: RawCandidate[] = [];
  let boxes: TreeDetectionBox[] = [];
  let stage1ActivePct: number | undefined = stage1Diag.gate2ActivePct;
  let stage1AuditRejectedAi: number | undefined;

  const confThresh = DETECTION_CONFIG.aiModelConfThresh;
  const lowConfThresh = Math.max(0.25, confThresh - DETECTION_CONFIG.confIntervalDelta);
  const minCenterDistPx = Math.max(5, Math.round(2.0 / Math.max(mpp, 0.12)));

  // Tier 1: Gemini Spatial Vision (/api/detect-trees)
  if (slices.length > 0 && modelStatus.geminiReady) {
    const geminiRes = await runGeminiVisionDetection({
      maskedCanvas: strictMaskedCanvas,
      slices,
      polygonRingPx: clippedRing,
      worldOriginX,
      worldOriginY,
      zoom,
      mpp,
      highAccuracy,
      signal,
      onProgress,
    });

    if (geminiRes.ok) {
      detectorMode = 'gemini';
      detectorEngine = geminiRes.engine;
      allThresholdCandidates = geminiRes.candidates.filter((c) => {
        const cx = Math.round((c.xmin + c.xmax) / 2);
        const cy = Math.round((c.ymin + c.ymax) / 2);
        return cx >= 0 && cx < w && cy >= 0 && cy < h && Boolean(strictMask[cy * w + cx]);
      });

      const midCands = allThresholdCandidates.filter((c) => c.score >= confThresh);
      const nms = mergeDetectionsNMS(midCands, DETECTION_CONFIG.nmsIouThresh, minCenterDistPx);

      let stage1WouldDrop = 0;
      let nextId = 1;
      for (const cand of nms) {
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        if (!stage1Diag.vegMask[cy * w + cx]) {
          stage1WouldDrop++;
        }

        const radiusPx = Math.max(4, Math.round((cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4));
        const evalRes = evaluateCrownShadowAndHealth(
          cx,
          cy,
          radiusPx,
          Math.max(0, Math.floor(cand.xmin)),
          Math.max(0, Math.floor(cand.ymin)),
          Math.min(w - 1, Math.ceil(cand.xmax)),
          Math.min(h - 1, Math.ceil(cand.ymax)),
          w,
          h,
          rgba,
          strictMask,
          maps.lum,
          maps.hsvV,
          maps.exg,
          maps.lapMag
        );

        const onBoundary = Boolean(boundaryMask[cy * w + cx]);
        const className = TREE_CLASSES[cand.classId] || 'orchard_tree';
        const diamM = cand.crownDiameterM ?? Number((radiusPx * 2 * mpp).toFixed(2));
        const [lon, lat] =
          cand.lon !== undefined && cand.lat !== undefined
            ? [cand.lon, cand.lat]
            : worldPxToLonLat(worldOriginX + cx, worldOriginY + cy, zoom);

        boxes.push({
          id: nextId++,
          xmin: Math.round(cand.xmin),
          ymin: Math.round(cand.ymin),
          xmax: Math.round(cand.xmax),
          ymax: Math.round(cand.ymax),
          cx,
          cy,
          radiusPx,
          score: Number(cand.score.toFixed(3)),
          classId: cand.classId,
          className,
          health: cand.health || evalRes?.health || 'healthy',
          onBoundary,
          shadowDrop: Number((evalRes?.relShadowDrop ?? 0).toFixed(3)),
          laplacianVar: Number((evalRes?.laplacianVar ?? 0).toFixed(1)),
          crownDiameterM: diamM,
          lat,
          lon,
          source: 'ai',
        });
      }
      stage1AuditRejectedAi = stage1WouldDrop;
    } else if (geminiRes.errorNotice) {
      aiVisionNotice = geminiRes.errorNotice;
    }
  } else if (!modelStatus.geminiReady) {
    aiVisionNotice =
      'GEMINI_API_KEY غير مفعّل في الخادم — تم تشغيل وضع التحليل المورفولوجي المحلي (4-Gate Watershed).';
  }

  // Tier 2: ONNX Runtime Web / FastAPI Batch YOLO (raw unmasked tiles -> strictMask post-filter)
  if (detectorMode !== 'gemini' && slices.length > 0) {
    let yoloCandidates: RawCandidate[] = [];
    let usedYolo = false;

    if (modelStatus.onnxReady && modelStatus.onnxUrl) {
      const session = await getOnnxSession(modelStatus.onnxUrl);
      if (session) {
        for (let i = 0; i < slices.length; i++) {
          if (signal?.aborted) throw new DOMException('Detection aborted.', 'AbortError');
          const s = slices[i];
          onProgress?.({
            stage: 'detect',
            loaded: i + 1,
            total: slices.length,
            message: `ONNX running tile ${i + 1}/${slices.length}`,
          });
          const subCanvas = document.createElement('canvas');
          subCanvas.width = s.w;
          subCanvas.height = s.h;
          subCanvas.getContext('2d')!.drawImage(rawCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
          const tileCands = await runOnnxTileInference(session, subCanvas, s.x, s.y, lowConfThresh);
          yoloCandidates.push(...tileCands);
        }
        usedYolo = true;
        detectorMode = 'yolo';
        detectorEngine = `YOLOv8 ONNX (SAHI ${DETECTION_CONFIG.sahiTileSize}px · Conf>=${confThresh})`;
      }
    }

    if (!usedYolo && modelStatus.backendYoloReady) {
      for (let i = 0; i < slices.length; i += 12) {
        if (signal?.aborted) throw new DOMException('Detection aborted.', 'AbortError');
        onProgress?.({
          stage: 'detect',
          loaded: i,
          total: slices.length,
          message: `YOLO FastAPI batch ${i + 1}-${Math.min(i + 12, slices.length)}/${slices.length}`,
        });
        const batchRes = await runBackendYoloBatch(
          rawCanvas,
          slices.slice(i, i + 12),
          lowConfThresh,
          signal
        );
        if (!batchRes.ok) {
          yoloCandidates = [];
          usedYolo = false;
          break;
        }
        usedYolo = true;
        detectorMode = 'yolo';
        detectorEngine = `${batchRes.engine} (SAHI ${DETECTION_CONFIG.sahiTileSize}px · Conf>=${confThresh})`;
        yoloCandidates.push(...batchRes.cands);
      }
    }

    if (usedYolo) {
      // Strictly apply polygon mask ONLY as a post-processing spatial filter
      allThresholdCandidates = yoloCandidates.filter((cand) => {
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        return cx >= 0 && cx < w && cy >= 0 && cy < h && Boolean(strictMask[cy * w + cx]);
      });

      const midCands = allThresholdCandidates.filter((c) => c.score >= confThresh);
      const nms = mergeDetectionsNMS(midCands, DETECTION_CONFIG.nmsIouThresh, minCenterDistPx);

      let stage1WouldDrop = 0;
      let nextId = 1;
      for (const cand of nms) {
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        if (!stage1Diag.vegMask[cy * w + cx]) {
          stage1WouldDrop++;
        }

        const radiusPx = Math.max(4, Math.round((cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4));
        const evalRes = evaluateCrownShadowAndHealth(
          cx,
          cy,
          radiusPx,
          Math.max(0, Math.floor(cand.xmin)),
          Math.max(0, Math.floor(cand.ymin)),
          Math.min(w - 1, Math.ceil(cand.xmax)),
          Math.min(h - 1, Math.ceil(cand.ymax)),
          w,
          h,
          rgba,
          strictMask,
          maps.lum,
          maps.hsvV,
          maps.exg,
          maps.lapMag
        );
        const onBoundary = Boolean(boundaryMask[cy * w + cx]);
        const [lon, lat] = worldPxToLonLat(worldOriginX + cx, worldOriginY + cy, zoom);

        // Single source of truth: keep YOLO predicted classId (no species-text override),
        // and bypass shadow-drop / Laplacian-var rejection heuristics on AI detections.
        boxes.push({
          id: nextId++,
          xmin: Math.round(cand.xmin),
          ymin: Math.round(cand.ymin),
          xmax: Math.round(cand.xmax),
          ymax: Math.round(cand.ymax),
          cx,
          cy,
          radiusPx,
          score: Number(cand.score.toFixed(3)),
          classId: cand.classId,
          className: TREE_CLASSES[cand.classId] || 'orchard_tree',
          health: evalRes?.health || 'healthy',
          onBoundary,
          shadowDrop: Number((evalRes?.relShadowDrop ?? 0).toFixed(3)),
          laplacianVar: Number((evalRes?.laplacianVar ?? 0).toFixed(1)),
          crownDiameterM: Number((radiusPx * 2 * mpp).toFixed(2)),
          lat,
          lon,
          source: 'ai',
        });
      }
      stage1AuditRejectedAi = stage1WouldDrop;
    }
  }

  // Tier 3: Classical 4-Gate Morphological Fallback
  if (detectorMode === 'morphological') {
    const morph = runMorphologicalFourGateFallback({
      rgba,
      width: w,
      height: h,
      strictMask,
      boundaryMask,
      mpp,
      species,
      maps,
      stage1MaskResult: stage1Diag,
    });
    stage1ActivePct = morph.gate2ActivePct;
    boxes = morph.boxes.map((b) => {
      const [lon, lat] = worldPxToLonLat(worldOriginX + b.cx, worldOriginY + b.cy, zoom);
      return { ...b, lat, lon };
    });
  }

  if (stage1AuditRejectedAi !== undefined && stage1AuditRejectedAi > 0) {
    console.info(
      `[TreeDetector] Stage-1 diagnostic audit: ${stage1AuditRejectedAi}/${boxes.length} AI detections sit outside Stage-1 mask (kept via decoupled AI path).`
    );
  }

  return {
    detectorMode,
    detectorEngine,
    aiVisionNotice,
    boxes,
    allThresholdCandidates,
    stage1ActivePct,
    stage1AuditRejectedAi,
  };
}
