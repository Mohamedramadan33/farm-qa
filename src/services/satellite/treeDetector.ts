import * as ort from 'onnxruntime-web';
import { normAr, present } from '../../lib/arabicNorm';
import { DETECTION_CONFIG, TREE_CLASSES, getDetectionAuthHeaders } from './config';
import {
  computeEuclideanDistanceTransform,
  computeStage1VegetationMask,
  evaluateCrownShadowAndHealth,
  verifyAndScoreAiCandidate,
  type SpectralMaps,
} from './imageProcessing';
import { worldPxToLonLat } from './tileFetcher';
import {
  buildConsensusEnsemble,
  runMultiScaleDoGDetector,
  runShadowTexturePairDetector,
  summarizeLibraryResult,
} from './multiLibraryEngine';
import type {
  DetectorMode,
  LibraryDetectionSummary,
  MultiLibraryComparisonReport,
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

export interface SquareTilePaddingSpec {
  targetSize: number;
  srcX: number;
  srcY: number;
  copyW: number;
  copyH: number;
  padRight: number;
  padBottom: number;
}

export function computeSquareTilePadding(
  slice: SahiSlice,
  targetSize: number
): SquareTilePaddingSpec {
  const safeTarget = Math.max(64, Math.round(targetSize));
  const copyW = Math.max(1, Math.min(safeTarget, Math.round(slice.w)));
  const copyH = Math.max(1, Math.min(safeTarget, Math.round(slice.h)));
  return {
    targetSize: safeTarget,
    srcX: slice.x,
    srcY: slice.y,
    copyW,
    copyH,
    padRight: safeTarget - copyW,
    padBottom: safeTarget - copyH,
  };
}

/**
 * Copies a SAHI slice onto a 1:1 square `targetSize x targetSize` canvas without stretching.
 * Small parcels or edge slices are zero-stretch padded with `#0f172a` to prevent crop scale distortion.
 */
export function extractSquarePaddedTile(
  srcCanvas: HTMLCanvasElement,
  slice: SahiSlice,
  targetSize: number
): HTMLCanvasElement {
  const spec = computeSquareTilePadding(slice, targetSize);
  const sub = document.createElement('canvas');
  sub.width = spec.targetSize;
  sub.height = spec.targetSize;
  const ctx = sub.getContext('2d')!;
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, spec.targetSize, spec.targetSize);
  ctx.drawImage(
    srcCanvas,
    spec.srcX,
    spec.srcY,
    spec.copyW,
    spec.copyH,
    0,
    0,
    spec.copyW,
    spec.copyH
  );
  return sub;
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
 * Class-aware Consensus Weighted Box Fusion (WBF).
 * Quadratic confidence-weighting tightens crown centers across SAHI tile seams while
 * rewarding candidates confirmed across multiple overlapping SAHI windows.
 */
export function fuseDetectionsWBF(
  candidates: RawCandidate[],
  iouThresh = DETECTION_CONFIG.wbfIouThresh,
  minCenterDistPx = 5,
  classAware = DETECTION_CONFIG.classAwareNms,
  crossClassIou = DETECTION_CONFIG.crossClassSuppressionIou
): RawCandidate[] {
  const ordered = [...candidates].sort((a, b) => b.score - a.score);
  interface Cluster {
    wSum: number;
    maxScore: number;
    votes: number;
    wxmin: number;
    wymin: number;
    wxmax: number;
    wymax: number;
    classId: 0 | 1 | 2;
    health?: TreeHealthStatus;
    circularity?: number;
    crownDiameterM?: number;
    lat?: number;
    lon?: number;
    rep: RawCandidate;
  }

  const clusters: Cluster[] = [];
  for (const cand of ordered) {
    const cx = (cand.xmin + cand.xmax) / 2;
    const cy = (cand.ymin + cand.ymax) / 2;
    let target: Cluster | null = null;

    for (const cl of clusters) {
      const iou = boxIoU(cand, cl.rep);
      const pcx = (cl.rep.xmin + cl.rep.xmax) / 2;
      const pcy = (cl.rep.ymin + cl.rep.ymax) / 2;
      const dist = Math.hypot(cx - pcx, cy - pcy);

      if (!classAware || cand.classId === cl.classId) {
        if (iou >= iouThresh || dist < minCenterDistPx) {
          target = cl;
          break;
        }
      } else if (iou >= crossClassIou) {
        target = cl;
        break;
      }
    }

    const w = Math.max(0.04, cand.score * cand.score);
    if (!target) {
      clusters.push({
        wSum: w,
        maxScore: cand.score,
        votes: 1,
        wxmin: cand.xmin * w,
        wymin: cand.ymin * w,
        wxmax: cand.xmax * w,
        wymax: cand.ymax * w,
        classId: cand.classId,
        health: cand.health,
        circularity: cand.circularity,
        crownDiameterM: cand.crownDiameterM,
        lat: cand.lat,
        lon: cand.lon,
        rep: { ...cand },
      });
      if (clusters.length >= 1200) break;
    } else {
      if (cand.score > target.maxScore) {
        target.classId = cand.classId;
        target.health = cand.health ?? target.health;
        target.circularity = cand.circularity ?? target.circularity;
        target.crownDiameterM = cand.crownDiameterM ?? target.crownDiameterM;
        target.lat = cand.lat ?? target.lat;
        target.lon = cand.lon ?? target.lon;
      }
      target.votes += 1;
      target.wSum += w;
      target.maxScore = Math.max(target.maxScore, cand.score);
      target.wxmin += cand.xmin * w;
      target.wymin += cand.ymin * w;
      target.wxmax += cand.xmax * w;
      target.wymax += cand.ymax * w;
      target.rep = {
        xmin: target.wxmin / target.wSum,
        ymin: target.wymin / target.wSum,
        xmax: target.wxmax / target.wSum,
        ymax: target.wymax / target.wSum,
        score: target.maxScore,
        classId: target.classId,
      };
    }
  }

  return clusters.map((cl) => {
    const consensusBoost = Math.min(0.08, Math.max(0, cl.votes - 1) * 0.04);
    const calibratedScore = Number(Math.min(0.98, cl.maxScore + consensusBoost).toFixed(3));
    return {
      xmin: cl.rep.xmin,
      ymin: cl.rep.ymin,
      xmax: cl.rep.xmax,
      ymax: cl.rep.ymax,
      score: calibratedScore,
      classId: cl.classId,
      health: cl.health,
      circularity: cl.circularity,
      crownDiameterM: cl.crownDiameterM,
      lat: cl.lat,
      lon: cl.lon,
    };
  });
}

export const mergeDetectionsNMS = fuseDetectionsWBF;

// ---------------------------------------------------------------------------
// Tier 1: Gemini Spatial Vision (/api/detect-trees)
// ---------------------------------------------------------------------------

async function runGeminiVisionDetection(params: {
  visionCanvas: HTMLCanvasElement;
  slices: SahiSlice[];
  targetTileSize: number;
  polygonRingPx: Array<[number, number]>;
  worldOriginX: number;
  worldOriginY: number;
  zoom: number;
  mpp: number;
  highRes: boolean;
  confThresh: number;
  minConf: number;
  signal?: AbortSignal;
  onProgress?: (p: SatelliteProgress) => void;
}): Promise<{
  ok: boolean;
  engine: string;
  candidates: RawCandidate[];
  errorNotice?: string;
}> {
  const {
    visionCanvas,
    slices,
    targetTileSize,
    polygonRingPx,
    worldOriginX,
    worldOriginY,
    zoom,
    mpp,
    highRes,
    confThresh,
    minConf,
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
    const sub = extractSquarePaddedTile(visionCanvas, s, targetTileSize);
    return {
      imageBase64: sub.toDataURL('image/png'),
      offsetX: s.x,
      offsetY: s.y,
      width: targetTileSize,
      height: targetTileSize,
    };
  });

  const batchSize = 8;
  const allCands: RawCandidate[] = [];
  let usedEngine = highRes
    ? 'AI Vision (Gemini 4 Hydrogen · High-Res SAHI WBF)'
    : 'AI Vision (Gemini 4 Hydrogen · Spatial WBF)';

  for (let i = 0; i < tilePayloads.length; i += batchSize) {
    if (signal?.aborted) {
      throw new DOMException('Detection aborted by user.', 'AbortError');
    }
    const batch = tilePayloads.slice(i, i + batchSize);
    onProgress?.({
      stage: 'detect',
      loaded: i,
      total: tilePayloads.length,
      message: `${highRes ? 'High-Res ' : ''}AI Vision scanning tiles ${i + 1}-${Math.min(i + batch.length, tilePayloads.length)}/${tilePayloads.length}`,
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
          highRes,
          highAccuracy: highRes,
          confThresh,
          minConf,
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
  rCtx.fillStyle = '#0f172a';
  rCtx.fillRect(0, 0, inputSize, inputSize);

  // Preserve 1:1 aspect ratio via uniform scaling + dark padding (never stretch rectangular tiles)
  const uniformScale = Math.min(
    inputSize / Math.max(1, tileCanvas.width),
    inputSize / Math.max(1, tileCanvas.height)
  );
  const drawW = Math.max(1, Math.round(tileCanvas.width * uniformScale));
  const drawH = Math.max(1, Math.round(tileCanvas.height * uniformScale));
  rCtx.drawImage(tileCanvas, 0, 0, tileCanvas.width, tileCanvas.height, 0, 0, drawW, drawH);
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

  const invScale = 1 / Math.max(uniformScale, 1e-6);
  const scaleX = invScale;
  const scaleY = invScale;
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
  targetTileSize: number,
  confThresh: number,
  signal?: AbortSignal
): Promise<{ ok: boolean; engine: string; cands: RawCandidate[] }> {
  try {
    const tiles = slices.map((s) => {
      // Domain alignment: pass raw unmasked 1:1 padded square tile matching YOLO training imagery
      const sub = extractSquarePaddedTile(rawCanvas, s, targetTileSize);
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
// Multi-Library Hybrid Orchestrator + Geospatial Consensus Ensemble
// ---------------------------------------------------------------------------

export async function detectTreesHybrid(params: {
  rawCanvas: HTMLCanvasElement;
  visionCanvas: HTMLCanvasElement;
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
  highRes?: boolean;
  highAccuracy?: boolean;
  confThresh?: number;
  inwardBufferM?: number;
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
  highResMode: boolean;
  sahiTileSizeUsed: number;
  sahiOverlapUsed: number;
  sahiTilesCount: number;
  libraryComparison: MultiLibraryComparisonReport;
}> {
  const {
    rawCanvas,
    visionCanvas,
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
    highRes,
    highAccuracy,
    confThresh: reqConfThresh,
    inwardBufferM = -1.5,
    maps,
    signal,
    onProgress,
  } = params;

  const isHighRes = Boolean(highRes ?? highAccuracy);
  const activeSliceSize = isHighRes
    ? DETECTION_CONFIG.highResTileSize
    : DETECTION_CONFIG.sahiTileSize;
  const activeOverlap = isHighRes
    ? DETECTION_CONFIG.highResOverlap
    : DETECTION_CONFIG.sahiOverlap;
  const minActiveRatio = isHighRes ? 0.03 : 0.05;

  const slices = generateSahiSlices(
    w,
    h,
    strictMask,
    activeSliceSize,
    activeOverlap,
    minActiveRatio
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

  const confThresh = Math.max(
    0.3,
    Math.min(0.85, reqConfThresh ?? DETECTION_CONFIG.aiModelConfThresh)
  );
  const lowConfThresh = Math.max(0.25, confThresh - DETECTION_CONFIG.confIntervalDelta);
  const minCenterDistPx = Math.max(
    3,
    Math.round((isHighRes ? 1.3 : 1.6) / Math.max(mpp, 0.1))
  );

  const stage1ActivePct: number | undefined = stage1Diag.gate2ActivePct;
  const baseLibraries: LibraryDetectionSummary[] = [];

  let geminiBoxes: TreeDetectionBox[] = [];
  let geminiCandidates: RawCandidate[] = [];
  let geminiEngineLabel = '';
  let aiVisionNotice: string | undefined;
  let stage1AuditRejectedAi: number | undefined;

  // -------------------------------------------------------------------------
  // Library 1: @google/genai (Gemini 4 Hydrogen Spatial Vision SAHI)
  // -------------------------------------------------------------------------
  const tGemini = performance.now();
  if (slices.length > 0 && modelStatus.geminiReady) {
    const geminiRes = await runGeminiVisionDetection({
      visionCanvas,
      slices,
      targetTileSize: activeSliceSize,
      polygonRingPx: clippedRing,
      worldOriginX,
      worldOriginY,
      zoom,
      mpp,
      highRes: isHighRes,
      confThresh,
      minConf: lowConfThresh,
      signal,
      onProgress,
    });

    if (geminiRes.ok) {
      geminiEngineLabel = `${geminiRes.engine} (${activeSliceSize}px · ${Math.round(activeOverlap * 100)}% ov · Conf>=${confThresh.toFixed(2)})`;

      const insidePolygon = geminiRes.candidates.filter((c) => {
        const cx = Math.round((c.xmin + c.xmax) / 2);
        const cy = Math.round((c.ymin + c.ymax) / 2);
        return cx >= 0 && cx < w && cy >= 0 && cy < h && Boolean(strictMask[cy * w + cx]);
      });

      const fusedAll = fuseDetectionsWBF(
        insidePolygon,
        DETECTION_CONFIG.wbfIouThresh,
        minCenterDistPx
      );

      const verifiedAll: Array<{
        cand: RawCandidate;
        evalRes: ReturnType<typeof evaluateCrownShadowAndHealth>;
      }> = [];
      for (const cand of fusedAll) {
        const check = verifyAndScoreAiCandidate({
          xmin: cand.xmin,
          ymin: cand.ymin,
          xmax: cand.xmax,
          ymax: cand.ymax,
          aiScore: cand.score,
          width: w,
          height: h,
          rgba,
          strictMask,
          maps,
        });
        if (!check.keep || check.adjustedScore < lowConfThresh) continue;
        verifiedAll.push({
          cand: { ...cand, score: check.adjustedScore },
          evalRes: check.evalRes,
        });
      }

      geminiCandidates = verifiedAll.map((v) => v.cand);

      let stage1WouldDrop = 0;
      let nextId = 1;
      for (const { cand, evalRes } of verifiedAll) {
        if (cand.score < confThresh) continue;
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        if (!stage1Diag.vegMask[cy * w + cx]) {
          stage1WouldDrop++;
        }

        const radiusPx = Math.max(
          4,
          Math.round((cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4)
        );
        const onBoundary = Boolean(boundaryMask[cy * w + cx]);
        const className = TREE_CLASSES[cand.classId] || 'orchard_tree';
        const diamM = cand.crownDiameterM ?? Number((radiusPx * 2 * mpp).toFixed(2));
        const [lon, lat] =
          cand.lon !== undefined && cand.lat !== undefined
            ? [cand.lon, cand.lat]
            : worldPxToLonLat(worldOriginX + cx, worldOriginY + cy, zoom);

        geminiBoxes.push({
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

      baseLibraries.push(
        summarizeLibraryResult({
          id: 'gemini_vision',
          name: 'Gemini 4 Hydrogen Spatial Vision',
          nameAr: 'رؤية الذكاء الاصطناعي (Gemini 4 Hydrogen)',
          libraryBadge: '@google/genai',
          category: 'ai_vision',
          status: 'ran',
          boxes: geminiBoxes,
          inwardBufferM,
          mpp,
          durationMs: Math.round(performance.now() - tGemini),
          notesAr: isHighRes
            ? `تحليل دلالي فائق الدقة (${activeSliceSize}px · تداخل ${Math.round(activeOverlap * 100)}% + تفكير عميق).`
            : `تحليل دلالي بالذكاء الاصطناعي يفرّق بين تيجان الأشجار والمحاصيل المسطحة.`,
        })
      );
    } else {
      aiVisionNotice = geminiRes.errorNotice;
      baseLibraries.push(
        summarizeLibraryResult({
          id: 'gemini_vision',
          name: 'Gemini 4 Hydrogen Spatial Vision',
          nameAr: 'رؤية الذكاء الاصطناعي (Gemini 4 Hydrogen)',
          libraryBadge: '@google/genai',
          category: 'ai_vision',
          status: 'error',
          unavailableReason: geminiRes.errorNotice || 'تعذر الاتصال بنموذج Gemini',
          boxes: [],
          inwardBufferM,
          mpp,
          durationMs: Math.round(performance.now() - tGemini),
          notesAr: geminiRes.errorNotice || 'تعذر تشغيل نموذج Gemini في هذه الجولة.',
        })
      );
    }
  } else {
    const reason = !modelStatus.geminiReady
      ? 'GEMINI_API_KEY غير مفعّل في الخادم'
      : 'لا توجد مربعات صالحة داخل المضلع';
    if (!modelStatus.geminiReady) {
      aiVisionNotice =
        'GEMINI_API_KEY غير مفعّل في الخادم — تم تشغيل محركات التحليل المكاني والبصري والمقارنة بين المكتبات.';
    }
    baseLibraries.push(
      summarizeLibraryResult({
        id: 'gemini_vision',
        name: 'Gemini 4 Hydrogen Spatial Vision',
        nameAr: 'رؤية الذكاء الاصطناعي (Gemini 4 Hydrogen)',
        libraryBadge: '@google/genai',
        category: 'ai_vision',
        status: 'unavailable',
        unavailableReason: reason,
        boxes: [],
        inwardBufferM,
        mpp,
        durationMs: 0,
        notesAr: reason,
      })
    );
  }

  // -------------------------------------------------------------------------
  // Library 2: ONNX Runtime Web / FastAPI Batch YOLOv8 (runs whenever weights loaded)
  // -------------------------------------------------------------------------
  const tYolo = performance.now();
  let yoloBoxes: TreeDetectionBox[] = [];
  let yoloCandidates: RawCandidate[] = [];
  let yoloEngineLabel = '';
  let yoloRan = false;

  if (slices.length > 0 && (modelStatus.onnxReady || modelStatus.backendYoloReady)) {
    const yoloSlices = slices;
    let rawYoloCands: RawCandidate[] = [];

    if (modelStatus.onnxReady && modelStatus.onnxUrl) {
      const session = await getOnnxSession(modelStatus.onnxUrl);
      if (session) {
        for (let i = 0; i < yoloSlices.length; i++) {
          if (signal?.aborted) throw new DOMException('Detection aborted.', 'AbortError');
          const s = yoloSlices[i];
          onProgress?.({
            stage: 'detect',
            loaded: i + 1,
            total: yoloSlices.length,
            message: `YOLOv8 ONNX scanning tile ${i + 1}/${yoloSlices.length}`,
          });
          const subCanvas = extractSquarePaddedTile(rawCanvas, s, activeSliceSize);
          const tileCands = await runOnnxTileInference(session, subCanvas, s.x, s.y, lowConfThresh);
          rawYoloCands.push(...tileCands);
        }
        yoloRan = true;
        yoloEngineLabel = `YOLOv8 ONNX (${isHighRes ? 'High-Res ' : ''}SAHI ${activeSliceSize}px)`;
      }
    }

    if (!yoloRan && modelStatus.backendYoloReady) {
      for (let i = 0; i < yoloSlices.length; i += 12) {
        if (signal?.aborted) throw new DOMException('Detection aborted.', 'AbortError');
        onProgress?.({
          stage: 'detect',
          loaded: i,
          total: yoloSlices.length,
          message: `YOLO FastAPI batch ${i + 1}-${Math.min(i + 12, yoloSlices.length)}/${yoloSlices.length}`,
        });
        const batchRes = await runBackendYoloBatch(
          rawCanvas,
          yoloSlices.slice(i, i + 12),
          activeSliceSize,
          lowConfThresh,
          signal
        );
        if (!batchRes.ok) {
          rawYoloCands = [];
          yoloRan = false;
          break;
        }
        yoloRan = true;
        yoloEngineLabel = `${batchRes.engine} (${isHighRes ? 'High-Res ' : ''}SAHI ${activeSliceSize}px)`;
        rawYoloCands.push(...batchRes.cands);
      }
    }

    if (yoloRan) {
      const insidePolygon = rawYoloCands.filter((cand) => {
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        return cx >= 0 && cx < w && cy >= 0 && cy < h && Boolean(strictMask[cy * w + cx]);
      });

      const fusedAll = fuseDetectionsWBF(
        insidePolygon,
        DETECTION_CONFIG.wbfIouThresh,
        minCenterDistPx
      );

      const verifiedAll: Array<{
        cand: RawCandidate;
        evalRes: ReturnType<typeof evaluateCrownShadowAndHealth>;
      }> = [];
      for (const cand of fusedAll) {
        const check = verifyAndScoreAiCandidate({
          xmin: cand.xmin,
          ymin: cand.ymin,
          xmax: cand.xmax,
          ymax: cand.ymax,
          aiScore: cand.score,
          width: w,
          height: h,
          rgba,
          strictMask,
          maps,
        });
        if (!check.keep || check.adjustedScore < lowConfThresh) continue;
        verifiedAll.push({
          cand: { ...cand, score: check.adjustedScore },
          evalRes: check.evalRes,
        });
      }

      yoloCandidates = verifiedAll.map((v) => v.cand);

      let nextId = 1;
      for (const { cand, evalRes } of verifiedAll) {
        if (cand.score < confThresh) continue;
        const cx = Math.round((cand.xmin + cand.xmax) / 2);
        const cy = Math.round((cand.ymin + cand.ymax) / 2);
        const radiusPx = Math.max(
          4,
          Math.round((cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4)
        );
        const onBoundary = Boolean(boundaryMask[cy * w + cx]);
        const [lon, lat] = worldPxToLonLat(worldOriginX + cx, worldOriginY + cy, zoom);

        yoloBoxes.push({
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

      baseLibraries.push(
        summarizeLibraryResult({
          id: 'yolov8_onnx',
          name: 'YOLOv8 Tree Detector',
          nameAr: 'نموذج التعلم العميق (YOLOv8)',
          libraryBadge: modelStatus.onnxReady ? 'onnxruntime-web' : 'YOLOv8 FastAPI',
          category: 'deep_learning',
          status: 'ran',
          boxes: yoloBoxes,
          inwardBufferM,
          mpp,
          durationMs: Math.round(performance.now() - tYolo),
          notesAr: 'كشف تيجان الأشجار باستخدام أوزان YOLO المدربة على البساتين المصرية.',
        })
      );
    }
  }

  if (!yoloRan) {
    baseLibraries.push(
      summarizeLibraryResult({
        id: 'yolov8_onnx',
        name: 'YOLOv8 Tree Detector',
        nameAr: 'نموذج التعلم العميق (YOLOv8)',
        libraryBadge: 'onnxruntime-web',
        category: 'deep_learning',
        status: 'unavailable',
        unavailableReason: 'أوزان best.onnx / خدمة FastAPI YOLO غير متوفرة حالياً',
        boxes: [],
        inwardBufferM,
        mpp,
        durationMs: 0,
        notesAr: 'يتفعّل تلقائياً عند وجود ملف models/best.onnx أو تشغيل خادم YOLO_INFER_URL.',
      })
    );
  }

  // -------------------------------------------------------------------------
  // Library 3: 4-Gate Morphological EDT Watershed (Always runs for benchmark!)
  // -------------------------------------------------------------------------
  onProgress?.({
    stage: 'postprocess',
    loaded: 1,
    total: 4,
    message: 'Running 4-Gate Morphological EDT Watershed...',
  });
  const tMorph = performance.now();
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
  const morphBoxes = morph.boxes.map((b) => {
    const [lon, lat] = worldPxToLonLat(worldOriginX + b.cx, worldOriginY + b.cy, zoom);
    return { ...b, lat, lon };
  });
  baseLibraries.push(
    summarizeLibraryResult({
      id: 'morphological_watershed',
      name: '4-Gate Morphological EDT Watershed',
      nameAr: 'التحليل المورفولوجي وتقسيم الأحواض (EDT Watershed)',
      libraryBadge: 'Meijster EDT + MarchingSquares',
      category: 'cv_classical',
      status: 'ran',
      boxes: morphBoxes,
      inwardBufferM,
      mpp,
      durationMs: Math.round(performance.now() - tMorph),
      notesAr: `فصل التيجان عبر مسافة أقليدس واستبعاد حقول البرسيم المتصلة (${morph.gate2ComponentsRejected} منطقة مستبعدة).`,
    })
  );

  // -------------------------------------------------------------------------
  // Library 4: Multi-Scale DoG / Scale-Space Blob Detector (Always runs!)
  // -------------------------------------------------------------------------
  onProgress?.({
    stage: 'postprocess',
    loaded: 2,
    total: 4,
    message: 'Running Multi-Scale Scale-Space DoG Detector...',
  });
  const dogRes = runMultiScaleDoGDetector({
    rgba,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    mpp,
    species,
    maps,
    stage1MaskResult: stage1Diag,
    worldOriginX,
    worldOriginY,
    zoom,
  });
  baseLibraries.push(
    summarizeLibraryResult({
      id: 'multiscale_dog',
      name: 'Multi-Scale DoG Scale-Space',
      nameAr: 'كاشف القمم متعددة المقاييس (Scale-Space DoG)',
      libraryBadge: 'Gaussian Pyramid DoG',
      category: 'cv_classical',
      status: 'ran',
      boxes: dogRes.boxes,
      inwardBufferM,
      mpp,
      durationMs: dogRes.durationMs,
      notesAr: 'يرصد التيجان الصغيرة والمتوسطة والكبيرة عند 3 مقاييس جاوسية لتقليل دمج الأشجار المتلاصقة.',
    })
  );

  // -------------------------------------------------------------------------
  // Library 5: Cast-Shadow Azimuth Pair & Texture Detector (Always runs!)
  // -------------------------------------------------------------------------
  onProgress?.({
    stage: 'postprocess',
    loaded: 3,
    total: 4,
    message: 'Running Cast-Shadow Azimuth & Texture Pairing...',
  });
  const shadowRes = runShadowTexturePairDetector({
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
  });
  baseLibraries.push(
    summarizeLibraryResult({
      id: 'shadow_texture_pair',
      name: 'Cast-Shadow Azimuth & Texture Pair',
      nameAr: 'مطابقة الظل الاتجاهي ونسيج التاج (Shadow-Texture)',
      libraryBadge: 'Azimuthal Shadow-Crown CV',
      category: 'cv_classical',
      status: 'ran',
      boxes: shadowRes.boxes,
      inwardBufferM,
      mpp,
      durationMs: shadowRes.durationMs,
      notesAr: `يربط بؤر الظل الأرضي (${shadowRes.shadowClustersFound}) بالتيجان المضاءة لمناعة عالية ضد الحشائش.`,
    })
  );

  // -------------------------------------------------------------------------
  // Library 6: Geo-DBSCAN Multi-Library Consensus Ensemble (@turf/clusters-dbscan)
  // -------------------------------------------------------------------------
  onProgress?.({
    stage: 'postprocess',
    loaded: 4,
    total: 4,
    message: 'Fusing libraries with Geo-DBSCAN Consensus...',
  });
  const { ensembleSummary, report } = buildConsensusEnsemble({
    baseLibraries,
    mpp,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    inwardBufferM,
    confThresh,
    highRes: isHighRes,
    worldOriginX,
    worldOriginY,
    zoom,
  });

  // Determine primary detectorMode / boxes / threshold candidates
  const geminiRan = geminiEngineLabel.length > 0;
  let detectorMode: DetectorMode = 'morphological';
  let detectorEngine = `Multi-Library Consensus (${report.libraries.filter((l) => l.status === 'ran').length} engines · Geo-DBSCAN)`;
  let boxes: TreeDetectionBox[] = ensembleSummary.boxes;
  let allThresholdCandidates: RawCandidate[] = [];

  if (geminiRan) {
    detectorMode = 'gemini';
    detectorEngine = `${geminiEngineLabel} + Geo-DBSCAN Ensemble`;
    allThresholdCandidates = geminiCandidates;
  } else if (yoloRan) {
    detectorMode = 'yolo';
    detectorEngine = `${yoloEngineLabel} + Geo-DBSCAN Ensemble`;
    allThresholdCandidates = yoloCandidates;
  } else {
    detectorMode = 'morphological';
    boxes = ensembleSummary.boxes;
    allThresholdCandidates = ensembleSummary.boxes.map((b) => ({
      xmin: b.xmin,
      ymin: b.ymin,
      xmax: b.xmax,
      ymax: b.ymax,
      score: b.score,
      classId: b.classId,
      health: b.health,
      crownDiameterM: b.crownDiameterM,
      lat: b.lat,
      lon: b.lon,
    }));
  }

  if (stage1AuditRejectedAi !== undefined && stage1AuditRejectedAi > 0) {
    console.info(
      `[TreeDetector] Stage-1 diagnostic audit: ${stage1AuditRejectedAi}/${geminiBoxes.length} AI detections sit outside Stage-1 mask (kept via decoupled AI path).`
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
    highResMode: isHighRes,
    sahiTileSizeUsed: activeSliceSize,
    sahiOverlapUsed: activeOverlap,
    sahiTilesCount: slices.length,
    libraryComparison: report,
  };
}
