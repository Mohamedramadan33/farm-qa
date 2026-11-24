import { DETECTION_CONFIG } from './config';
import { fuseDetectionsWBF, type RawCandidate } from './treeDetector';
import type { TreeDetectionBox } from '../../types/qa';

export interface PolygonCountSummary {
  aiCount: number;
  aiLow: number;
  aiHigh: number;
  aiInteriorTrees: number;
  aiRaw: number;
  aiFactor: number;
  crownM: number | null;
}

/**
 * Computes clipped polygon tree counts and multi-threshold confidence intervals
 * without unvalidated species multipliers (aiFactor = 1.0).
 */
export function countTreesInPolygon(params: {
  boxes: TreeDetectionBox[];
  allThresholdCandidates: RawCandidate[];
  strictMask: Uint8Array;
  boundaryMask: Uint8Array;
  width: number;
  height: number;
  mpp: number;
  inwardBufferM: number;
  confThresh?: number;
}): PolygonCountSummary {
  const {
    boxes,
    allThresholdCandidates,
    strictMask,
    boundaryMask,
    width: w,
    height: h,
    mpp,
    inwardBufferM,
    confThresh = DETECTION_CONFIG.aiModelConfThresh,
  } = params;

  const boundaryBoxes = boxes.filter((b) => b.onBoundary);
  const interiorBoxes = boxes.filter((b) => !b.onBoundary);
  const primaryBoxes = inwardBufferM < -0.5 ? boxes : boundaryBoxes;

  const delta = DETECTION_CONFIG.confIntervalDelta;
  const minCenterDistPx = Math.max(5, Math.round(2.0 / Math.max(mpp, 0.12)));

  const countBand = (minScore: number): number => {
    if (allThresholdCandidates.length === 0) return primaryBoxes.length;
    const subset = fuseDetectionsWBF(
      allThresholdCandidates.filter((c) => c.score >= minScore),
      DETECTION_CONFIG.wbfIouThresh,
      minCenterDistPx
    );
    let total = 0;
    for (const c of subset) {
      const cx = Math.round((c.xmin + c.xmax) / 2);
      const cy = Math.round((c.ymin + c.ymax) / 2);
      if (cx < 0 || cx >= w || cy < 0 || cy >= h || !strictMask[cy * w + cx]) continue;
      if (inwardBufferM < -0.5 || boundaryMask[cy * w + cx]) {
        total++;
      }
    }
    return total;
  };

  let lo =
    allThresholdCandidates.length > 0
      ? countBand(Math.min(0.95, confThresh + delta))
      : primaryBoxes.filter((b) => b.score >= Math.min(0.95, confThresh + delta)).length;
  let hi =
    allThresholdCandidates.length > 0
      ? countBand(Math.max(0.25, confThresh - delta))
      : primaryBoxes.length;

  const mid = primaryBoxes.length;
  if (lo === mid && hi === mid && mid > 0) {
    const varianceSum = primaryBoxes.reduce((acc, b) => acc + b.score * (1 - b.score), 0);
    const margin = Math.max(1, Math.round(1.645 * Math.sqrt(varianceSum)));
    lo = Math.max(0, mid - margin);
    hi = mid + margin;
  }

  let crownM: number | null = null;
  if (primaryBoxes.length > 0) {
    const diams = primaryBoxes
      .map((b) => b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * mpp)
      .sort((a, b) => a - b);
    crownM = Number(diams[Math.floor(diams.length / 2)].toFixed(2));
  }

  return {
    aiCount: mid,
    aiLow: Math.min(lo, mid),
    aiHigh: Math.max(hi, mid),
    aiInteriorTrees: interiorBoxes.length,
    aiRaw: mid,
    aiFactor: 1.0,
    crownM,
  };
}

export function computePseudoNdviFromMosaic(params: {
  rgba: Uint8ClampedArray;
  strictMask: Uint8Array;
  width: number;
  height: number;
  vegPct: number;
  ndviSceneFound?: boolean;
}): number | null {
  const { rgba, strictMask, width: w, height: h, vegPct, ndviSceneFound } = params;
  let interiorGreen = 0;
  let interiorTotal = 0;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) continue;
    const idx = i * 4;
    const r = rgba[idx];
    const g = rgba[idx + 1];
    interiorGreen += (g - r) / (g + r + 1e-6);
    interiorTotal++;
  }
  const meanVari = interiorTotal > 0 ? interiorGreen / interiorTotal : 0;
  const estimated = Math.max(0.08, Math.min(0.78, 0.28 + meanVari * 1.45 + (vegPct / 100) * 0.22));
  if (ndviSceneFound === false && interiorTotal === 0) {
    return null;
  }
  return Number(estimated.toFixed(2));
}
