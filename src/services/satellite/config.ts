import detectionConfigJson from '../../../config/detection.config.json';
import type { TreeClassName } from '../../types/qa';

export interface DetectionConfig {
  version: string;
  classes: TreeClassName[];
  aiModelConfThresh: number;
  confIntervalDelta: number;
  sahiTileSize: number;
  sahiOverlap: number;
  highResTileSize: number;
  highResOverlap: number;
  highAccuracyTileSize: number;
  highAccuracyOverlap: number;
  nmsIouThresh: number;
  wbfIouThresh: number;
  classAwareNms: boolean;
  crossClassSuppressionIou: number;
  minCastShadowDrop: number;
  minWatershedCircularity: number;
  minCrownDiamM: number;
  maxCrownDiamM: number;
  cropSheetAreaM2: number;
  minCrownSpacingM: number;
  smallCrownPxRange: [number, number];
  fallbackImgSz: number;
}

export const DETECTION_CONFIG: DetectionConfig = {
  version: detectionConfigJson.version,
  classes: detectionConfigJson.classes as TreeClassName[],
  aiModelConfThresh: detectionConfigJson.aiModelConfThresh,
  confIntervalDelta: detectionConfigJson.confIntervalDelta,
  sahiTileSize: detectionConfigJson.sahiTileSize,
  sahiOverlap: detectionConfigJson.sahiOverlap,
  highResTileSize: detectionConfigJson.highResTileSize ?? 288,
  highResOverlap: detectionConfigJson.highResOverlap ?? 0.4,
  highAccuracyTileSize: detectionConfigJson.highAccuracyTileSize ?? 288,
  highAccuracyOverlap: detectionConfigJson.highAccuracyOverlap ?? 0.4,
  nmsIouThresh: detectionConfigJson.nmsIouThresh,
  wbfIouThresh: detectionConfigJson.wbfIouThresh ?? 0.42,
  classAwareNms: detectionConfigJson.classAwareNms,
  crossClassSuppressionIou: detectionConfigJson.crossClassSuppressionIou,
  minCastShadowDrop: detectionConfigJson.minCastShadowDrop,
  minWatershedCircularity: detectionConfigJson.minWatershedCircularity,
  minCrownDiamM: detectionConfigJson.minCrownDiamM,
  maxCrownDiamM: detectionConfigJson.maxCrownDiamM,
  cropSheetAreaM2: detectionConfigJson.cropSheetAreaM2,
  minCrownSpacingM: detectionConfigJson.minCrownSpacingM,
  smallCrownPxRange: detectionConfigJson.smallCrownPxRange as [number, number],
  fallbackImgSz: detectionConfigJson.fallbackImgSz,
};

export const TREE_CLASSES: TreeClassName[] = DETECTION_CONFIG.classes;

// In-memory Bearer token only (never persisted to localStorage or sessionStorage)
let inMemoryDetectionToken: string | null = null;

export function setDetectionAuthToken(token: string | null): void {
  inMemoryDetectionToken = token ? token.trim() : null;
}

export function getDetectionAuthHeaders(): Record<string, string> {
  if (!inMemoryDetectionToken) return {};
  return { Authorization: `Bearer ${inMemoryDetectionToken}` };
}
