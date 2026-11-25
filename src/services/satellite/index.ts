import {
  bufferPolygonMeters,
  polygonBounds,
  polygonCentroid,
  shiftPolygonMeters,
} from '../../lib/geometry';
import { VEG_EXG } from '../../lib/farmQaCore';
import { renderCleanAndAnnotatedCanvases, renderComparisonCanvasUrl } from './canvasRenderer';
import { DETECTION_CONFIG } from './config';
import {
  buildContextPreservingVisionCanvas,
  buildMaskedCanvas,
  buildSharpenedVisionCanvas,
  computeLuminanceAndLaplacianAsync,
  createBoundaryCorridorMask,
  createPolygonMask,
} from './imageProcessing';
import { requestAiLibraryArbitration } from './multiLibraryEngine';
import { computePseudoNdviFromMosaic, countTreesInPolygon } from './polygonCounter';
import { fetchProviderMosaic } from './tileFetcher';
import { detectTreesHybrid } from './treeDetector';
import type {
  BasemapProvider,
  BasemapViewMode,
  PointLonLat,
  SatelliteAnalysisResult,
  SatelliteProgress,
  TreeDetectionBox,
} from '../../types/qa';

export async function analyzeFarmSatellite(params: {
  farmId?: string;
  poly: PointLonLat[];
  species: string;
  cropType?: string;
  engineerCount?: number | null;
  forecast?: number | null;
  submissionTime: string;
  useNdvi: boolean;
  providerMode?: BasemapViewMode;
  offsetX_m?: number;
  offsetY_m?: number;
  inwardBufferM?: number;
  highRes?: boolean;
  highAccuracy?: boolean;
  confThresh?: number;
  signal?: AbortSignal;
  onProgress?: (p: SatelliteProgress) => void;
}): Promise<SatelliteAnalysisResult> {
  const {
    farmId = 'FARM',
    poly,
    species,
    cropType = '',
    engineerCount = null,
    forecast = null,
    submissionTime,
    useNdvi,
    providerMode = 'google',
    offsetX_m = 0,
    offsetY_m = 0,
    inwardBufferM = -1.5,
    highRes,
    highAccuracy = false,
    confThresh = DETECTION_CONFIG.aiModelConfThresh,
    signal,
    onProgress,
  } = params;
  const isHighRes = Boolean(highRes ?? highAccuracy);

  const calibratedPoly = shiftPolygonMeters(poly, offsetX_m, offsetY_m);
  const clippedPoly =
    Math.abs(inwardBufferM) >= 0.05
      ? bufferPolygonMeters(calibratedPoly, inwardBufferM)
      : calibratedPoly;

  const [cLon, cLat] = polygonCentroid(calibratedPoly);
  const [minx, miny, maxx, maxy] = polygonBounds(calibratedPoly);

  const metaPromise = fetch(
    `/api/satellite/metadata?lon=${cLon}&lat=${cLat}&minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&useNdvi=${useNdvi ? '1' : '0'}`,
    { signal }
  )
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null) as Promise<{
    imageryDateIso?: string | null;
    ndviSceneFound?: boolean;
  } | null>;

  const primaryPref: BasemapProvider = providerMode === 'esri' ? 'esri' : 'google';
  const secondaryPref: BasemapProvider = primaryPref === 'google' ? 'esri' : 'google';
  const strictSingleProvider = providerMode !== 'compare';

  let primaryMosaic = await fetchProviderMosaic(
    poly,
    calibratedPoly,
    clippedPoly,
    primaryPref,
    strictSingleProvider,
    { preferUltraZoom: true, highRes: isHighRes, highAccuracy: isHighRes, signal, onProgress }
  );
  if (!primaryMosaic) {
    primaryMosaic = await fetchProviderMosaic(
      poly,
      calibratedPoly,
      clippedPoly,
      secondaryPref,
      false,
      { preferUltraZoom: true, highRes: isHighRes, highAccuracy: isHighRes, signal, onProgress }
    );
  }
  if (!primaryMosaic) {
    throw new Error(
      'تعذّر تحميل صور القمر الصناعي الحية (Google Satellite / Esri World Imagery) لهذه الإحداثيات.'
    );
  }

  let compareImageUrl: string | null = null;
  let compareProvider: string | null = null;
  let compareZoomUsed: number | null = null;
  if (providerMode === 'compare') {
    const altMosaic = await fetchProviderMosaic(
      poly,
      calibratedPoly,
      clippedPoly,
      secondaryPref,
      true,
      { signal }
    );
    if (altMosaic) {
      compareImageUrl = renderComparisonCanvasUrl(altMosaic, inwardBufferM);
      compareProvider = altMosaic.provider;
      compareZoomUsed = altMosaic.zoom;
    }
  }

  const rawCanvas = primaryMosaic.canvas;
  const w = rawCanvas.width;
  const h = rawCanvas.height;
  const rawCtx = rawCanvas.getContext('2d')!;
  const imgData = rawCtx.getImageData(0, 0, w, h);
  const rawImageUrl = rawCanvas.toDataURL('image/png');

  const strictMask = createPolygonMask(w, h, primaryMosaic.clippedRing);
  const { mask: boundaryMask } = createBoundaryCorridorMask(w, h, primaryMosaic.ring, primaryMosaic.mpp);
  const strictMaskedCanvas = buildMaskedCanvas(rawCanvas, strictMask);
  const maskedImageUrl = strictMaskedCanvas.toDataURL('image/png');
  const contextCanvas = buildContextPreservingVisionCanvas(
    rawCanvas,
    strictMask,
    primaryMosaic.clippedRing,
    false
  );
  const visionCanvas = buildSharpenedVisionCanvas(contextCanvas, isHighRes ? 0.34 : 0.28);

  let polyPxCount = 0;
  let polyVegCount = 0;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) continue;
    polyPxCount++;
    const o = i * 4;
    const sum = imgData.data[o] + imgData.data[o + 1] + imgData.data[o + 2] + 1e-6;
    const exgVal = (2 * imgData.data[o + 1] - imgData.data[o] - imgData.data[o + 2]) / sum;
    if (exgVal > VEG_EXG) polyVegCount++;
  }
  const vegPct = polyPxCount > 0 ? Number(((polyVegCount / polyPxCount) * 100).toFixed(1)) : 0;

  const maps = await computeLuminanceAndLaplacianAsync(imgData.data, w, h, strictMask);

  const detectOut = await detectTreesHybrid({
    rawCanvas,
    visionCanvas,
    rgba: imgData.data,
    width: w,
    height: h,
    strictMask,
    boundaryMask,
    clippedRing: primaryMosaic.clippedRing,
    worldOriginX: primaryMosaic.worldOriginX,
    worldOriginY: primaryMosaic.worldOriginY,
    zoom: primaryMosaic.zoom,
    mpp: primaryMosaic.mpp,
    species,
    highRes: isHighRes,
    highAccuracy: isHighRes,
    confThresh,
    inwardBufferM,
    maps,
    signal,
    onProgress,
  });

  onProgress?.({ stage: 'postprocess', loaded: 4, total: 4, message: 'AI Arbitrator comparing libraries...' });

  const comparison = detectOut.libraryComparison;
  let finalBoxes: TreeDetectionBox[] = detectOut.boxes;

  const arbitration = await requestAiLibraryArbitration({
    farmId,
    species,
    cropType,
    vegPct,
    mpp: primaryMosaic.mpp,
    engineerCount,
    forecast,
    inwardBufferM,
    highResMode: isHighRes,
    provider: primaryMosaic.provider,
    comparison,
    signal,
  });

  if (arbitration) {
    comparison.aiArbitration = arbitration;
    const winnerLib = comparison.libraries.find(
      (l) => l.id === arbitration.winnerEngineId && l.status === 'ran'
    );
    if (winnerLib) {
      comparison.recommendedEngineId = winnerLib.id;
      comparison.activeEngineId = winnerLib.id;
      comparison.recommendedCount = winnerLib.count;
      finalBoxes = winnerLib.boxes;
    }
  }

  onProgress?.({ stage: 'postprocess', loaded: 1, total: 1, message: 'Counting & rendering overlays...' });

  const countOut = countTreesInPolygon({
    boxes: finalBoxes,
    allThresholdCandidates: detectOut.allThresholdCandidates,
    strictMask,
    boundaryMask,
    width: w,
    height: h,
    mpp: primaryMosaic.mpp,
    inwardBufferM,
    confThresh,
  });

  const { cleanImageUrl, annotatedImageUrl } = renderCleanAndAnnotatedCanvases(
    rawCanvas,
    strictMask,
    primaryMosaic.ring,
    primaryMosaic.clippedRing,
    finalBoxes,
    inwardBufferM
  );

  const meta = await metaPromise;
  let imageryDate: string | null = null;
  let imgAge: number | null = null;
  if (meta?.imageryDateIso) {
    const idate = new Date(meta.imageryDateIso);
    if (!Number.isNaN(idate.getTime())) {
      imageryDate = meta.imageryDateIso.slice(0, 7);
      const refParsed = new Date(submissionTime);
      const ref = Number.isNaN(refParsed.getTime()) ? new Date() : refParsed;
      imgAge = Number(((ref.getTime() - idate.getTime()) / (365.25 * 24 * 3600 * 1000)).toFixed(2));
    }
  }

  const pseudoNdvi = useNdvi
    ? computePseudoNdviFromMosaic({
        rgba: imgData.data,
        strictMask,
        width: w,
        height: h,
        vegPct,
        ndviSceneFound: meta?.ndviSceneFound,
      })
    : null;

  return {
    aiCount: countOut.aiCount,
    aiLow: countOut.aiLow,
    aiHigh: countOut.aiHigh,
    aiInteriorTrees: countOut.aiInteriorTrees,
    aiRaw: countOut.aiRaw,
    aiFactor: countOut.aiFactor,
    vegPct,
    crownM: countOut.crownM,
    imageryDate,
    imgAge,
    pseudoNdvi,
    zoomUsed: primaryMosaic.zoom,
    mpp: Number(primaryMosaic.mpp.toFixed(3)),
    provider: primaryMosaic.provider,
    providerId: primaryMosaic.providerId,
    detectorMode: detectOut.detectorMode,
    detectorEngine: detectOut.detectorEngine,
    aiVisionNotice: detectOut.aiVisionNotice,
    inwardBufferM,
    offsetX_m,
    offsetY_m,
    confThreshUsed: confThresh,
    canvasWidth: w,
    canvasHeight: h,
    worldOriginX: primaryMosaic.worldOriginX,
    worldOriginY: primaryMosaic.worldOriginY,
    ringPx: primaryMosaic.ring,
    clippedRingPx: primaryMosaic.clippedRing,
    boxes: finalBoxes,
    rawImageUrl,
    cleanImageUrl,
    maskedImageUrl,
    annotatedImageUrl,
    compareImageUrl,
    compareProvider,
    compareZoomUsed,
    stage1ActivePct: detectOut.stage1ActivePct,
    stage1AuditRejectedAi: detectOut.stage1AuditRejectedAi,
    highResMode: detectOut.highResMode,
    sahiTileSizeUsed: detectOut.sahiTileSizeUsed,
    sahiOverlapUsed: detectOut.sahiOverlapUsed,
    sahiTilesCount: detectOut.sahiTilesCount,
    xyzTilesSavedByPolyFilter: primaryMosaic.tilesSkipped ?? 0,
    libraryComparison: comparison,
  };
}

export * from './config';
export * from './tileFetcher';
export * from './imageProcessing';
export * from './treeDetector';
export * from './multiLibraryEngine';
export * from './polygonCounter';
export * from './canvasRenderer';
export * from './auditorIntelligence';

