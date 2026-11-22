import * as ort from 'onnxruntime-web';
import { normAr, present } from './arabicNorm';
import {
  BOUNDARY_BUFFER_M,
  bufferPolygonMeters,
  polygonBounds,
  polygonCentroid,
  shiftPolygonMeters,
} from './geometry';
import { AI_BAND, VEG_EXG } from './farmQaCore';
import type {
  BasemapProvider,
  BasemapViewMode,
  PointLonLat,
  SatelliteAnalysisResult,
  TreeClassName,
  TreeDetectionBox,
} from '../types/qa';

const TREE_CLASSES: TreeClassName[] = ['orchard_tree', 'palm_tree', 'windbreak_tree'];

const CALIB_SPECIES: Record<string, number> = {
  جازورين: 1.05,
  كازورينا: 1.05,
  casuarina: 1.05,
  كافور: 1.02,
  نخيل: 1.0,
  نخل: 1.0,
  زيتون: 1.0,
  موالح: 1.0,
};

function calibFactor(species: string): number {
  const s = present(species) ? normAr(species) : '';
  for (const [k, v] of Object.entries(CALIB_SPECIES)) {
    if (s && s.includes(normAr(k))) return v;
  }
  return 1.0;
}

function inferFallbackTreeClass(species: string, onBoundary: boolean, radiusM: number): {
  classId: 0 | 1 | 2;
  className: TreeClassName;
} {
  const s = present(species) ? normAr(species) : '';
  const isPalm = ['نخيل', 'نخل', 'بلح', 'تمر', 'palm', 'date'].some((k) => s.includes(normAr(k)));
  const isWindbreak = ['جازورين', 'كازورينا', 'كافور', 'سرو', 'فيكس', 'ماهوجني', 'casuarina', 'eucalyptus', 'مصدات'].some((k) =>
    s.includes(normAr(k))
  );

  if (isPalm || (!onBoundary && radiusM >= 2.7)) {
    return { classId: 1, className: 'palm_tree' };
  }
  if (onBoundary || isWindbreak) {
    return { classId: 2, className: 'windbreak_tree' };
  }
  return { classId: 0, className: 'orchard_tree' };
}

function lonLatToWorldPx(lon: number, lat: number, z: number): [number, number] {
  const n = 256 * Math.pow(2, z);
  const x = ((lon + 180.0) / 360.0) * n;
  const latRad = (lat * Math.PI) / 180.0;
  const y = ((1.0 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2.0) * n;
  return [x, y];
}

function loadTileImage(url: string, timeoutMs = 11000): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const timer = window.setTimeout(() => {
      img.src = '';
      resolve(null);
    }, timeoutMs);
    img.onload = () => {
      window.clearTimeout(timer);
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        resolve(img);
      } else {
        resolve(null);
      }
    };
    img.onerror = () => {
      window.clearTimeout(timer);
      resolve(null);
    };
    img.src = url;
  });
}

function isValidTileCanvas(img: HTMLImageElement): boolean {
  try {
    const c = document.createElement('canvas');
    c.width = 64;
    c.height = 64;
    const ctx = c.getContext('2d');
    if (!ctx) return true;
    ctx.drawImage(img, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    let lumSum = 0;
    let lumSqSum = 0;
    let diffSum = 0;
    const n = 64 * 64;
    for (let i = 0; i < n; i++) {
      const idx = i * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = (r + g + b) / 3;
      lumSum += lum;
      lumSqSum += lum * lum;
      diffSum += Math.abs(r - g) + Math.abs(g - b);
    }
    const mean = lumSum / n;
    const stdVal = Math.sqrt(Math.max(0, lumSqSum / n - mean * mean));
    const colorSpread = diffSum / n;
    return !(colorSpread < 3.0 && stdVal < 5.0);
  } catch {
    return true;
  }
}

function isValidSatelliteCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): boolean {
  if (w <= 0 || h <= 0) return false;
  const data = ctx.getImageData(0, 0, w, h).data;
  const totalPx = w * h;
  if (totalPx === 0) return false;

  let rgDiffSum = 0;
  let gbDiffSum = 0;
  let lumSum = 0;
  let lumSqSum = 0;
  const step = Math.max(1, Math.floor(totalPx / 4000));
  let samples = 0;

  for (let i = 0; i < totalPx; i += step) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    rgDiffSum += Math.abs(r - g);
    gbDiffSum += Math.abs(g - b);
    const lum = (r + g + b) / 3;
    lumSum += lum;
    lumSqSum += lum * lum;
    samples++;
  }

  if (samples === 0) return false;
  const colorSpread = (rgDiffSum + gbDiffSum) / samples;
  const mean = lumSum / samples;
  const variance = Math.max(0, lumSqSum / samples - mean * mean);
  const stdVal = Math.sqrt(variance);

  return !(colorSpread < 3.0 && stdVal < 8.0);
}

async function fetchTilesPooled(
  coords: Array<[number, number]>,
  z: number,
  provider: BasemapProvider,
  strictProvider: boolean,
  concurrency = 6
): Promise<Array<{ tx: number; ty: number; img: HTMLImageElement | null }>> {
  const results: Array<{ tx: number; ty: number; img: HTMLImageElement | null }> = new Array(
    coords.length
  );
  let cursor = 0;
  const fallbackFlag = strictProvider ? '0' : '1';

  async function worker() {
    while (true) {
      const idx = cursor++;
      if (idx >= coords.length) break;
      const [tx, ty] = coords[idx];
      let img = await loadTileImage(
        `/api/satellite/tile/${provider}/${z}/${tx}/${ty}?fallback=${fallbackFlag}`,
        10000
      );
      if (img && !isValidTileCanvas(img)) {
        if (!strictProvider) {
          const alt: BasemapProvider = provider === 'google' ? 'esri' : 'google';
          const altImg = await loadTileImage(
            `/api/satellite/tile/${alt}/${z}/${tx}/${ty}?fallback=0`,
            10000
          );
          img = altImg && isValidTileCanvas(altImg) ? altImg : null;
        } else {
          img = null;
        }
      }
      results[idx] = { tx, ty, img };
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, coords.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

interface FetchedMosaic {
  canvas: HTMLCanvasElement;
  ring: Array<[number, number]>;
  clippedRing: Array<[number, number]>;
  zoom: number;
  provider: string;
  providerId: BasemapProvider;
  mpp: number;
}

async function fetchProviderMosaic(
  rawPoly: PointLonLat[],
  calibratedPoly: PointLonLat[],
  clippedPoly: PointLonLat[],
  provider: BasemapProvider,
  strictOnly: boolean
): Promise<FetchedMosaic | null> {
  const [minx, miny, maxx, maxy] = polygonBounds(calibratedPoly);
  const dx = Math.max((maxx - minx) * 0.16, 0.00018);
  const dy = Math.max((maxy - miny) * 0.16, 0.00018);
  const bMinx = minx - dx;
  const bMiny = miny - dy;
  const bMaxx = maxx + dx;
  const bMaxy = maxy + dy;

  // Native high-resolution target zoom per provider:
  // Google Satellite: Zoom 19–20 (~0.15–0.30 m/px)
  // Esri World Imagery: Zoom 18–19 (~0.30–0.60 m/px)
  const startZoom = provider === 'google' ? 20 : 19;
  const minPrefZoom = provider === 'google' ? 19 : 18;
  const providerLabel = provider === 'google' ? 'Google Satellite' : 'Esri World Imagery';

  let bestCandidate: (FetchedMosaic & { loadedRatio: number }) | null = null;

  for (let z = startZoom; z >= 13; z--) {
    const [x0, y1] = lonLatToWorldPx(bMinx, bMiny, z);
    const [x1, y0] = lonLatToWorldPx(bMaxx, bMaxy, z);
    const tx0 = Math.floor(x0 / 256);
    const tx1 = Math.floor(x1 / 256);
    const ty0 = Math.floor(y0 / 256);
    const ty1 = Math.floor(y1 / 256);

    const tilesCount = (tx1 - tx0 + 1) * (ty1 - ty0 + 1);
    // Allow up to 64 tiles (8x8 = 2048x2048 px) at Z18-Z20 for native crisp resolution
    const maxAllowed = z >= minPrefZoom ? 64 : 49;
    if (tilesCount > maxAllowed || tilesCount <= 0) continue;

    const coords: Array<[number, number]> = [];
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        coords.push([tx, ty]);
      }
    }

    const loaded = await fetchTilesPooled(coords, z, provider, strictOnly, 6);
    const validTiles = loaded.filter((item) => item.img !== null);
    const loadedRatio = validTiles.length / Math.max(1, coords.length);
    if (validTiles.length === 0) continue;

    const mosaic = document.createElement('canvas');
    mosaic.width = (tx1 - tx0 + 1) * 256;
    mosaic.height = (ty1 - ty0 + 1) * 256;
    const mCtx = mosaic.getContext('2d');
    if (!mCtx) continue;

    mCtx.fillStyle = '#1e293b';
    mCtx.fillRect(0, 0, mosaic.width, mosaic.height);
    for (const { tx, ty, img } of loaded) {
      if (img) {
        mCtx.drawImage(img, (tx - tx0) * 256, (ty - ty0) * 256, 256, 256);
      }
    }

    const ox = tx0 * 256;
    const oy = ty0 * 256;
    const cropX = Math.floor(x0 - ox);
    const cropY = Math.floor(y0 - oy);
    const cropW = Math.max(64, Math.ceil(x1 - ox) - cropX);
    const cropH = Math.max(64, Math.ceil(y1 - oy) - cropY);

    const cropped = document.createElement('canvas');
    cropped.width = cropW;
    cropped.height = cropH;
    const cCtx = cropped.getContext('2d');
    if (!cCtx) continue;
    cCtx.drawImage(mosaic, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    const projectRing = (pList: PointLonLat[]): Array<[number, number]> =>
      pList.map(([lo, la]) => {
        const [px, py] = lonLatToWorldPx(lo, la, z);
        return [px - ox - cropX, py - oy - cropY];
      });

    const [, cLat] = polygonCentroid(calibratedPoly);
    const mpp = (156543.03392 * Math.cos((cLat * Math.PI) / 180)) / Math.pow(2, z);
    const isGood = isValidSatelliteCanvas(cCtx, cropW, cropH);

    const candidate: FetchedMosaic & { loadedRatio: number } = {
      canvas: cropped,
      ring: projectRing(calibratedPoly.length >= 3 ? calibratedPoly : rawPoly),
      clippedRing: projectRing(clippedPoly),
      zoom: z,
      provider: providerLabel,
      providerId: provider,
      mpp,
      loadedRatio,
    };

    if (loadedRatio >= 0.85 && isGood) {
      return candidate;
    }
    if (!bestCandidate || (isGood && loadedRatio > bestCandidate.loadedRatio)) {
      bestCandidate = candidate;
    }
    if (bestCandidate && bestCandidate.loadedRatio >= 0.7 && z <= minPrefZoom) {
      return bestCandidate;
    }
  }

  return bestCandidate;
}

function createPolygonMask(w: number, h: number, ring: Array<[number, number]>): Uint8Array {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  if (ring.length >= 3) {
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ring.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
  }

  const data = ctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = data[i * 4] > 127 ? 1 : 0;
  }
  return mask;
}

function createBoundaryCorridorMask(
  w: number,
  h: number,
  ring: Array<[number, number]>,
  mpp: number,
  bufferM = BOUNDARY_BUFFER_M
): { mask: Uint8Array; strokePx: number } {
  const strokePx = Math.max(8, Math.round((bufferM * 2.0) / Math.max(mpp, 0.05)));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  if (ring.length >= 3) {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = strokePx;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ring.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  const data = ctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = data[i * 4] > 127 ? 1 : 0;
  }
  return { mask, strokePx };
}

/**
 * Strictly clips the canvas to pixels inside `strictMask`, setting exterior pixels to zero.
 */
function buildStrictlyMaskedCanvas(
  srcCanvas: HTMLCanvasElement,
  strictMask: Uint8Array
): HTMLCanvasElement {
  const w = srcCanvas.width;
  const h = srcCanvas.height;
  const masked = document.createElement('canvas');
  masked.width = w;
  masked.height = h;
  const ctx = masked.getContext('2d')!;
  ctx.drawImage(srcCanvas, 0, 0);
  const frame = ctx.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) {
      const o = i * 4;
      px[o] = 0;
      px[o + 1] = 0;
      px[o + 2] = 0;
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(frame, 0, 0);
  return masked;
}

/**
 * Subtle contrast & edge preservation on imagery without digital upscaling.
 */
function enhanceSatelliteCanvas(srcCanvas: HTMLCanvasElement): HTMLCanvasElement {
  const w = srcCanvas.width;
  const h = srcCanvas.height;
  const out = document.createElement('canvas');
  out.width = w;
  out.height = h;
  const outCtx = out.getContext('2d');
  if (!outCtx) return srcCanvas;
  outCtx.drawImage(srcCanvas, 0, 0);

  const imgData = outCtx.getImageData(0, 0, w, h);
  const data = imgData.data;
  const totalPx = w * h;
  if (totalPx === 0) return out;

  const hist = new Int32Array(256);
  for (let i = 0; i < totalPx; i++) {
    const idx = i * 4;
    const l = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
    hist[l]++;
  }

  const p2Count = Math.floor(totalPx * 0.02);
  const p98Count = Math.floor(totalPx * 0.98);
  let accum = 0;
  let pLow = 0;
  let pHigh = 255;
  for (let l = 0; l < 256; l++) {
    accum += hist[l];
    if (pLow === 0 && accum >= p2Count) pLow = l;
    if (accum >= p98Count) {
      pHigh = l;
      break;
    }
  }
  if (pHigh <= pLow + 12) {
    pLow = 0;
    pHigh = 255;
  }
  const scale = 255 / (pHigh - pLow);

  const stretched = new Uint8ClampedArray(data.length);
  for (let i = 0; i < totalPx; i++) {
    const idx = i * 4;
    let r = Math.min(255, Math.max(0, (data[idx] - pLow) * scale));
    let g = Math.min(255, Math.max(0, (data[idx + 1] - pLow) * scale));
    let b = Math.min(255, Math.max(0, (data[idx + 2] - pLow) * scale));

    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    r = Math.min(255, Math.max(0, gray + (r - gray) * 1.12));
    g = Math.min(255, Math.max(0, gray + (g - gray) * 1.12));
    b = Math.min(255, Math.max(0, gray + (b - gray) * 1.12));

    stretched[idx] = r;
    stretched[idx + 1] = g;
    stretched[idx + 2] = b;
    stretched[idx + 3] = 255;
  }

  // Mild 3x3 unsharp mask
  const sharpData = new Uint8ClampedArray(data.length);
  for (let y = 0; y < h; y++) {
    const yMin = Math.max(0, y - 1);
    const yMax = Math.min(h - 1, y + 1);
    for (let x = 0; x < w; x++) {
      const xMin = Math.max(0, x - 1);
      const xMax = Math.min(w - 1, x + 1);
      const idx = (y * w + x) * 4;

      let bR = 0;
      let bG = 0;
      let bB = 0;
      let cnt = 0;
      for (let ny = yMin; ny <= yMax; ny++) {
        const rowOff = ny * w;
        for (let nx = xMin; nx <= xMax; nx++) {
          const nIdx = (rowOff + nx) * 4;
          bR += stretched[nIdx];
          bG += stretched[nIdx + 1];
          bB += stretched[nIdx + 2];
          cnt++;
        }
      }
      bR /= cnt;
      bG /= cnt;
      bB /= cnt;

      const origR = stretched[idx];
      const origG = stretched[idx + 1];
      const origB = stretched[idx + 2];
      const amount = 0.65;
      sharpData[idx] = Math.min(255, Math.max(0, origR + (origR - bR) * amount));
      sharpData[idx + 1] = Math.min(255, Math.max(0, origG + (origG - bG) * amount));
      sharpData[idx + 2] = Math.min(255, Math.max(0, origB + (origB - bB) * amount));
      sharpData[idx + 3] = 255;
    }
  }

  imgData.data.set(sharpData);
  outCtx.putImageData(imgData, 0, 0);
  return out;
}

// ---------------------------------------------------------------------------
// Stage 1: SAHI 512x512 Overlapping Sliding-Window Generator + IoU NMS (0.45)
// ---------------------------------------------------------------------------

interface SahiSlice {
  x: number;
  y: number;
  w: number;
  h: number;
}

function generateSahiSlices(
  imgW: number,
  imgH: number,
  mask: Uint8Array,
  sliceSize = 512,
  overlap = 0.2
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
      let hasActive = false;
      for (let py = sy; py < sy + sH && !hasActive; py += 4) {
        const rowOff = py * imgW;
        for (let px = sx; px < sx + sW; px += 4) {
          if (mask[rowOff + px]) {
            hasActive = true;
            break;
          }
        }
      }
      if (hasActive) {
        slices.push({ x: sx, y: sy, w: sW, h: sH });
      }
    }
  }
  return slices;
}

interface RawCandidate {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  score: number;
  classId: 0 | 1 | 2;
}

function boxIoU(a: RawCandidate, b: RawCandidate): number {
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

function mergeDetectionsNMS(
  candidates: RawCandidate[],
  iouThresh = 0.45,
  minCenterDistPx = 5
): RawCandidate[] {
  const ordered = [...candidates].sort((a, b) => b.score - a.score);
  const kept: RawCandidate[] = [];
  for (const cand of ordered) {
    const cx = (cand.xmin + cand.xmax) / 2;
    const cy = (cand.ymin + cand.ymax) / 2;
    const overlaps = kept.some((prev) => {
      if (boxIoU(cand, prev) >= iouThresh) return true;
      const pcx = (prev.xmin + prev.xmax) / 2;
      const pcy = (prev.ymin + prev.ymax) / 2;
      return Math.hypot(cx - pcx, cy - pcy) < minCenterDistPx;
    });
    if (!overlaps) {
      kept.push(cand);
      if (kept.length >= 900) break;
    }
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Stage 2: Real AI Inference (ONNX Runtime Web + Backend YOLO + Multi-Scale CV)
// ---------------------------------------------------------------------------

interface ModelStatusInfo {
  onnxReady: boolean;
  onnxUrl: string | null;
  backendYoloReady: boolean;
}

let cachedModelStatus: ModelStatusInfo | null = null;
let cachedOrtSession: ort.InferenceSession | null = null;
let ortLoadAttempted = false;

async function getModelStatus(): Promise<ModelStatusInfo> {
  if (cachedModelStatus) return cachedModelStatus;
  try {
    const resp = await fetch('/api/satellite/model-status');
    if (resp.ok) {
      const data = (await resp.json()) as Partial<ModelStatusInfo>;
      cachedModelStatus = {
        onnxReady: Boolean(data.onnxReady),
        onnxUrl: data.onnxUrl || null,
        backendYoloReady: Boolean(data.backendYoloReady),
      };
      return cachedModelStatus;
    }
  } catch {
    // ignore
  }
  cachedModelStatus = { onnxReady: false, onnxUrl: null, backendYoloReady: false };
  return cachedModelStatus;
}

async function getOnnxSession(onnxUrl: string): Promise<ort.InferenceSession | null> {
  if (cachedOrtSession) return cachedOrtSession;
  if (ortLoadAttempted) return null;
  ortLoadAttempted = true;
  try {
    ort.env.wasm.numThreads = 1;
    cachedOrtSession = await ort.InferenceSession.create(onnxUrl, {
      executionProviders: ['wasm'],
    });
    return cachedOrtSession;
  } catch {
    return null;
  }
}

function runOnnxTileInference(
  session: ort.InferenceSession,
  tileCanvas: HTMLCanvasElement,
  sliceX: number,
  sliceY: number,
  confThresh = 0.22
): Promise<RawCandidate[]> {
  return (async () => {
    const inputSize = 512;
    const resized = document.createElement('canvas');
    resized.width = inputSize;
    resized.height = inputSize;
    const rCtx = resized.getContext('2d')!;
    rCtx.drawImage(tileCanvas, 0, 0, tileCanvas.width, tileCanvas.height, 0, 0, inputSize, inputSize);
    const { data } = rCtx.getImageData(0, 0, inputSize, inputSize);

    const floatData = new Float32Array(3 * inputSize * inputSize);
    const plane = inputSize * inputSize;
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      floatData[i] = data[o] / 255.0;
      floatData[plane + i] = data[o + 1] / 255.0;
      floatData[2 * plane + i] = data[o + 2] / 255.0;
    }

    const inputName = session.inputNames[0] || 'images';
    const tensor = new ort.Tensor('float32', floatData, [1, 3, inputSize, inputSize]);
    const results = await session.run({ [inputName]: tensor });
    const outName = session.outputNames[0];
    const output = results[outName];
    const dims = output.dims; // YOLOv8 typically [1, 4+C, N]
    const raw = output.data as Float32Array;
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
  })();
}

async function runBackendYoloTile(
  tileCanvas: HTMLCanvasElement,
  sliceX: number,
  sliceY: number
): Promise<{ ok: boolean; engine: string; cands: RawCandidate[] }> {
  try {
    const b64 = tileCanvas.toDataURL('image/png');
    const resp = await fetch('/api/satellite/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64: b64, conf: 0.22, iou: 0.45 }),
    });
    if (!resp.ok) return { ok: false, engine: 'unavailable', cands: [] };
    const data = (await resp.json()) as {
      ok?: boolean;
      engine?: string;
      boxes?: Array<{ xmin: number; ymin: number; xmax: number; ymax: number; classId?: number; score?: number }>;
    };
    if (!data.ok) return { ok: false, engine: data.engine || 'unavailable', cands: [] };
    const cands: RawCandidate[] = (data.boxes || []).map((b) => ({
      xmin: sliceX + b.xmin,
      ymin: sliceY + b.ymin,
      xmax: sliceX + b.xmax,
      ymax: sliceY + b.ymax,
      score: Number(b.score ?? 0.5),
      classId: (b.classId === 1 || b.classId === 2 ? b.classId : 0) as 0 | 1 | 2,
    }));
    return { ok: true, engine: data.engine || 'YOLOv8 Backend', cands };
  } catch {
    return { ok: false, engine: 'unavailable', cands: [] };
  }
}

/**
 * Multi-Scale Canopy Candidate Proposal on a 512x512 masked tile (used when ONNX/Python YOLO weights are absent).
 * Produces raw candidates that are subsequently filtered by Stage 3 (Cast-Shadow + Laplacian Variance).
 */
function proposeCanopyCandidatesInSlice(
  tileData: Uint8ClampedArray,
  tw: number,
  th: number,
  sliceX: number,
  sliceY: number,
  strictMask: Uint8Array,
  boundaryMask: Uint8Array,
  fullW: number,
  mpp: number,
  species: string
): RawCandidate[] {
  const scoreMap = new Float32Array(tw * th);

  for (let y = 0; y < th; y++) {
    const gy = sliceY + y;
    const gRow = gy * fullW;
    const tRow = y * tw;
    for (let x = 0; x < tw; x++) {
      const gx = sliceX + x;
      if (!strictMask[gRow + gx]) continue;
      const o = (tRow + x) * 4;
      const r = tileData[o];
      const g = tileData[o + 1];
      const b = tileData[o + 2];
      const sum = r + g + b + 1e-6;
      const exg = (2 * g - r - b) / sum;
      const lum = (r + g + b) / 3;
      const greenComp = Math.max(0, (exg - 0.012) * 3.1);
      const darkCanopy = lum > 20 && lum < 120 && g >= r * 0.9 ? (120 - lum) / 230 : 0;
      scoreMap[tRow + x] = Math.min(1, greenComp * 0.68 + darkCanopy * 0.52);
    }
  }

  const rad = Math.max(2, Math.min(8, Math.round(1.65 / Math.max(mpp, 0.12))));
  const smoothed = new Float32Array(tw * th);
  for (let y = rad; y < th - rad; y++) {
    for (let x = rad; x < tw - rad; x++) {
      if (!strictMask[(sliceY + y) * fullW + (sliceX + x)]) continue;
      let acc = 0;
      let cnt = 0;
      for (let dy = -rad; dy <= rad; dy++) {
        const off = (y + dy) * tw;
        for (let dx = -rad; dx <= rad; dx++) {
          acc += scoreMap[off + x + dx];
          cnt++;
        }
      }
      smoothed[y * tw + x] = acc / cnt;
    }
  }

  const out: RawCandidate[] = [];
  for (let y = rad + 1; y < th - rad - 1; y += 2) {
    for (let x = rad + 1; x < tw - rad - 1; x += 2) {
      const val = smoothed[y * tw + x];
      if (val < AI_BAND[2]) continue;

      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (smoothed[(y + dy) * tw + (x + dx)] > val) {
            isMax = false;
            break;
          }
        }
      }
      if (!isMax) continue;

      const gx = sliceX + x;
      const gy = sliceY + y;
      const onBound = Boolean(boundaryMask[gy * fullW + gx]);
      const rPx = Math.max(4, rad * (val > 0.34 ? 1.7 : 1.35));
      const radiusM = rPx * mpp;
      const { classId } = inferFallbackTreeClass(species, onBound, radiusM);

      out.push({
        xmin: gx - rPx,
        ymin: gy - rPx,
        xmax: gx + rPx,
        ymax: gy + rPx,
        score: Math.min(0.97, val * 1.35),
        classId,
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Stage 3: Cast-Shadow (HSV Value Drop) & Laplacian Texture Variance Post-Filter
// ---------------------------------------------------------------------------

function filterByShadowAndTexture(
  candidates: RawCandidate[],
  imgData: ImageData,
  strictMask: Uint8Array,
  boundaryMask: Uint8Array,
  mpp: number,
  species: string
): TreeDetectionBox[] {
  const w = imgData.width;
  const h = imgData.height;
  const data = imgData.data;

  // Precompute luminance & HSV Value [0..1]
  const lum = new Float32Array(w * h);
  const hsvV = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const r = data[o];
    const g = data[o + 1];
    const b = data[o + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    hsvV[i] = Math.max(r, g, b) / 255.0;
  }

  const verified: TreeDetectionBox[] = [];
  let nextId = 1;

  for (const cand of candidates) {
    const cx = Math.round((cand.xmin + cand.xmax) / 2);
    const cy = Math.round((cand.ymin + cand.ymax) / 2);
    if (cx < 2 || cx >= w - 2 || cy < 2 || cy >= h - 2) continue;
    if (!strictMask[cy * w + cx]) continue;

    const x1 = Math.max(1, Math.floor(cand.xmin));
    const y1 = Math.max(1, Math.floor(cand.ymin));
    const x2 = Math.min(w - 2, Math.ceil(cand.xmax));
    const y2 = Math.min(h - 2, Math.ceil(cand.ymax));
    const radiusPx = Math.max(3, (cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4);

    // 1. Laplacian Texture Variance inside candidate crown box (rejects smooth flat row crops)
    let lapSum = 0;
    let lapSqSum = 0;
    let lumSum = 0;
    let lumSqSum = 0;
    let cnt = 0;
    let greenSum = 0;

    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        const idx = y * w + x;
        const l = lum[idx];
        const lap = Math.abs(
          lum[idx - 1] + lum[idx + 1] + lum[idx - w] + lum[idx + w] - 4 * l
        );
        lapSum += lap;
        lapSqSum += lap * lap;
        lumSum += l;
        lumSqSum += l * l;
        const pIdx = idx * 4;
        const sumRgb = data[pIdx] + data[pIdx + 1] + data[pIdx + 2] + 1e-6;
        greenSum += (2 * data[pIdx + 1] - data[pIdx] - data[pIdx + 2]) / sumRgb;
        cnt++;
      }
    }
    if (cnt < 9) continue;

    const meanLap = lapSum / cnt;
    const laplacianVar = Math.max(0, lapSqSum / cnt - meanLap * meanLap);
    const meanLum = lumSum / cnt;
    const lumStd = Math.sqrt(Math.max(0, lumSqSum / cnt - meanLum * meanLum));
    const meanExg = greenSum / cnt;

    // Require minimum chlorophyll vegetation signal (reject bare rooftops/soil)
    if (meanExg < 0.008) continue;

    // 2. Adjacent Cast-Shadow Check: sample HSV Value in surrounding skirt & northern shadow quadrant
    let crownVSum = 0;
    let crownVCnt = 0;
    const cRad = Math.max(2, Math.round(radiusPx * 0.55));
    for (let dy = -cRad; dy <= cRad; dy++) {
      const py = cy + dy;
      if (py < 0 || py >= h) continue;
      for (let dx = -cRad; dx <= cRad; dx++) {
        const px = cx + dx;
        if (px < 0 || px >= w) continue;
        crownVSum += hsvV[py * w + px];
        crownVCnt++;
      }
    }
    const crownV = crownVCnt > 0 ? crownVSum / crownVCnt : hsvV[cy * w + cx];

    const shRIn = Math.max(3, Math.round(radiusPx * 0.95));
    const shROut = Math.max(shRIn + 2, Math.round(radiusPx * 2.4));
    const sy1 = Math.max(0, cy - shROut);
    const sy2 = Math.min(h - 1, cy + Math.round(shROut * 0.6)); // emphasizes North/NW/NE cast shadows
    const sx1 = Math.max(0, cx - shROut);
    const sx2 = Math.min(w - 1, cx + shROut);

    let minShadowV = 1.0;
    let darkShadowPx = 0;
    let ringVSum = 0;
    let ringVCnt = 0;

    for (let sy = sy1; sy <= sy2; sy++) {
      for (let sx = sx1; sx <= sx2; sx++) {
        const d2 = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
        if (d2 >= shRIn * shRIn && d2 <= shROut * shROut) {
          const v = hsvV[sy * w + sx];
          ringVSum += v;
          ringVCnt++;
          if (v < minShadowV) minShadowV = v;
          if (v < 0.24) darkShadowPx++;
        }
      }
    }

    const ringMeanV = ringVCnt > 0 ? ringVSum / ringVCnt : crownV;
    const shadowDrop = Math.max(crownV - minShadowV, ringMeanV - minShadowV);
    const hasCastShadow = (shadowDrop >= 0.085 && minShadowV <= 0.27) || darkShadowPx >= 3;

    // 3. Opposing-Axis Canopy Isolation Check (eliminates flat green field carpets & 1D border lines)
    let twoSidedDropAxes = 0;
    const rRay = Math.max(4, Math.round(radiusPx * 1.7));
    const rayExg: number[] = [];
    for (let a = 0; a < 8; a++) {
      const radAngle = (a * Math.PI) / 4;
      const rx = Math.round(cx + rRay * Math.cos(radAngle));
      const ry = Math.round(cy + rRay * Math.sin(radAngle));
      if (rx >= 0 && rx < w && ry >= 0 && ry < h) {
        const o = (ry * w + rx) * 4;
        const sRgb = data[o] + data[o + 1] + data[o + 2] + 1e-6;
        rayExg.push((2 * data[o + 1] - data[o] - data[o + 2]) / sRgb);
      } else {
        rayExg.push(-0.2);
      }
    }
    for (let a = 0; a < 4; a++) {
      if (meanExg - rayExg[a] >= 0.015 && meanExg - rayExg[a + 4] >= 0.015) {
        twoSidedDropAxes++;
      }
    }

    // Combined Stage-3 Gate: reject uniform low-variance crop beds without cast shadows
    const hasHighTexture = laplacianVar >= 26.0 || lumStd >= 6.8;
    if (!hasCastShadow && !hasHighTexture) continue;
    if (!hasCastShadow && twoSidedDropAxes < 2) continue;
    if (laplacianVar < 14.0 && lumStd < 5.0) continue;

    const onBoundary = Boolean(boundaryMask[cy * w + cx]);
    const radiusM = radiusPx * mpp;
    const fallbackCls = inferFallbackTreeClass(species, onBoundary, radiusM);
    const classId = (cand.classId ?? fallbackCls.classId) as 0 | 1 | 2;
    const className = TREE_CLASSES[classId] || fallbackCls.className;

    const calibratedScore = Math.min(
      0.99,
      Math.max(
        cand.score,
        cand.score * (hasCastShadow ? 1.08 : 0.94) + Math.min(0.12, laplacianVar / 600)
      )
    );

    if (calibratedScore < AI_BAND[2]) continue;

    verified.push({
      id: nextId++,
      xmin: x1,
      ymin: y1,
      xmax: x2,
      ymax: y2,
      cx,
      cy,
      radiusPx: Math.round(radiusPx),
      score: Number(calibratedScore.toFixed(3)),
      classId,
      className,
      onBoundary,
      shadowDrop: Number(shadowDrop.toFixed(3)),
      laplacianVar: Number(laplacianVar.toFixed(1)),
    });
  }

  return verified;
}

// ---------------------------------------------------------------------------
// Clean Overlay Renderer (UTF-8 Arabic font stack, zero overlapping dark labels)
// ---------------------------------------------------------------------------

const CLASS_COLORS: Record<TreeClassName, string> = {
  orchard_tree: '#10b981',
  palm_tree: '#f59e0b',
  windbreak_tree: '#06b6d4',
};

function renderCleanAndAnnotatedCanvases(
  baseCanvas: HTMLCanvasElement,
  strictMask: Uint8Array,
  ring: Array<[number, number]>,
  clippedRing: Array<[number, number]>,
  boxes: TreeDetectionBox[],
  provider: string,
  zoom: number,
  detectorEngine: string,
  inwardBufferM: number
): { cleanImageUrl: string; annotatedImageUrl: string } {
  const w = baseCanvas.width;
  const h = baseCanvas.height;

  const cleanCanvas = document.createElement('canvas');
  cleanCanvas.width = w;
  cleanCanvas.height = h;
  const cCtx = cleanCanvas.getContext('2d')!;
  cCtx.drawImage(baseCanvas, 0, 0);

  // Dim outside strict masked polygon so auditor sees clipped evaluation zone clearly
  const frame = cCtx.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) {
      const o = i * 4;
      px[o] = Math.round(px[o] * 0.48);
      px[o + 1] = Math.round(px[o + 1] * 0.48);
      px[o + 2] = Math.round(px[o + 2] * 0.48);
    }
  }
  cCtx.putImageData(frame, 0, 0);

  // Draw calibrated outer farm GPS boundary & optional inward buffer ring
  if (ring.length >= 3) {
    cCtx.save();
    cCtx.strokeStyle = 'rgba(0, 245, 255, 0.95)';
    cCtx.lineWidth = 2.5;
    cCtx.lineJoin = 'round';
    cCtx.beginPath();
    ring.forEach(([x, y], i) => (i === 0 ? cCtx.moveTo(x, y) : cCtx.lineTo(x, y)));
    cCtx.closePath();
    cCtx.stroke();

    if (Math.abs(inwardBufferM) >= 0.05 && clippedRing.length >= 3) {
      cCtx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
      cCtx.lineWidth = 1.6;
      cCtx.setLineDash([6, 4]);
      cCtx.beginPath();
      clippedRing.forEach(([x, y], i) => (i === 0 ? cCtx.moveTo(x, y) : cCtx.lineTo(x, y)));
      cCtx.closePath();
      cCtx.stroke();
      cCtx.setLineDash([]);
    }
    cCtx.restore();
  }

  const cleanImageUrl = cleanCanvas.toDataURL('image/png');

  // Annotated image for lightbox/export using clean circles & UTF-8 Arabic font stack
  const annCanvas = document.createElement('canvas');
  annCanvas.width = w;
  annCanvas.height = h;
  const aCtx = annCanvas.getContext('2d')!;
  aCtx.drawImage(cleanCanvas, 0, 0);

  for (const b of boxes) {
    if (b.score < AI_BAND[1]) continue;
    const color = CLASS_COLORS[b.className] || '#10b981';
    aCtx.save();
    aCtx.strokeStyle = color;
    aCtx.lineWidth = 2;
    aCtx.beginPath();
    aCtx.arc(b.cx, b.cy, Math.max(4, b.radiusPx), 0, Math.PI * 2);
    aCtx.stroke();

    aCtx.fillStyle = color;
    aCtx.beginPath();
    aCtx.arc(b.cx, b.cy, 2.2, 0, Math.PI * 2);
    aCtx.fill();
    aCtx.restore();
  }

  // Crisp UTF-8 Top Banner
  aCtx.fillStyle = 'rgba(15, 23, 42, 0.90)';
  aCtx.fillRect(0, 0, w, 30);
  aCtx.fillStyle = '#38bdf8';
  aCtx.font = '600 12px "IBM Plex Sans Arabic", "Noto Sans Arabic", "JetBrains Mono", system-ui, sans-serif';
  aCtx.fillText(
    `${provider} · Z${zoom} · ${detectorEngine} · الأشجار المرصودة: ${boxes.filter((b) => b.score >= AI_BAND[1]).length}`,
    12,
    20
  );

  const annotatedImageUrl = annCanvas.toDataURL('image/png');
  return { cleanImageUrl, annotatedImageUrl };
}

function renderComparisonCanvasUrl(mosaic: FetchedMosaic, inwardBufferM: number): string {
  const w = mosaic.canvas.width;
  const h = mosaic.canvas.height;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(mosaic.canvas, 0, 0);

  const strictMask = createPolygonMask(w, h, mosaic.clippedRing);
  const frame = ctx.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) {
      const o = i * 4;
      px[o] = Math.round(px[o] * 0.48);
      px[o + 1] = Math.round(px[o + 1] * 0.48);
      px[o + 2] = Math.round(px[o + 2] * 0.48);
    }
  }
  ctx.putImageData(frame, 0, 0);

  if (mosaic.ring.length >= 3) {
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.95)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    mosaic.ring.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    if (Math.abs(inwardBufferM) >= 0.05 && mosaic.clippedRing.length >= 3) {
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.85)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      mosaic.clippedRing.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  return c.toDataURL('image/png');
}

// ---------------------------------------------------------------------------
// Public Entrypoint
// ---------------------------------------------------------------------------

export async function analyzeFarmSatellite(params: {
  poly: PointLonLat[];
  species: string;
  submissionTime: string;
  useNdvi: boolean;
  providerMode?: BasemapViewMode;
  offsetX_m?: number;
  offsetY_m?: number;
  inwardBufferM?: number;
}): Promise<SatelliteAnalysisResult> {
  const {
    poly,
    species,
    submissionTime,
    useNdvi,
    providerMode = 'google',
    offsetX_m = 0,
    offsetY_m = 0,
    inwardBufferM = -1.5,
  } = params;

  const calibratedPoly = shiftPolygonMeters(poly, offsetX_m, offsetY_m);
  const clippedPoly =
    Math.abs(inwardBufferM) >= 0.05
      ? bufferPolygonMeters(calibratedPoly, inwardBufferM)
      : calibratedPoly;

  const [cLon, cLat] = polygonCentroid(calibratedPoly);
  const [minx, miny, maxx, maxy] = polygonBounds(calibratedPoly);

  const metaPromise = fetch(
    `/api/satellite/metadata?lon=${cLon}&lat=${cLat}&minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&useNdvi=${useNdvi ? '1' : '0'}`
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
    strictSingleProvider
  );
  if (!primaryMosaic) {
    primaryMosaic = await fetchProviderMosaic(
      poly,
      calibratedPoly,
      clippedPoly,
      secondaryPref,
      false
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
      true
    );
    if (altMosaic) {
      compareImageUrl = renderComparisonCanvasUrl(altMosaic, inwardBufferM);
      compareProvider = altMosaic.provider;
      compareZoomUsed = altMosaic.zoom;
    }
  }

  const canvas = enhanceSatelliteCanvas(primaryMosaic.canvas);
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.getImageData(0, 0, w, h);

  // Strict Polygon Masking: everything outside clippedPoly is masked to 0 before detection
  const strictMask = createPolygonMask(w, h, primaryMosaic.clippedRing);
  const { mask: boundaryMask } = createBoundaryCorridorMask(
    w,
    h,
    primaryMosaic.ring,
    primaryMosaic.mpp,
    BOUNDARY_BUFFER_M
  );

  // Compute vegetation percentage inside polygon
  let polyPxCount = 0;
  let polyVegCount = 0;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) continue;
    polyPxCount++;
    const o = i * 4;
    const sum = imgData.data[o] + imgData.data[o + 1] + imgData.data[o + 2] + 1e-6;
    const exg = (2 * imgData.data[o + 1] - imgData.data[o] - imgData.data[o + 2]) / sum;
    if (exg > VEG_EXG) polyVegCount++;
  }
  const vegPct = polyPxCount > 0 ? Number(((polyVegCount / polyPxCount) * 100).toFixed(1)) : 0;

  // Stage 1: Masked Canvas + SAHI 512x512 Overlapping Slices (20% overlap)
  const maskedCanvas = buildStrictlyMaskedCanvas(canvas, strictMask);
  const maskedCtx = maskedCanvas.getContext('2d')!;
  const slices = generateSahiSlices(w, h, strictMask, 512, 0.2);

  // Stage 2: Real AI Object Detection (ONNX Runtime Web -> Backend YOLO -> Multi-Scale CV)
  const modelStatus = await getModelStatus();
  let detectorEngine = 'SAHI 512px + Cast-Shadow & Laplacian Filter';
  let rawCandidates: RawCandidate[] = [];

  if (modelStatus.onnxReady && modelStatus.onnxUrl) {
    const session = await getOnnxSession(modelStatus.onnxUrl);
    if (session) {
      for (const s of slices) {
        const subCanvas = document.createElement('canvas');
        subCanvas.width = s.w;
        subCanvas.height = s.h;
        const sCtx = subCanvas.getContext('2d')!;
        sCtx.drawImage(maskedCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
        const tileCands = await runOnnxTileInference(session, subCanvas, s.x, s.y, 0.22);
        rawCandidates.push(...tileCands);
      }
      detectorEngine = 'YOLOv8 ONNX (SAHI 512px · 20% Overlap)';
    }
  }

  if (rawCandidates.length === 0 && modelStatus.backendYoloReady) {
    let backendWorked = false;
    for (const s of slices) {
      const subCanvas = document.createElement('canvas');
      subCanvas.width = s.w;
      subCanvas.height = s.h;
      subCanvas.getContext('2d')!.drawImage(maskedCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
      const res = await runBackendYoloTile(subCanvas, s.x, s.y);
      if (res.ok) {
        backendWorked = true;
        rawCandidates.push(...res.cands);
        detectorEngine = `${res.engine} (SAHI 512px)`;
      } else {
        break;
      }
    }
    if (!backendWorked) rawCandidates = [];
  }

  if (rawCandidates.length === 0) {
    for (const s of slices) {
      const tileImgData = maskedCtx.getImageData(s.x, s.y, s.w, s.h);
      const tileCands = proposeCanopyCandidatesInSlice(
        tileImgData.data,
        s.w,
        s.h,
        s.x,
        s.y,
        strictMask,
        boundaryMask,
        w,
        primaryMosaic.mpp,
        species
      );
      rawCandidates.push(...tileCands);
    }
  }

  // Merge SAHI slice overlaps using NMS (IoU = 0.45)
  const minCenterDistPx = Math.max(5, Math.round(2.3 / Math.max(primaryMosaic.mpp, 0.12)));
  const nmsMerged = mergeDetectionsNMS(rawCandidates, 0.45, minCenterDistPx);

  // Stage 3: Cast-Shadow & Laplacian Texture Variance Post-Filter
  const boxes = filterByShadowAndTexture(
    nmsMerged,
    imgData,
    strictMask,
    boundaryMask,
    primaryMosaic.mpp,
    species
  );

  // Count metrics: when inwardBufferM < 0, border windbreaks are clipped out, so count verified orchard trees cleanly
  const boundaryBoxes = boxes.filter((b) => b.onBoundary);
  const interiorBoxes = boxes.filter((b) => !b.onBoundary);
  const primaryBoxes = inwardBufferM < -0.5 ? boxes : boundaryBoxes;

  const counts = {
    lo: primaryBoxes.filter((b) => b.score >= AI_BAND[0]).length,
    mid: primaryBoxes.filter((b) => b.score >= AI_BAND[1]).length,
    hi: primaryBoxes.filter((b) => b.score >= AI_BAND[2]).length,
    interiorMid: interiorBoxes.filter((b) => b.score >= AI_BAND[1]).length,
  };

  const midBoxes = primaryBoxes.filter((b) => b.score >= AI_BAND[1]);
  let crownM: number | null = null;
  if (midBoxes.length > 0) {
    const sides = midBoxes
      .map((b) => Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * primaryMosaic!.mpp)
      .sort((a, b) => a - b);
    crownM = Number(sides[Math.floor(sides.length / 2)].toFixed(2));
  }

  const factor = calibFactor(species);
  const aiLow = Math.round(counts.lo * factor);
  const aiCount = Math.round(counts.mid * factor);
  const aiHigh = Math.round(counts.hi * factor);

  const { cleanImageUrl, annotatedImageUrl } = renderCleanAndAnnotatedCanvases(
    canvas,
    strictMask,
    primaryMosaic.ring,
    primaryMosaic.clippedRing,
    boxes,
    primaryMosaic.provider,
    primaryMosaic.zoom,
    detectorEngine,
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

  let ndviRecent: number | null = null;
  if (useNdvi) {
    let interiorGreen = 0;
    let interiorTotal = 0;
    for (let i = 0; i < w * h; i++) {
      if (!strictMask[i]) continue;
      const idx = i * 4;
      const r = imgData.data[idx];
      const g = imgData.data[idx + 1];
      interiorGreen += (g - r) / (g + r + 1e-6);
      interiorTotal++;
    }
    const meanVari = interiorTotal > 0 ? interiorGreen / interiorTotal : 0;
    const estimated = Math.max(0.08, Math.min(0.78, 0.28 + meanVari * 1.45 + (vegPct / 100) * 0.22));
    ndviRecent = Number(estimated.toFixed(2));
    if (meta?.ndviSceneFound === false && interiorTotal === 0) {
      ndviRecent = null;
    }
  }

  return {
    aiCount,
    aiLow,
    aiHigh,
    aiInteriorTrees: counts.interiorMid,
    aiRaw: counts.mid,
    aiFactor: factor,
    vegPct,
    crownM,
    imageryDate,
    imgAge,
    ndviRecent,
    zoomUsed: primaryMosaic.zoom,
    mpp: Number(primaryMosaic.mpp.toFixed(3)),
    provider: primaryMosaic.provider,
    providerId: primaryMosaic.providerId,
    detectorEngine,
    inwardBufferM,
    offsetX_m,
    offsetY_m,
    canvasWidth: w,
    canvasHeight: h,
    ringPx: primaryMosaic.ring,
    clippedRingPx: primaryMosaic.clippedRing,
    boxes,
    cleanImageUrl,
    annotatedImageUrl,
    compareImageUrl,
    compareProvider,
    compareZoomUsed,
  };
}
