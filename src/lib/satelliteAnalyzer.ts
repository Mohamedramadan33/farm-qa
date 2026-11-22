import * as ort from 'onnxruntime-web';
import { normAr, present } from './arabicNorm';
import {
  BOUNDARY_BUFFER_M,
  bufferPolygonMeters,
  polygonBounds,
  polygonCentroid,
  shiftPolygonMeters,
} from './geometry';
import { VEG_EXG } from './farmQaCore';
import type {
  BasemapProvider,
  BasemapViewMode,
  PointLonLat,
  SatelliteAnalysisResult,
  TreeClassName,
  TreeDetectionBox,
} from '../types/qa';

const TREE_CLASSES: TreeClassName[] = ['orchard_tree', 'palm_tree', 'windbreak_tree'];
const AI_MODEL_CONF_THRESH = 0.55;
const SAHI_TILE_SIZE = 512;
const SAHI_OVERLAP = 0.2;
const NMS_IOU_THRESH = 0.45;
const MIN_CAST_SHADOW_DROP = 0.35; // 35% luminance drop required relative to crown center
const MIN_BLOB_CIRCULARITY = 0.65; // > 0.65 circularity required for DoG crown blobs
const MIN_CROWN_DIAM_M = 2.0;
const MAX_CROWN_DIAM_M = 6.5;

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

  if (isPalm || (!onBoundary && diameterM >= 5.0)) {
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
  concurrency = 8
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
        11000
      );
      if (img && !isValidTileCanvas(img)) {
        if (!strictProvider) {
          const alt: BasemapProvider = provider === 'google' ? 'esri' : 'google';
          const altImg = await loadTileImage(
            `/api/satellite/tile/${alt}/${z}/${tx}/${ty}?fallback=0`,
            11000
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

/**
 * Offscreen high-resolution tile stitcher targeting native 256x256 tiles at Zoom 19 (~0.26 m/px)
 * covering the exact GeoJSON polygon bounding box + ~14m shadow margin.
 */
async function fetchProviderMosaic(
  rawPoly: PointLonLat[],
  calibratedPoly: PointLonLat[],
  clippedPoly: PointLonLat[],
  provider: BasemapProvider,
  strictOnly: boolean
): Promise<FetchedMosaic | null> {
  const [minx, miny, maxx, maxy] = polygonBounds(calibratedPoly);
  const dx = Math.max((maxx - minx) * 0.08, 0.00014);
  const dy = Math.max((maxy - miny) * 0.08, 0.00014);
  const bMinx = minx - dx;
  const bMiny = miny - dy;
  const bMaxx = maxx + dx;
  const bMaxy = maxy + dy;

  // Target Zoom 19 (~0.26 m/px in Egypt) so large farms retain native tree canopy resolution
  const startZoom = 19;
  const providerLabel = provider === 'google' ? 'Google Satellite' : 'Esri World Imagery';

  let bestCandidate: (FetchedMosaic & { loadedRatio: number }) | null = null;

  for (let z = startZoom; z >= 15; z--) {
    const [x0, y1] = lonLatToWorldPx(bMinx, bMiny, z);
    const [x1, y0] = lonLatToWorldPx(bMaxx, bMaxy, z);
    const tx0 = Math.floor(x0 / 256);
    const tx1 = Math.floor(x1 / 256);
    const ty0 = Math.floor(y0 / 256);
    const ty1 = Math.floor(y1 / 256);

    const tilesX = tx1 - tx0 + 1;
    const tilesY = ty1 - ty0 + 1;
    const tilesCount = tilesX * tilesY;
    // Allow up to 16x16 = 256 tiles (4096x4096 px) at Zoom 19 (~1.06 km x 1.06 km farm at native resolution)
    const maxAllowed = z === 19 ? 256 : 121;
    if (tilesCount > maxAllowed || tilesCount <= 0) continue;

    const coords: Array<[number, number]> = [];
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        coords.push([tx, ty]);
      }
    }

    const loaded = await fetchTilesPooled(coords, z, provider, strictOnly, 8);
    const validTiles = loaded.filter((item) => item.img !== null);
    const loadedRatio = validTiles.length / Math.max(1, coords.length);
    if (validTiles.length === 0) continue;

    const mosaic = document.createElement('canvas');
    mosaic.width = tilesX * 256;
    mosaic.height = tilesY * 256;
    const mCtx = mosaic.getContext('2d');
    if (!mCtx) continue;

    mCtx.fillStyle = '#0f172a';
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
    if (bestCandidate && bestCandidate.loadedRatio >= 0.7 && z <= 18) {
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

function buildMaskedCanvas(srcCanvas: HTMLCanvasElement, mask: Uint8Array): HTMLCanvasElement {
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
    if (!mask[i]) {
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

// ---------------------------------------------------------------------------
// Stage 1: Flat-Crop Texture & Shadow Rejection (4m x 4m Pre-Filter Grid Mask)
// ---------------------------------------------------------------------------

interface Stage1Result {
  activeMask: Uint8Array;
  lum: Float32Array;
  hsvV: Float32Array;
  lapMag: Float32Array;
  cellsKept: number;
  cellsTotal: number;
  activePct: number;
}

/**
 * Divides the strictly masked polygon into 4m x 4m physical cells at native Zoom 19.
 * Computes per-cell:
 *   1. Local Texture Roughness / Laplacian Variance
 *   2. Dark Shadow Pixel Ratio (HSV Value V < 0.22 adjacent to brighter vegetation)
 * Zeroes out smooth uniform crop basins (berseem/clover, wheat, vegetables) BEFORE detection.
 */
function applyStage1FlatCropMask(
  imgData: ImageData,
  strictMask: Uint8Array,
  mpp: number
): Stage1Result {
  const w = imgData.width;
  const h = imgData.height;
  const px = imgData.data;
  const n = w * h;

  const lum = new Float32Array(n);
  const hsvV = new Float32Array(n);
  const lapMag = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const r = px[o];
    const g = px[o + 1];
    const b = px[o + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    hsvV[i] = Math.max(r, g, b) / 255.0;
  }

  // Discrete 3x3 Laplacian magnitude on raw un-stretched luminance
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (!strictMask[idx]) continue;
      lapMag[idx] = Math.abs(
        lum[idx - 1] + lum[idx + 1] + lum[idx - w] + lum[idx + w] - 4 * lum[idx]
      );
    }
  }

  // Compute bright-adjacent cast-shadow pocket mask:
  // Pixel has HSV Value V < 0.22 AND has a brighter neighboring pixel (V >= 0.30, ΔV >= 0.11) within ~2.2m
  const shadowPocket = new Uint8Array(n);
  const neighborRad = Math.max(4, Math.min(11, Math.round(2.2 / Math.max(mpp, 0.12))));
  const stepN = Math.max(1, Math.floor(neighborRad / 3));

  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    const ny0 = Math.max(0, y - neighborRad);
    const ny1 = Math.min(h - 1, y + neighborRad);
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      if (!strictMask[idx]) continue;
      const v = hsvV[idx];
      if (v >= 0.22) continue;

      const nx0 = Math.max(0, x - neighborRad);
      const nx1 = Math.min(w - 1, x + neighborRad);
      let maxNeighborV = 0;
      for (let ny = ny0; ny <= ny1; ny += stepN) {
        const nRow = ny * w;
        for (let nx = nx0; nx <= nx1; nx += stepN) {
          if (!strictMask[nRow + nx]) continue;
          const nv = hsvV[nRow + nx];
          if (nv > maxNeighborV) maxNeighborV = nv;
        }
      }
      if (maxNeighborV >= 0.30 && maxNeighborV - v >= 0.11) {
        shadowPocket[idx] = 1;
      }
    }
  }

  // Divide polygon into 4m x 4m spatial grid cells
  const cellPx = Math.max(10, Math.round(4.0 / Math.max(mpp, 0.12)));
  const haloPx = Math.max(3, Math.round(1.5 / Math.max(mpp, 0.12)));

  const activeMask = new Uint8Array(n);
  let cellsTotal = 0;
  let cellsKept = 0;
  const lumHist = new Int32Array(256);

  for (let cy0 = 0; cy0 < h; cy0 += cellPx) {
    const cy1 = Math.min(h, cy0 + cellPx);
    for (let cx0 = 0; cx0 < w; cx0 += cellPx) {
      const cx1 = Math.min(w, cx0 + cellPx);

      let validCnt = 0;
      let lapSum = 0;
      let lapSqSum = 0;
      lumHist.fill(0);

      for (let y = cy0; y < cy1; y++) {
        const row = y * w;
        for (let x = cx0; x < cx1; x++) {
          const idx = row + x;
          if (!strictMask[idx]) continue;
          validCnt++;
          const lm = lapMag[idx];
          lapSum += lm;
          lapSqSum += lm * lm;
          const bLum = Math.max(0, Math.min(255, Math.round(lum[idx])));
          lumHist[bLum]++;
        }
      }

      if (validCnt < 12) continue;
      cellsTotal++;

      const meanLap = lapSum / validCnt;
      const lapVar = Math.max(0, lapSqSum / validCnt - meanLap * meanLap);

      // Compute robust p5..p95 luminance contrast spread inside cell
      const p5Target = Math.floor(validCnt * 0.05);
      const p95Target = Math.floor(validCnt * 0.95);
      let acc = 0;
      let p5 = 0;
      let p95 = 255;
      let gotP5 = false;
      for (let b = 0; b < 256; b++) {
        acc += lumHist[b];
        if (!gotP5 && acc >= p5Target) {
          p5 = b;
          gotP5 = true;
        }
        if (acc >= p95Target) {
          p95 = b;
          break;
        }
      }
      const contrastRange = p95 - p5;

      // Measure shadow pocket ratio in cell + 1.5m boundary halo so crown-edge shadows aren't split
      const hy0 = Math.max(0, cy0 - haloPx);
      const hy1 = Math.min(h, cy1 + haloPx);
      const hx0 = Math.max(0, cx0 - haloPx);
      const hx1 = Math.min(w, cx1 + haloPx);
      let shadowCnt = 0;
      for (let y = hy0; y < hy1; y++) {
        const row = y * w;
        for (let x = hx0; x < hx1; x++) {
          if (shadowPocket[row + x]) shadowCnt++;
        }
      }
      const shadowRatio = shadowCnt / validCnt;

      // Reject smooth uniform crop cells lacking high-frequency texture AND cast-shadow pockets
      const keepCell =
        lapVar >= 35.0 && contrastRange >= 25 && shadowRatio >= 0.012 && shadowCnt >= 3;

      if (keepCell) {
        cellsKept++;
        for (let y = cy0; y < cy1; y++) {
          const row = y * w;
          for (let x = cx0; x < cx1; x++) {
            const idx = row + x;
            if (strictMask[idx]) activeMask[idx] = 1;
          }
        }
      }
    }
  }

  let strictTotal = 0;
  let activeTotal = 0;
  for (let i = 0; i < n; i++) {
    if (strictMask[i]) {
      strictTotal++;
      if (activeMask[i]) activeTotal++;
    }
  }
  const activePct = strictTotal > 0 ? Number(((activeTotal / strictTotal) * 100).toFixed(1)) : 0;

  return {
    activeMask,
    lum,
    hsvV,
    lapMag,
    cellsKept,
    cellsTotal,
    activePct,
  };
}

// ---------------------------------------------------------------------------
// SAHI 512x512 Overlapping Sliding-Window Generator + IoU NMS (0.45)
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
  sliceSize = SAHI_TILE_SIZE,
  overlap = SAHI_OVERLAP
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
      for (let py = sy; py < sy + sH && !hasActive; py += 2) {
        const rowOff = py * imgW;
        for (let px = sx; px < sx + sW; px += 2) {
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
  circularity?: number;
  crownDiameterM?: number;
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
  iouThresh = NMS_IOU_THRESH,
  minCenterDistPx = 6
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
// Stage 2A: Primary Path — Real AI Inference (ONNX Runtime Web / Backend YOLO)
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

async function runOnnxTileInference(
  session: ort.InferenceSession,
  tileCanvas: HTMLCanvasElement,
  sliceX: number,
  sliceY: number,
  confThresh = AI_MODEL_CONF_THRESH
): Promise<RawCandidate[]> {
  const inputSize = SAHI_TILE_SIZE;
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
  const dims = output.dims;
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
      body: JSON.stringify({
        imageBase64: b64,
        conf: AI_MODEL_CONF_THRESH,
        iou: NMS_IOU_THRESH,
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
      .filter((b) => Number(b.score ?? 0) >= AI_MODEL_CONF_THRESH)
      .map((b) => ({
        xmin: sliceX + b.xmin,
        ymin: sliceY + b.ymin,
        xmax: sliceX + b.xmax,
        ymax: sliceY + b.ymax,
        score: Number(b.score ?? 0.55),
        classId: (b.classId === 1 || b.classId === 2 ? b.classId : 0) as 0 | 1 | 2,
      }));
    return { ok: true, engine: data.engine || 'YOLOv8 Backend', cands };
  } catch {
    return { ok: false, engine: 'unavailable', cands: [] };
  }
}

// ---------------------------------------------------------------------------
// Stage 2B: Structural Multi-Scale DoG + Circularity (>0.65) + Paired Cast Shadow (>=35%)
// ---------------------------------------------------------------------------

function gaussianBlur2D(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const rad = Math.max(1, Math.min(18, Math.ceil(sigma * 2.4)));
  const kLen = rad * 2 + 1;
  const kernel = new Float32Array(kLen);
  let kSum = 0;
  const denom = 2 * sigma * sigma;
  for (let i = -rad; i <= rad; i++) {
    const val = Math.exp(-(i * i) / denom);
    kernel[i + rad] = val;
    kSum += val;
  }
  for (let i = 0; i < kLen; i++) kernel[i] /= kSum;

  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const sx = Math.max(0, Math.min(w - 1, x + k));
        acc += src[row + sx] * kernel[k + rad];
      }
      tmp[row + x] = acc;
    }
  }

  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const sy = Math.max(0, Math.min(h - 1, y + k));
        acc += tmp[sy * w + x] * kernel[k + rad];
      }
      out[row + x] = acc;
    }
  }
  return out;
}

// Exact 2x2 Marching Squares contour perimeter lookup table
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

interface BlobShapeStats {
  areaPx: number;
  perimeterPx: number;
  circularity: number;
  aspectRatio: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function measureDoGCrownBlob(
  dogMap: Float32Array,
  stage1Mask: Uint8Array,
  w: number,
  h: number,
  cx: number,
  cy: number,
  peakVal: number,
  maxRadiusPx: number
): BlobShapeStats | null {
  const rad = Math.max(5, Math.min(20, Math.ceil(maxRadiusPx)));
  const x0 = Math.max(1, cx - rad);
  const y0 = Math.max(1, cy - rad);
  const x1 = Math.min(w - 2, cx + rad);
  const y1 = Math.min(h - 2, cy + rad);
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  if (bw < 3 || bh < 3) return null;

  const thresh = peakVal * 0.42;
  const visited = new Uint8Array(bw * bh);
  const qx: number[] = [cx];
  const qy: number[] = [cy];
  visited[(cy - y0) * bw + (cx - x0)] = 1;

  let head = 0;
  let minBx = cx;
  let maxBx = cx;
  let minBy = cy;
  let maxBy = cy;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumYY = 0;
  let sumXY = 0;

  while (head < qx.length) {
    const px = qx[head];
    const py = qy[head];
    head++;

    if (px < minBx) minBx = px;
    if (px > maxBx) maxBx = px;
    if (py < minBy) minBy = py;
    if (py > maxBy) maxBy = py;
    sumX += px;
    sumY += py;
    sumXX += px * px;
    sumYY += py * py;
    sumXY += px * py;

    const dirs = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ];
    for (const [dx, dy] of dirs) {
      const nx = px + dx;
      const ny = py + dy;
      if (nx < x0 || nx > x1 || ny < y0 || ny > y1) continue;
      const lIdx = (ny - y0) * bw + (nx - x0);
      if (visited[lIdx]) continue;
      const gIdx = ny * w + nx;
      if (!stage1Mask[gIdx] || dogMap[gIdx] < thresh) continue;
      if ((nx - cx) * (nx - cx) + (ny - cy) * (ny - cy) > rad * rad) continue;
      visited[lIdx] = 1;
      qx.push(nx);
      qy.push(ny);
    }
  }

  const areaPx = qx.length;
  if (areaPx < 10) return null;

  // Compute exact 2x2 Marching Squares perimeter over local binary mask
  let perimeterPx = 0;
  for (let ly = 0; ly < bh - 1; ly++) {
    const row0 = ly * bw;
    const row1 = (ly + 1) * bw;
    for (let lx = 0; lx < bw - 1; lx++) {
      const state =
        (visited[row0 + lx] ? 8 : 0) |
        (visited[row0 + lx + 1] ? 4 : 0) |
        (visited[row1 + lx + 1] ? 2 : 0) |
        (visited[row1 + lx] ? 1 : 0);
      perimeterPx += MARCHING_PERIM[state];
    }
  }
  if (perimeterPx <= 1) return null;

  const circularity = (4 * Math.PI * areaPx) / (perimeterPx * perimeterPx);

  // Covariance eigenvalue aspect ratio to reject linear canal banks / field walls
  const mx = sumX / areaPx;
  const my = sumY / areaPx;
  const cxx = sumXX / areaPx - mx * mx;
  const cyy = sumYY / areaPx - my * my;
  const cxy = sumXY / areaPx - mx * my;
  const trace = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  const lam1 = trace / 2 + disc;
  const lam2 = Math.max(0.15, trace / 2 - disc);
  const aspectRatio = Math.sqrt(lam1 / lam2);

  return {
    areaPx,
    perimeterPx,
    circularity,
    aspectRatio,
    minX: minBx,
    minY: minBy,
    maxX: maxBx,
    maxY: maxBy,
  };
}

/**
 * Multi-Scale DoG Fallback Detector:
 * Runs ONLY inside Stage-1-active pixels (never across flat green crop beds).
 * Proposes candidates ONLY for circular DoG crown blobs (diameter 2.0m..6.5m, circularity > 0.65).
 */
function detectStructuralCrownsDoG(
  imgData: ImageData,
  lum: Float32Array,
  stage1Mask: Uint8Array,
  mpp: number,
  species: string,
  boundaryMask: Uint8Array
): RawCandidate[] {
  const w = imgData.width;
  const h = imgData.height;
  const px = imgData.data;
  const n = w * h;

  // Build structural canopy signal gated to Stage-1 active cells
  const canopySignal = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (!stage1Mask[i]) continue;
    const o = i * 4;
    const r = px[o];
    const g = px[o + 1];
    const b = px[o + 2];
    // Weighted green-structural intensity (attenuates bright white roofs while preserving dark & sunlit tree domes)
    const rgSpread = Math.max(0, g - r * 0.85);
    const baseStruct = (0.22 * r + 0.62 * g + 0.16 * b) * (g >= r * 0.88 ? 1.0 : 0.45);
    canopySignal[i] = Math.min(255, baseStruct * 0.82 + rgSpread * 0.65);
  }

  // Three DoG scales covering physical tree canopy diameters 2.0m to 6.5m
  const targetRadiiM = [1.25, 2.0, 2.85];
  const dogMaps: Float32Array[] = [];
  const scaleRadiiPx: number[] = [];

  for (const rM of targetRadiiM) {
    const rPx = Math.max(3.8, rM / Math.max(mpp, 0.12));
    scaleRadiiPx.push(rPx);
    const sigFine = Math.max(1.0, rPx * 0.52);
    const sigCoarse = Math.max(sigFine + 1.4, rPx * 1.45);
    const bFine = gaussianBlur2D(canopySignal, w, h, sigFine);
    const bCoarse = gaussianBlur2D(canopySignal, w, h, sigCoarse);
    const dog = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (stage1Mask[i]) {
        dog[i] = bFine[i] - bCoarse[i];
      }
    }
    dogMaps.push(dog);
  }

  const rawCandidates: RawCandidate[] = [];
  const stride = Math.max(2, Math.round(1.0 / Math.max(mpp, 0.12)));

  for (let y = 4; y < h - 4; y += stride) {
    for (let x = 4; x < w - 4; x += stride) {
      const idx = y * w + x;
      if (!stage1Mask[idx]) continue;

      let bestScale = 0;
      let bestVal = -999;
      for (let s = 0; s < dogMaps.length; s++) {
        const v = dogMaps[s][idx];
        if (v > bestVal) {
          bestVal = v;
          bestScale = s;
        }
      }

      if (bestVal < 4.5) continue;

      // Local 5x5 peak check on winning DoG scale
      const winDog = dogMaps[bestScale];
      let isPeak = true;
      for (let dy = -2; dy <= 2 && isPeak; dy++) {
        const py = y + dy;
        if (py < 0 || py >= h) continue;
        for (let dx = -2; dx <= 2; dx++) {
          if (dx === 0 && dy === 0) continue;
          const pxx = x + dx;
          if (pxx < 0 || pxx >= w) continue;
          if (winDog[py * w + pxx] > bestVal) {
            isPeak = false;
            break;
          }
        }
      }
      if (!isPeak) continue;

      const rPxEst = scaleRadiiPx[bestScale];
      const blob = measureDoGCrownBlob(
        winDog,
        stage1Mask,
        w,
        h,
        x,
        y,
        bestVal,
        rPxEst * 1.55
      );
      if (!blob) continue;

      const eqDiamM = 2 * Math.sqrt(blob.areaPx / Math.PI) * mpp;
      if (eqDiamM < MIN_CROWN_DIAM_M || eqDiamM > MAX_CROWN_DIAM_M) continue;
      if (blob.circularity <= MIN_BLOB_CIRCULARITY) continue;
      if (blob.aspectRatio > 2.2) continue;

      // Reject pure white/neutral-gray non-vegetation rooftops
      const o = idx * 4;
      const r = px[o];
      const g = px[o + 1];
      const b = px[o + 2];
      const sumRgb = r + g + b + 1e-6;
      const chromaExg = (2 * g - r - b) / sumRgb;
      const maxCh = Math.max(r, g, b);
      const minCh = Math.min(r, g, b);
      if (lum[idx] < 40) continue;
      if (chromaExg < 0.01 && g < r * 0.92) continue;
      if (maxCh - minCh < 7) continue;

      const boxR = Math.max(4, Math.round((eqDiamM / 2) / Math.max(mpp, 0.12)));
      const onBound = Boolean(boundaryMask[idx]);
      const { classId } = inferFallbackTreeClass(species, onBound, eqDiamM);

      rawCandidates.push({
        xmin: Math.max(1, x - boxR),
        ymin: Math.max(1, y - boxR),
        xmax: Math.min(w - 2, x + boxR),
        ymax: Math.min(h - 2, y + boxR),
        score: Math.min(0.96, 0.48 + blob.circularity * 0.35 + Math.min(0.18, bestVal / 40)),
        classId,
        circularity: Number(blob.circularity.toFixed(2)),
        crownDiameterM: Number(eqDiamM.toFixed(2)),
      });
    }
  }

  return rawCandidates;
}

// ---------------------------------------------------------------------------
// Stage 2 Verification: Paired Directional Cast Shadow (>= 35% Luminance Drop)
// ---------------------------------------------------------------------------

function verifyCastShadowAndTelemetry(
  candidates: RawCandidate[],
  imgData: ImageData,
  strictMask: Uint8Array,
  stage1Mask: Uint8Array,
  boundaryMask: Uint8Array,
  lum: Float32Array,
  hsvV: Float32Array,
  lapMag: Float32Array,
  mpp: number,
  species: string,
  requireStrictCastShadow: boolean
): TreeDetectionBox[] {
  const w = imgData.width;
  const h = imgData.height;
  const px = imgData.data;

  const verified: TreeDetectionBox[] = [];
  let nextId = 1;

  for (const cand of candidates) {
    const cx = Math.round((cand.xmin + cand.xmax) / 2);
    const cy = Math.round((cand.ymin + cand.ymax) / 2);
    if (cx < 2 || cx >= w - 2 || cy < 2 || cy >= h - 2) continue;
    if (!strictMask[cy * w + cx] || !stage1Mask[cy * w + cx]) continue;

    const x1 = Math.max(1, Math.floor(cand.xmin));
    const y1 = Math.max(1, Math.floor(cand.ymin));
    const x2 = Math.min(w - 2, Math.ceil(cand.xmax));
    const y2 = Math.min(h - 2, Math.ceil(cand.ymax));
    const radiusPx = Math.max(3.5, (cand.xmax - cand.xmin + (cand.ymax - cand.ymin)) / 4);
    const diameterM = cand.crownDiameterM ?? Number((radiusPx * 2 * mpp).toFixed(2));

    // Crown center luminance & Laplacian variance
    const coreRad = Math.max(2, Math.round(radiusPx * 0.45));
    let crownLumSum = 0;
    let crownLumCnt = 0;
    let exgSum = 0;
    let greenDomCnt = 0;

    for (let dy = -coreRad; dy <= coreRad; dy++) {
      const py = cy + dy;
      if (py < 0 || py >= h) continue;
      for (let dx = -coreRad; dx <= coreRad; dx++) {
        const pxx = cx + dx;
        if (pxx < 0 || pxx >= w) continue;
        if (dx * dx + dy * dy > coreRad * coreRad + 1) continue;
        const idx = py * w + pxx;
        const o = idx * 4;
        const r = px[o];
        const g = px[o + 1];
        const b = px[o + 2];
        crownLumSum += lum[idx];
        crownLumCnt++;
        exgSum += (2 * g - r - b) / (r + g + b + 1e-6);
        if (g >= r * 0.92) greenDomCnt++;
      }
    }
    if (crownLumCnt === 0) continue;

    const crownLum = crownLumSum / crownLumCnt;
    const meanExg = exgSum / crownLumCnt;
    if (crownLum < 42) continue;
    if (meanExg < 0.01 && greenDomCnt < crownLumCnt * 0.5) continue;

    let lapSum = 0;
    let lapSqSum = 0;
    let boxCnt = 0;
    for (let y = y1; y <= y2; y++) {
      const row = y * w;
      for (let x = x1; x <= x2; x++) {
        const lm = lapMag[row + x];
        lapSum += lm;
        lapSqSum += lm * lm;
        boxCnt++;
      }
    }
    const meanLap = boxCnt > 0 ? lapSum / boxCnt : 0;
    const laplacianVar = boxCnt > 0 ? Math.max(0, lapSqSum / boxCnt - meanLap * meanLap) : 0;

    // Scan 8 directional angular wedges adjacent to crown edge (0.85*R .. 2.65*R)
    const sectorLums: number[][] = Array.from({ length: 8 }, () => []);
    const sectorVs: number[][] = Array.from({ length: 8 }, () => []);
    const rIn = Math.max(2, radiusPx * 0.85);
    const rOut = Math.min(34, Math.max(rIn + 3, radiusPx * 2.65));
    const iROut = Math.ceil(rOut);

    for (let dy = -iROut; dy <= iROut; dy++) {
      const sy = cy + dy;
      if (sy < 0 || sy >= h) continue;
      for (let dx = -iROut; dx <= iROut; dx++) {
        const sx = cx + dx;
        if (sx < 0 || sx >= w) continue;
        const dist = Math.hypot(dx, dy);
        if (dist < rIn || dist > rOut) continue;
        const sIdx = sy * w + sx;
        if (!strictMask[sIdx]) continue;
        let angle = Math.atan2(dy, dx);
        if (angle < 0) angle += Math.PI * 2;
        const sector = Math.floor(((angle + Math.PI / 8) % (Math.PI * 2)) / (Math.PI / 4));
        sectorLums[sector].push(lum[sIdx]);
        sectorVs[sector].push(hsvV[sIdx]);
      }
    }

    let bestSector = -1;
    let bestShadowLum = crownLum;
    let bestShadowV = 1.0;
    const sectorMedians = new Float32Array(8);

    for (let s = 0; s < 8; s++) {
      const arrL = sectorLums[s];
      if (arrL.length < 4) {
        sectorMedians[s] = crownLum;
        continue;
      }
      const sortedL = [...arrL].sort((a, b) => a - b);
      const sortedV = [...sectorVs[s]].sort((a, b) => a - b);
      sectorMedians[s] = sortedL[Math.floor(sortedL.length * 0.5)];
      // Average darkest 25% (minimum 3 pixels) in this directional cast-shadow pocket
      const kDark = Math.max(3, Math.floor(sortedL.length * 0.25));
      let darkLSum = 0;
      let darkVSum = 0;
      for (let k = 0; k < kDark; k++) {
        darkLSum += sortedL[k];
        darkVSum += sortedV[k];
      }
      const pocketL = darkLSum / kDark;
      const pocketV = darkVSum / kDark;
      if (pocketL < bestShadowLum) {
        bestShadowLum = pocketL;
        bestShadowV = pocketV;
        bestSector = s;
      }
    }

    const relShadowDrop = bestSector >= 0 ? (crownLum - bestShadowLum) / crownLum : 0;
    const oppSector = bestSector >= 0 ? (bestSector + 4) % 8 : 0;
    const oppMedianLum = bestSector >= 0 ? sectorMedians[oppSector] : bestShadowLum;
    const isDirectional = oppMedianLum >= bestShadowLum * 1.28;

    const hasPairedCastShadow =
      bestSector >= 0 &&
      relShadowDrop >= MIN_CAST_SHADOW_DROP &&
      bestShadowV <= 0.28 &&
      isDirectional;

    if (requireStrictCastShadow && !hasPairedCastShadow) {
      continue;
    }
    if (!requireStrictCastShadow && relShadowDrop < 0.22 && laplacianVar < 28) {
      continue;
    }

    const onBoundary = Boolean(boundaryMask[cy * w + cx]);
    const fallbackCls = inferFallbackTreeClass(species, onBoundary, diameterM);
    const classId = (cand.classId ?? fallbackCls.classId) as 0 | 1 | 2;
    const className = TREE_CLASSES[classId] || fallbackCls.className;

    const finalScore = Math.min(
      0.99,
      Math.max(0.25, cand.score * (hasPairedCastShadow ? 1.05 : 0.9))
    );

    verified.push({
      id: nextId++,
      xmin: x1,
      ymin: y1,
      xmax: x2,
      ymax: y2,
      cx,
      cy,
      radiusPx: Math.round(radiusPx),
      score: Number(finalScore.toFixed(3)),
      classId,
      className,
      onBoundary,
      shadowDrop: Number(relShadowDrop.toFixed(3)),
      laplacianVar: Number(laplacianVar.toFixed(1)),
      circularity: cand.circularity,
      crownDiameterM: diameterM,
    });
  }

  return verified;
}

// ---------------------------------------------------------------------------
// Clean Image & Annotated Export Rendering (Zero Canvas Text — HUD lives in HTML/CSS)
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
  inwardBufferM: number
): { cleanImageUrl: string; annotatedImageUrl: string } {
  const w = baseCanvas.width;
  const h = baseCanvas.height;

  const cleanCanvas = document.createElement('canvas');
  cleanCanvas.width = w;
  cleanCanvas.height = h;
  const cCtx = cleanCanvas.getContext('2d')!;
  cCtx.drawImage(baseCanvas, 0, 0);

  // Dim outside strict polygon mask
  const frame = cCtx.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let i = 0; i < w * h; i++) {
    if (!strictMask[i]) {
      const o = i * 4;
      px[o] = Math.round(px[o] * 0.45);
      px[o + 1] = Math.round(px[o + 1] * 0.45);
      px[o + 2] = Math.round(px[o + 2] * 0.45);
    }
  }
  cCtx.putImageData(frame, 0, 0);

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

  // Lightbox export image with clean circles ONLY (statistics HUD rendered cleanly in HTML/CSS)
  const annCanvas = document.createElement('canvas');
  annCanvas.width = w;
  annCanvas.height = h;
  const aCtx = annCanvas.getContext('2d')!;
  aCtx.drawImage(cleanCanvas, 0, 0);

  for (const b of boxes) {
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
      px[o] = Math.round(px[o] * 0.45);
      px[o + 1] = Math.round(px[o + 1] * 0.45);
      px[o + 2] = Math.round(px[o + 2] * 0.45);
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

  // Use unaltered raw native Zoom 19 tile pixels for all structural & shadow computations
  const rawCanvas = primaryMosaic.canvas;
  const w = rawCanvas.width;
  const h = rawCanvas.height;
  const rawCtx = rawCanvas.getContext('2d')!;
  const imgData = rawCtx.getImageData(0, 0, w, h);

  // Strict GeoJSON Polygon Masking
  const strictMask = createPolygonMask(w, h, primaryMosaic.clippedRing);
  const { mask: boundaryMask } = createBoundaryCorridorMask(
    w,
    h,
    primaryMosaic.ring,
    primaryMosaic.mpp,
    BOUNDARY_BUFFER_M
  );

  // Compute vegetation percentage inside polygon (diagnostic telemetry only; NOT used for tree counting)
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

  // Stage 1: Flat-Crop Texture & Shadow Rejection (4m x 4m Pre-Filter Grid Mask)
  const stage1 = applyStage1FlatCropMask(imgData, strictMask, primaryMosaic.mpp);
  const stage1MaskedCanvas = buildMaskedCanvas(rawCanvas, stage1.activeMask);
  const slices = generateSahiSlices(w, h, stage1.activeMask, SAHI_TILE_SIZE, SAHI_OVERLAP);

  // Stage 2: True Tree Crown + Cast-Shadow Detection
  const modelStatus = await getModelStatus();
  let detectorEngine = 'DoG Structural Crowns + Cast-Shadow (>=35% Drop)';
  let rawCandidates: RawCandidate[] = [];
  let usedAiModel = false;

  if (slices.length > 0 && modelStatus.onnxReady && modelStatus.onnxUrl) {
    const session = await getOnnxSession(modelStatus.onnxUrl);
    if (session) {
      for (const s of slices) {
        const subCanvas = document.createElement('canvas');
        subCanvas.width = s.w;
        subCanvas.height = s.h;
        subCanvas
          .getContext('2d')!
          .drawImage(stage1MaskedCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
        const tileCands = await runOnnxTileInference(
          session,
          subCanvas,
          s.x,
          s.y,
          AI_MODEL_CONF_THRESH
        );
        rawCandidates.push(...tileCands);
      }
      usedAiModel = true;
      detectorEngine = 'YOLOv8 ONNX (SAHI 512px · Conf>=0.55 · Stage-1 Masked)';
    }
  }

  if (!usedAiModel && slices.length > 0 && modelStatus.backendYoloReady) {
    let backendWorked = false;
    for (const s of slices) {
      const subCanvas = document.createElement('canvas');
      subCanvas.width = s.w;
      subCanvas.height = s.h;
      subCanvas
        .getContext('2d')!
        .drawImage(stage1MaskedCanvas, s.x, s.y, s.w, s.h, 0, 0, s.w, s.h);
      const res = await runBackendYoloTile(subCanvas, s.x, s.y);
      if (res.ok) {
        backendWorked = true;
        rawCandidates.push(...res.cands);
        detectorEngine = `${res.engine} (SAHI 512px · Conf>=0.55 · Stage-1 Masked)`;
      } else {
        break;
      }
    }
    usedAiModel = backendWorked;
    if (!backendWorked) rawCandidates = [];
  }

  // Algorithmic Fallback (when ONNX/PyTorch weights are unavailable)
  if (!usedAiModel && slices.length > 0) {
    rawCandidates = detectStructuralCrownsDoG(
      imgData,
      stage1.lum,
      stage1.activeMask,
      primaryMosaic.mpp,
      species,
      boundaryMask
    );
  }

  const minCenterDistPx = Math.max(6, Math.round(2.4 / Math.max(primaryMosaic.mpp, 0.12)));
  const nmsMerged = mergeDetectionsNMS(rawCandidates, NMS_IOU_THRESH, minCenterDistPx);

  // Enforce paired directional cast shadow (>= 35% luminance drop relative to crown center)
  const boxes = verifyCastShadowAndTelemetry(
    nmsMerged,
    imgData,
    strictMask,
    stage1.activeMask,
    boundaryMask,
    stage1.lum,
    stage1.hsvV,
    stage1.lapMag,
    primaryMosaic.mpp,
    species,
    !usedAiModel
  );

  const boundaryBoxes = boxes.filter((b) => b.onBoundary);
  const interiorBoxes = boxes.filter((b) => !b.onBoundary);
  const primaryBoxes = inwardBufferM < -0.5 ? boxes : boundaryBoxes;

  const counts = {
    lo: primaryBoxes.length,
    mid: primaryBoxes.length,
    hi: primaryBoxes.length,
    interiorMid: interiorBoxes.length,
  };

  let crownM: number | null = null;
  if (primaryBoxes.length > 0) {
    const diams = primaryBoxes
      .map((b) => b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * primaryMosaic!.mpp)
      .sort((a, b) => a - b);
    crownM = Number(diams[Math.floor(diams.length / 2)].toFixed(2));
  }

  const factor = calibFactor(species);
  const aiLow = Math.round(counts.lo * factor);
  const aiCount = Math.round(counts.mid * factor);
  const aiHigh = Math.round(counts.hi * factor);

  const { cleanImageUrl, annotatedImageUrl } = renderCleanAndAnnotatedCanvases(
    rawCanvas,
    strictMask,
    primaryMosaic.ring,
    primaryMosaic.clippedRing,
    boxes,
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
    stage1ActivePct: stage1.activePct,
    stage1CellsKept: stage1.cellsKept,
    stage1CellsTotal: stage1.cellsTotal,
  };
}
