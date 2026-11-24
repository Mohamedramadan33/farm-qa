import { BOUNDARY_BUFFER_M } from '../../lib/geometry';
import { DETECTION_CONFIG } from './config';
import type { TreeHealthStatus } from '../../types/qa';

export interface SpectralMaps {
  lum: Float32Array;
  hsvV: Float32Array;
  hsvS: Float32Array;
  exg: Float32Array;
  lapMag: Float32Array;
}

export function createPolygonMask(w: number, h: number, ring: Array<[number, number]>): Uint8Array {
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

export function createBoundaryCorridorMask(
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

export function buildMaskedCanvas(srcCanvas: HTMLCanvasElement, mask: Uint8Array): HTMLCanvasElement {
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

/**
 * Exact O(W*H) 2-pass Meijster / Felzenszwalb Euclidean Distance Transform.
 */
export function computeEuclideanDistanceTransform(mask: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e9;
  const g = new Float32Array(w * h);

  for (let x = 0; x < w; x++) {
    g[x] = mask[x] ? INF : 0;
    for (let y = 1; y < h; y++) {
      const idx = y * w + x;
      g[idx] = mask[idx] ? Math.min(INF, g[(y - 1) * w + x] + 1) : 0;
    }
    for (let y = h - 2; y >= 0; y--) {
      const idx = y * w + x;
      const below = g[(y + 1) * w + x] + 1;
      if (below < g[idx]) g[idx] = below;
    }
  }

  const dt = new Float32Array(w * h);
  const s = new Int32Array(w);
  const t = new Int32Array(w);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    let q = 0;
    s[0] = 0;
    t[0] = 0;

    const f = (x: number, i: number) => {
      const gi = g[row + i];
      return (x - i) * (x - i) + gi * gi;
    };

    const sep = (i: number, u: number) => {
      const gi = g[row + i];
      const gu = g[row + u];
      return Math.floor((u * u - i * i + gu * gu - gi * gi) / (2 * (u - i)));
    };

    for (let u = 1; u < w; u++) {
      while (q >= 0 && f(t[q], s[q]) > f(t[q], u)) {
        q--;
      }
      if (q < 0) {
        q = 0;
        s[0] = u;
        t[0] = 0;
      } else {
        const wSep = 1 + sep(s[q], u);
        if (wSep < w) {
          q++;
          s[q] = u;
          t[q] = wSep;
        }
      }
    }

    for (let u = w - 1; u >= 0; u--) {
      const val = f(u, s[q]);
      dt[row + u] = val >= INF * 0.5 ? 0 : Math.sqrt(val);
      if (u === t[q]) q--;
    }
  }

  return dt;
}

export function computeLuminanceAndLaplacian(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  strictMask: Uint8Array
): SpectralMaps {
  const n = w * h;
  const lum = new Float32Array(n);
  const hsvV = new Float32Array(n);
  const hsvS = new Float32Array(n);
  const exg = new Float32Array(n);
  const lapMag = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    hsvV[i] = maxC / 255.0;
    hsvS[i] = maxC > 0 ? (maxC - minC) / maxC : 0;
    exg[i] = (2 * g - r - b) / (r + g + b + 1e-6);
  }

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

  return { lum, hsvV, hsvS, exg, lapMag };
}

/**
 * Offloads heavy pixel-level spectral + Laplacian passes to an inline Web Worker on large mosaics
 * (>=200k pixels), falling back cleanly to synchronous computation in Node/test environments.
 */
export async function computeLuminanceAndLaplacianAsync(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  strictMask: Uint8Array
): Promise<SpectralMaps> {
  if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || w * h < 200_000) {
    return computeLuminanceAndLaplacian(rgba, w, h, strictMask);
  }

  const workerSrc = `
    self.onmessage = function(ev) {
      var d = ev.data;
      var rgba = new Uint8ClampedArray(d.rgba);
      var strictMask = new Uint8Array(d.strictMask);
      var w = d.w, h = d.h, n = w * h;
      var lum = new Float32Array(n);
      var hsvV = new Float32Array(n);
      var hsvS = new Float32Array(n);
      var exg = new Float32Array(n);
      var lapMag = new Float32Array(n);
      for (var i = 0; i < n; i++) {
        var o = i * 4;
        var r = rgba[o], g = rgba[o + 1], b = rgba[o + 2];
        lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        var maxC = Math.max(r, g, b), minC = Math.min(r, g, b);
        hsvV[i] = maxC / 255.0;
        hsvS[i] = maxC > 0 ? (maxC - minC) / maxC : 0;
        exg[i] = (2 * g - r - b) / (r + g + b + 1e-6);
      }
      for (var y = 1; y < h - 1; y++) {
        var row = y * w;
        for (var x = 1; x < w - 1; x++) {
          var idx = row + x;
          if (!strictMask[idx]) continue;
          lapMag[idx] = Math.abs(lum[idx - 1] + lum[idx + 1] + lum[idx - w] + lum[idx + w] - 4 * lum[idx]);
        }
      }
      self.postMessage(
        { lum: lum.buffer, hsvV: hsvV.buffer, hsvS: hsvS.buffer, exg: exg.buffer, lapMag: lapMag.buffer },
        [lum.buffer, hsvV.buffer, hsvS.buffer, exg.buffer, lapMag.buffer]
      );
    };
  `;

  try {
    const blob = new Blob([workerSrc], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    return await new Promise<SpectralMaps>((resolve) => {
      const worker = new Worker(url);
      const rgbaCopy = new Uint8ClampedArray(rgba).buffer;
      const maskCopy = new Uint8Array(strictMask).buffer;

      worker.onmessage = (ev) => {
        URL.revokeObjectURL(url);
        worker.terminate();
        const out = ev.data;
        resolve({
          lum: new Float32Array(out.lum),
          hsvV: new Float32Array(out.hsvV),
          hsvS: new Float32Array(out.hsvS),
          exg: new Float32Array(out.exg),
          lapMag: new Float32Array(out.lapMag),
        });
      };
      worker.onerror = () => {
        URL.revokeObjectURL(url);
        worker.terminate();
        resolve(computeLuminanceAndLaplacian(rgba, w, h, strictMask));
      };

      worker.postMessage({ rgba: rgbaCopy, strictMask: maskCopy, w, h }, [rgbaCopy, maskCopy]);
    });
  } catch {
    return computeLuminanceAndLaplacian(rgba, w, h, strictMask);
  }
}

/**
 * Downsampled 2x-4x DoG (Difference-of-Gaussians) coarse canopy response map
 * to accelerate coarse vegetation/texture screening on large tiles.
 */
export function estimateDownsampledDoGResponse(
  lum: Float32Array,
  w: number,
  h: number,
  stride = 2
): { dw: number; dh: number; dog: Float32Array } {
  const s = Math.max(2, Math.min(4, stride));
  const dw = Math.max(4, Math.floor(w / s));
  const dh = Math.max(4, Math.floor(h / s));
  const small = new Float32Array(dw * dh);

  for (let dy = 0; dy < dh; dy++) {
    const sy = Math.min(h - 1, dy * s);
    for (let dx = 0; dx < dw; dx++) {
      const sx = Math.min(w - 1, dx * s);
      small[dy * dw + dx] = lum[sy * w + sx];
    }
  }

  const boxBlur = (src: Float32Array, rad: number): Float32Array => {
    const tmp = new Float32Array(dw * dh);
    const out = new Float32Array(dw * dh);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        let sum = 0;
        let cnt = 0;
        for (let k = -rad; k <= rad; k++) {
          const xx = Math.max(0, Math.min(dw - 1, x + k));
          sum += src[y * dw + xx];
          cnt++;
        }
        tmp[y * dw + x] = sum / cnt;
      }
    }
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        let sum = 0;
        let cnt = 0;
        for (let k = -rad; k <= rad; k++) {
          const yy = Math.max(0, Math.min(dh - 1, y + k));
          sum += tmp[yy * dw + x];
          cnt++;
        }
        out[y * dw + x] = sum / cnt;
      }
    }
    return out;
  };

  const fine = boxBlur(small, 1);
  const coarse = boxBlur(small, 3);
  const dog = new Float32Array(dw * dh);
  for (let i = 0; i < dw * dh; i++) {
    dog[i] = fine[i] - coarse[i];
  }
  return { dw, dh, dog };
}

/**
 * Computes Gate 1 (ExG + HSV S/V) and Gate 2 (>85 m^2 crop sheet rejection) vegetation mask.
 * Used as the primary candidate mask in the 4-Gate morphological fallback AND as a non-blocking
 * diagnostic audit metric on trained AI paths.
 */
export function computeStage1VegetationMask(params: {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  strictMask: Uint8Array;
  mpp: number;
  maps: SpectralMaps;
}): {
  vegMask: Uint8Array;
  gate1VegPixels: number;
  gate2ComponentsRejected: number;
  gate2KeptPixels: number;
  gate2ActivePct: number;
} {
  const { rgba, width: w, height: h, strictMask, mpp, maps } = params;
  const { lum, hsvV, hsvS, exg, lapMag } = maps;
  const n = w * h;

  const vegMask = new Uint8Array(n);
  let gate1VegPixels = 0;
  let strictTotal = 0;

  for (let i = 0; i < n; i++) {
    if (!strictMask[i]) continue;
    strictTotal++;
    const o = i * 4;
    const r = rgba[o];
    const g = rgba[o + 1];
    const b = rgba[o + 2];
    const v = hsvV[i];
    const s = hsvS[i];
    const e = exg[i];
    if (e > 0.045 && g > r * 0.94 && s >= 0.14 && v >= 0.16 && v <= 0.88 && b <= g * 1.05) {
      vegMask[i] = 1;
      gate1VegPixels++;
    }
  }

  const pxAreaM2 = mpp * mpp;
  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);
  let gate2ComponentsRejected = 0;

  for (let seed = 0; seed < n; seed++) {
    if (!vegMask[seed] || visited[seed]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    visited[seed] = 1;

    let minX = w;
    let minY = h;
    let maxX = 0;
    let maxY = 0;
    let lumSum = 0;

    while (head < tail) {
      const cur = queue[head++];
      const cy = Math.floor(cur / w);
      const cx = cur - cy * w;
      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      lumSum += lum[cur];

      if (cx + 1 < w && vegMask[cur + 1] && !visited[cur + 1]) {
        visited[cur + 1] = 1;
        queue[tail++] = cur + 1;
      }
      if (cx - 1 >= 0 && vegMask[cur - 1] && !visited[cur - 1]) {
        visited[cur - 1] = 1;
        queue[tail++] = cur - 1;
      }
      if (cy + 1 < h && vegMask[cur + w] && !visited[cur + w]) {
        visited[cur + w] = 1;
        queue[tail++] = cur + w;
      }
      if (cy - 1 >= 0 && vegMask[cur - w] && !visited[cur - w]) {
        visited[cur - w] = 1;
        queue[tail++] = cur - w;
      }
    }

    const compAreaM2 = tail * pxAreaM2;
    if (compAreaM2 > DETECTION_CONFIG.cropSheetAreaM2) {
      const compMeanLum = lumSum / Math.max(1, tail);
      const shadowCutoff = compMeanLum * (1 - DETECTION_CONFIG.minCastShadowDrop);
      let bboxValid = 0;
      let deepShadowCount = 0;
      let lapSum = 0;
      let lapSqSum = 0;

      for (let y = minY; y <= maxY; y++) {
        const row = y * w;
        for (let x = minX; x <= maxX; x++) {
          const idx = row + x;
          if (!strictMask[idx]) continue;
          bboxValid++;
          if (lum[idx] <= shadowCutoff && hsvV[idx] < 0.25) {
            deepShadowCount++;
          }
          const lm = lapMag[idx];
          lapSum += lm;
          lapSqSum += lm * lm;
        }
      }

      const shadowRatio = bboxValid > 0 ? deepShadowCount / bboxValid : 0;
      const meanLap = bboxValid > 0 ? lapSum / bboxValid : 0;
      const lapVar = bboxValid > 0 ? Math.max(0, lapSqSum / bboxValid - meanLap * meanLap) : 0;

      if (shadowRatio < 0.025 || lapVar < 36.0) {
        gate2ComponentsRejected++;
        for (let k = 0; k < tail; k++) {
          vegMask[queue[k]] = 0;
        }
      }
    }
  }

  let gate2KeptPixels = 0;
  for (let i = 0; i < n; i++) {
    if (vegMask[i]) gate2KeptPixels++;
  }
  const gate2ActivePct =
    strictTotal > 0 ? Number(((gate2KeptPixels / strictTotal) * 100).toFixed(1)) : 0;

  return {
    vegMask,
    gate1VegPixels,
    gate2ComponentsRejected,
    gate2KeptPixels,
    gate2ActivePct,
  };
}

export function evaluateCrownShadowAndHealth(
  cx: number,
  cy: number,
  radiusPx: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  w: number,
  h: number,
  rgba: Uint8ClampedArray,
  strictMask: Uint8Array,
  lum: Float32Array,
  hsvV: Float32Array,
  exg: Float32Array,
  lapMag: Float32Array
): {
  crownLum: number;
  meanExg: number;
  laplacianVar: number;
  relShadowDrop: number;
  hasPairedCastShadow: boolean;
  health: TreeHealthStatus;
} | null {
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
      const r = rgba[o];
      const g = rgba[o + 1];
      crownLumSum += lum[idx];
      crownLumCnt++;
      exgSum += exg[idx];
      if (g >= r * 0.92) greenDomCnt++;
    }
  }
  if (crownLumCnt === 0) return null;

  const crownLum = crownLumSum / crownLumCnt;
  const meanExg = exgSum / crownLumCnt;
  if (crownLum < 38) return null;

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

  const sectorLums: number[][] = Array.from({ length: 8 }, () => []);
  const sectorVs: number[][] = Array.from({ length: 8 }, () => []);
  const rIn = Math.max(2, radiusPx * 0.85);
  const rOut = Math.min(36, Math.max(rIn + 3, radiusPx * 2.75));
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
  const isDirectional = oppMedianLum >= bestShadowLum * 1.25;

  const hasPairedCastShadow =
    bestSector >= 0 &&
    relShadowDrop >= DETECTION_CONFIG.minCastShadowDrop &&
    bestShadowV <= 0.29 &&
    isDirectional;

  const health: TreeHealthStatus =
    meanExg >= 0.14 && greenDomCnt >= crownLumCnt * 0.65
      ? 'healthy'
      : meanExg >= 0.055
        ? 'stressed'
        : 'dead';

  return {
    crownLum,
    meanExg,
    laplacianVar,
    relShadowDrop,
    hasPairedCastShadow,
    health,
  };
}
