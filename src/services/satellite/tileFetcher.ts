import {
  polygonBounds,
  polygonCentroid,
} from '../../lib/geometry';
import type {
  BasemapProvider,
  PointLonLat,
  SatelliteProgress,
} from '../../types/qa';

export function lonLatToWorldPx(lon: number, lat: number, z: number): [number, number] {
  const n = 256 * Math.pow(2, z);
  const x = ((lon + 180.0) / 360.0) * n;
  const latRad = (lat * Math.PI) / 180.0;
  const y = ((1.0 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2.0) * n;
  return [x, y];
}

export function worldPxToLonLat(wx: number, wy: number, z: number): [number, number] {
  const n = 256 * Math.pow(2, z);
  const lon = (wx / n) * 360.0 - 180.0;
  const latRad = Math.atan(Math.sinh(Math.PI * (1.0 - (2.0 * wy) / n)));
  const lat = (latRad * 180.0) / Math.PI;
  return [Number(lon.toFixed(7)), Number(lat.toFixed(7))];
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Satellite analysis aborted by user.', 'AbortError');
  }
}

function loadTileImage(
  url: string,
  timeoutMs = 11000,
  signal?: AbortSignal
): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(null);
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    let settled = false;

    const cleanup = () => {
      window.clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    };

    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      img.src = '';
      resolve(null);
    };

    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      img.src = '';
      resolve(null);
    }, timeoutMs);

    signal?.addEventListener('abort', onAbort, { once: true });

    img.onload = () => {
      if (settled) return;
      settled = true;
      cleanup();
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        resolve(img);
      } else {
        resolve(null);
      }
    };
    img.onerror = () => {
      if (settled) return;
      settled = true;
      cleanup();
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
  concurrency = 8,
  signal?: AbortSignal,
  onProgress?: (p: SatelliteProgress) => void
): Promise<Array<{ tx: number; ty: number; img: HTMLImageElement | null }>> {
  const results: Array<{ tx: number; ty: number; img: HTMLImageElement | null }> = new Array(
    coords.length
  );
  let cursor = 0;
  let completed = 0;
  const fallbackFlag = strictProvider ? '0' : '1';

  onProgress?.({
    stage: 'tiles',
    loaded: 0,
    total: coords.length,
    message: `Fetching ${coords.length} tiles (Z${z})...`,
  });

  async function worker() {
    while (true) {
      throwIfAborted(signal);
      const idx = cursor++;
      if (idx >= coords.length) break;
      const [tx, ty] = coords[idx];
      let img = await loadTileImage(
        `/api/satellite/tile/${provider}/${z}/${tx}/${ty}?fallback=${fallbackFlag}`,
        11000,
        signal
      );
      throwIfAborted(signal);
      if (img && !isValidTileCanvas(img)) {
        if (!strictProvider) {
          const alt: BasemapProvider = provider === 'google' ? 'esri' : 'google';
          const altImg = await loadTileImage(
            `/api/satellite/tile/${alt}/${z}/${tx}/${ty}?fallback=0`,
            11000,
            signal
          );
          img = altImg && isValidTileCanvas(altImg) ? altImg : null;
        } else {
          img = null;
        }
      }
      results[idx] = { tx, ty, img };
      completed++;
      onProgress?.({
        stage: 'tiles',
        loaded: completed,
        total: coords.length,
        message: `Loaded ${completed}/${coords.length} tiles (Z${z})`,
      });
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, coords.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

export interface FetchedMosaic {
  canvas: HTMLCanvasElement;
  ring: Array<[number, number]>;
  clippedRing: Array<[number, number]>;
  zoom: number;
  provider: string;
  providerId: BasemapProvider;
  mpp: number;
  worldOriginX: number;
  worldOriginY: number;
}

export async function fetchProviderMosaic(
  rawPoly: PointLonLat[],
  calibratedPoly: PointLonLat[],
  clippedPoly: PointLonLat[],
  provider: BasemapProvider,
  strictOnly: boolean,
  options?: {
    signal?: AbortSignal;
    onProgress?: (p: SatelliteProgress) => void;
  }
): Promise<FetchedMosaic | null> {
  const signal = options?.signal;
  const onProgress = options?.onProgress;
  throwIfAborted(signal);

  const [minx, miny, maxx, maxy] = polygonBounds(calibratedPoly);
  const dx = Math.max((maxx - minx) * 0.08, 0.00014);
  const dy = Math.max((maxy - miny) * 0.08, 0.00014);
  const bMinx = minx - dx;
  const bMiny = miny - dy;
  const bMaxx = maxx + dx;
  const bMaxy = maxy + dy;

  const startZoom = 19;
  const providerLabel = provider === 'google' ? 'Google Satellite' : 'Esri World Imagery';

  let bestCandidate: (FetchedMosaic & { loadedRatio: number }) | null = null;

  for (let z = startZoom; z >= 15; z--) {
    throwIfAborted(signal);
    const [x0, y1] = lonLatToWorldPx(bMinx, bMiny, z);
    const [x1, y0] = lonLatToWorldPx(bMaxx, bMaxy, z);
    const tx0 = Math.floor(x0 / 256);
    const tx1 = Math.floor(x1 / 256);
    const ty0 = Math.floor(y0 / 256);
    const ty1 = Math.floor(y1 / 256);

    const tilesX = tx1 - tx0 + 1;
    const tilesY = ty1 - ty0 + 1;
    const tilesCount = tilesX * tilesY;
    const maxAllowed = z === 19 ? 256 : 121;
    if (tilesCount > maxAllowed || tilesCount <= 0) continue;

    const coords: Array<[number, number]> = [];
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        coords.push([tx, ty]);
      }
    }

    const loaded = await fetchTilesPooled(coords, z, provider, strictOnly, 8, signal, onProgress);
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

    const worldOriginX = ox + cropX;
    const worldOriginY = oy + cropY;

    const projectRing = (pList: PointLonLat[]): Array<[number, number]> =>
      pList.map(([lo, la]) => {
        const [px, py] = lonLatToWorldPx(lo, la, z);
        return [px - worldOriginX, py - worldOriginY];
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
      worldOriginX,
      worldOriginY,
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
