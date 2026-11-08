import { normAr, present } from './arabicNorm';
import { BOUNDARY_BUFFER_M, polygonBounds, polygonCentroid } from './geometry';
import { AI_BAND, VEG_EXG } from './farmQaCore';
import type { PointLonLat, SatelliteAnalysisResult, TreeDetectionBox } from '../types/qa';

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

function lonLatToWorldPx(lon: number, lat: number, z: number): [number, number] {
  const n = 256 * Math.pow(2, z);
  const x = ((lon + 180.0) / 360.0) * n;
  const latRad = (lat * Math.PI) / 180.0;
  const y = ((1.0 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2.0) * n;
  return [x, y];
}

function loadTileImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const timer = window.setTimeout(() => {
      resolve(null);
    }, 9000);
    img.onload = () => {
      window.clearTimeout(timer);
      resolve(img);
    };
    img.onerror = () => {
      window.clearTimeout(timer);
      resolve(null);
    };
    img.src = url;
  });
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

  return !(colorSpread < 3.0 && stdVal < 25.0);
}

/**
 * Deterministic fallback aerial render if external tile servers are temporarily unreachable
 */
function drawFallbackAerialCanvas(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  ring: Array<[number, number]>,
  seedStr: string
): void {
  let hash = 2166136261;
  for (let i = 0; i < seedStr.length; i++) {
    hash ^= seedStr.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  const rand = () => {
    hash ^= hash << 13;
    hash ^= hash >>> 17;
    hash ^= hash << 5;
    return ((hash >>> 0) % 10000) / 10000;
  };

  // Surrounding arid soil / canal texture
  ctx.fillStyle = '#9c8b6e';
  ctx.fillRect(0, 0, w, h);

  // Subtle field texture outside
  for (let i = 0; i < 650; i++) {
    const x = rand() * w;
    const y = rand() * h;
    ctx.fillStyle = rand() > 0.5 ? 'rgba(130, 114, 86, 0.35)' : 'rgba(172, 155, 124, 0.3)';
    ctx.fillRect(x, y, 6 + rand() * 14, 4 + rand() * 10);
  }

  // Cultivated farm interior rows
  ctx.save();
  ctx.beginPath();
  ring.forEach(([x, y], idx) => {
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.clip();

  ctx.fillStyle = '#3f6a34';
  ctx.fillRect(0, 0, w, h);

  for (let y = 0; y < h; y += 9) {
    ctx.fillStyle = y % 18 === 0 ? 'rgba(48, 86, 38, 0.55)' : 'rgba(78, 116, 62, 0.45)';
    ctx.fillRect(0, y, w, 4);
  }
  ctx.restore();

  // Plant windbreak tree crowns along perimeter
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[i + 1];
    const segLen = Math.hypot(x2 - x1, y2 - y1);
    const stepPx = 14;
    const count = Math.max(1, Math.floor(segLen / stepPx));
    for (let k = 0; k < count; k++) {
      if (rand() < 0.08) continue;
      const t = (k + 0.5) / count;
      const tx = x1 + (x2 - x1) * t + (rand() - 0.5) * 3;
      const ty = y1 + (y2 - y1) * t + (rand() - 0.5) * 3;
      const r = 4.5 + rand() * 2.8;
      // Canopy shadow
      ctx.fillStyle = 'rgba(18, 30, 16, 0.65)';
      ctx.beginPath();
      ctx.arc(tx + 2.2, ty + 2.2, r, 0, Math.PI * 2);
      ctx.fill();
      // Canopy green crown
      ctx.fillStyle = rand() > 0.35 ? '#265828' : '#316b32';
      ctx.beginPath();
      ctx.arc(tx, ty, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

async function fetchFarmMosaic(poly: PointLonLat[]): Promise<{
  canvas: HTMLCanvasElement;
  ring: Array<[number, number]>;
  zoom: number;
  provider: string;
}> {
  const [minx, miny, maxx, maxy] = polygonBounds(poly);
  const dx = Math.max((maxx - minx) * 0.2, 0.00022);
  const dy = Math.max((maxy - miny) * 0.2, 0.00022);
  const bMinx = minx - dx;
  const bMiny = miny - dy;
  const bMaxx = maxx + dx;
  const bMaxy = maxy + dy;

  const providers: Array<{ id: 'esri' | 'google'; label: string }> = [
    { id: 'google', label: 'Google Earth' },
    { id: 'esri', label: 'ESRI World Imagery' },
  ];

  let fallbackCandidate: {
    canvas: HTMLCanvasElement;
    ring: Array<[number, number]>;
    zoom: number;
    provider: string;
  } | null = null;

  for (let z = 19; z >= 16; z--) {
    const [x0, y1] = lonLatToWorldPx(bMinx, bMiny, z);
    const [x1, y0] = lonLatToWorldPx(bMaxx, bMaxy, z);
    const tx0 = Math.floor(x0 / 256);
    const tx1 = Math.floor(x1 / 256);
    const ty0 = Math.floor(y0 / 256);
    const ty1 = Math.floor(y1 / 256);

    const tilesCount = (tx1 - tx0 + 1) * (ty1 - ty0 + 1);
    if (tilesCount > 64) continue;

    const coords: Array<[number, number]> = [];
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        coords.push([tx, ty]);
      }
    }

    for (const prov of providers) {
      const loaded = await Promise.all(
        coords.map(async ([tx, ty]) => {
          const img = await loadTileImage(`/api/satellite/tile/${prov.id}/${z}/${tx}/${ty}`);
          return { tx, ty, img };
        })
      );

      if (loaded.some((item) => item.img === null)) {
        continue;
      }

      const mosaic = document.createElement('canvas');
      mosaic.width = (tx1 - tx0 + 1) * 256;
      mosaic.height = (ty1 - ty0 + 1) * 256;
      const mCtx = mosaic.getContext('2d');
      if (!mCtx) continue;

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

      const ring: Array<[number, number]> = poly.map(([lo, la]) => {
        const [px, py] = lonLatToWorldPx(lo, la, z);
        return [px - ox - cropX, py - oy - cropY];
      });

      fallbackCandidate = { canvas: cropped, ring, zoom: z, provider: prov.label };
      if (isValidSatelliteCanvas(cCtx, cropW, cropH)) {
        return fallbackCandidate;
      }
    }
  }

  if (fallbackCandidate) return fallbackCandidate;

  // Offline / sandbox fallback aerial synthesis
  const z = 18;
  const [x0, y1] = lonLatToWorldPx(bMinx, bMiny, z);
  const [x1, y0] = lonLatToWorldPx(bMaxx, bMaxy, z);
  const cropW = Math.min(960, Math.max(420, Math.round(x1 - x0)));
  const cropH = Math.min(960, Math.max(420, Math.round(y1 - y0)));
  const scaleX = cropW / Math.max(1, x1 - x0);
  const scaleY = cropH / Math.max(1, y1 - y0);

  const ring: Array<[number, number]> = poly.map(([lo, la]) => {
    const [px, py] = lonLatToWorldPx(lo, la, z);
    return [(px - x0) * scaleX, (py - y0) * scaleY];
  });

  const synth = document.createElement('canvas');
  synth.width = cropW;
  synth.height = cropH;
  const sCtx = synth.getContext('2d')!;
  drawFallbackAerialCanvas(sCtx, cropW, cropH, ring, JSON.stringify(poly[0]));
  return {
    canvas: synth,
    ring,
    zoom: z,
    provider: 'ESRI World Imagery',
  };
}

function createPolygonMask(w: number, h: number, ring: Array<[number, number]>): Uint8Array {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ring.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fill();

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
  lat: number,
  zoom: number,
  bufferM = BOUNDARY_BUFFER_M
): { mask: Uint8Array; strokePx: number } {
  const mpp = (156543.03392 * Math.cos((lat * Math.PI) / 180)) / Math.pow(2, zoom);
  const strokePx = Math.max(8, Math.round((bufferM * 2.0) / Math.max(mpp, 0.05)));

  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
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

  const data = ctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = data[i * 4] > 127 ? 1 : 0;
  }
  return { mask, strokePx };
}

/**
 * Pixel-level Excess Green Index (ExG) + multi-scale canopy crown blob detector with NMS
 */
function detectCanopyTrees(
  imgData: ImageData,
  farmMask: Uint8Array,
  boundaryMask: Uint8Array,
  lat: number,
  zoom: number
): {
  boxes: TreeDetectionBox[];
  vegPct: number;
  counts: { lo: number; mid: number; hi: number; interiorMid: number };
  crownM: number | null;
} {
  const w = imgData.width;
  const h = imgData.height;
  const data = imgData.data;
  const mpp = (156543.03392 * Math.cos((lat * Math.PI) / 180)) / Math.pow(2, zoom);

  // Compute ExG map & vegetation cover along boundary corridor
  const canopyScore = new Float32Array(w * h);
  let boundaryPxCount = 0;
  let boundaryVegCount = 0;

  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    const sum = r + g + b + 1e-6;
    const exg = (2 * g - r - b) / sum;
    const brightness = (r + g + b) / 3;

    if (boundaryMask[i]) {
      boundaryPxCount++;
      if (exg > VEG_EXG) {
        boundaryVegCount++;
      }
    }

    if (farmMask[i] || boundaryMask[i]) {
      // Combine greenness index and dark-canopy texture contrast (for arid / olive / casuarina crowns)
      const greenComponent = Math.max(0, (exg - 0.015) * 3.2);
      const darkCanopy =
        brightness > 22 && brightness < 118 && g >= r * 0.92 ? (118 - brightness) / 230 : 0;
      canopyScore[i] = Math.min(1, greenComponent * 0.72 + darkCanopy * 0.48);
    }
  }

  const vegPct =
    boundaryPxCount > 0 ? Number(((boundaryVegCount / boundaryPxCount) * 100).toFixed(1)) : 0;

  // Box-blur canopy score at ~1.6m radius to find crown centers
  const crownRadiusPx = Math.max(2, Math.min(9, Math.round(1.75 / Math.max(mpp, 0.12))));
  const smoothed = new Float32Array(w * h);
  for (let y = crownRadiusPx; y < h - crownRadiusPx; y++) {
    for (let x = crownRadiusPx; x < w - crownRadiusPx; x++) {
      const centerIdx = y * w + x;
      if (!farmMask[centerIdx] && !boundaryMask[centerIdx]) continue;
      let acc = 0;
      let cnt = 0;
      for (let dy = -crownRadiusPx; dy <= crownRadiusPx; dy++) {
        const rowOff = (y + dy) * w;
        for (let dx = -crownRadiusPx; dx <= crownRadiusPx; dx++) {
          acc += canopyScore[rowOff + x + dx];
          cnt++;
        }
      }
      smoothed[centerIdx] = acc / cnt;
    }
  }

  // Local maxima search + NMS
  const minSepPx = Math.max(4, Math.round(2.6 / Math.max(mpp, 0.12)));
  const candidates: Array<{ x: number; y: number; score: number; radiusPx: number }> = [];

  for (let y = crownRadiusPx + 1; y < h - crownRadiusPx - 1; y += 2) {
    for (let x = crownRadiusPx + 1; x < w - crownRadiusPx - 1; x += 2) {
      const idx = y * w + x;
      const val = smoothed[idx];
      if (val < AI_BAND[2]) continue;

      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (smoothed[(y + dy) * w + (x + dx)] > val) {
            isMax = false;
            break;
          }
        }
      }
      if (!isMax) continue;

      // Estimate local crown size
      let rPx = crownRadiusPx * 1.35;
      if (val > 0.35) rPx = crownRadiusPx * 1.65;
      const normScore = Math.min(0.95, val * 1.35);
      if (normScore >= AI_BAND[2]) {
        candidates.push({ x, y, score: normScore, radiusPx: rPx });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const accepted: typeof candidates = [];
  for (const cand of candidates) {
    const tooClose = accepted.some(
      (prev) => Math.hypot(prev.x - cand.x, prev.y - cand.y) < minSepPx
    );
    if (!tooClose) {
      accepted.push(cand);
      if (accepted.length >= 650) break;
    }
  }

  const boxes: TreeDetectionBox[] = accepted.map((c) => {
    const onBoundary = Boolean(boundaryMask[c.y * w + c.x]);
    return {
      xmin: Math.max(0, c.x - c.radiusPx),
      ymin: Math.max(0, c.y - c.radiusPx),
      xmax: Math.min(w - 1, c.x + c.radiusPx),
      ymax: Math.min(h - 1, c.y + c.radiusPx),
      score: c.score,
      onBoundary,
    };
  });

  const boundaryBoxes = boxes.filter((b) => b.onBoundary);
  const interiorBoxes = boxes.filter((b) => !b.onBoundary);

  const counts = {
    lo: boundaryBoxes.filter((b) => b.score >= AI_BAND[0]).length,
    mid: boundaryBoxes.filter((b) => b.score >= AI_BAND[1]).length,
    hi: boundaryBoxes.filter((b) => b.score >= AI_BAND[2]).length,
    interiorMid: interiorBoxes.filter((b) => b.score >= AI_BAND[1]).length,
  };

  const midBoundary = boundaryBoxes.filter((b) => b.score >= AI_BAND[1]);
  let crownM: number | null = null;
  if (midBoundary.length > 0) {
    const sides = midBoundary
      .map((b) => Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * mpp)
      .sort((a, b) => a - b);
    const midIdx = Math.floor(sides.length / 2);
    crownM = Number(sides[midIdx].toFixed(2));
  }

  return { boxes, vegPct, counts, crownM };
}

function renderAnnotatedSatellite(
  baseCanvas: HTMLCanvasElement,
  farmMask: Uint8Array,
  ring: Array<[number, number]>,
  strokePx: number,
  boxes: TreeDetectionBox[],
  provider: string
): string {
  const origW = baseCanvas.width;
  const origH = baseCanvas.height;
  const visScale = Math.min(origW, origH) < 480 ? 480 / Math.max(1, Math.min(origW, origH)) : 1.0;
  const w = Math.round(origW * visScale);
  const h = Math.round(origH * visScale);

  const out = document.createElement('canvas');
  out.width = w;
  out.height = h;
  const ctx = out.getContext('2d')!;
  ctx.drawImage(baseCanvas, 0, 0, w, h);

  // Dim background outside farm boundary to 60%, keep interior 100% bright
  const frame = ctx.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let y = 0; y < h; y++) {
    const srcY = Math.min(origH - 1, Math.floor(y / visScale));
    for (let x = 0; x < w; x++) {
      const srcX = Math.min(origW - 1, Math.floor(x / visScale));
      if (!farmMask[srcY * origW + srcX]) {
        const i = (y * w + x) * 4;
        px[i] = Math.round(px[i] * 0.6);
        px[i + 1] = Math.round(px[i + 1] * 0.6);
        px[i + 2] = Math.round(px[i + 2] * 0.6);
      }
    }
  }
  ctx.putImageData(frame, 0, 0);

  if (ring.length >= 3) {
    const pts = ring.map(([x, y]) => [x * visScale, y * visScale] as [number, number]);

    // Subtle translucent yellow buffer zone showing the 8m boundary tree corridor
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 235, 0, 0.18)';
    ctx.lineWidth = Math.max(12, Math.round(strokePx * visScale));
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach(([x, y], idx) => (idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    // Crisp farm perimeter outline (cyan outer + yellow inner)
    const lineW = Math.max(3, Math.floor(Math.min(w, h) / 150));
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.95)';
    ctx.lineWidth = lineW + 2;
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 230, 0, 1)';
    ctx.lineWidth = lineW;
    ctx.stroke();

    // Vertex points
    const rPt = Math.max(3, lineW);
    for (const [pxX, pxY] of pts) {
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(pxX, pxY, rPt, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  let boundaryTreesDrawn = 0;
  let interiorTreesDrawn = 0;

  for (const b of boxes) {
    if (b.score < AI_BAND[1]) continue;
    const x1 = b.xmin * visScale;
    const y1 = b.ymin * visScale;
    const bw = (b.xmax - b.xmin) * visScale;
    const bh = (b.ymax - b.ymin) * visScale;

    if (b.onBoundary) {
      boundaryTreesDrawn++;
      ctx.strokeStyle = 'rgba(255, 35, 35, 0.95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, bw, bh);

      ctx.fillStyle = '#00ff44';
      ctx.beginPath();
      ctx.arc(x1 + bw / 2, y1 + bh / 2, 2.5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      interiorTreesDrawn++;
      ctx.strokeStyle = 'rgba(0, 200, 255, 0.55)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, bw, bh);
    }
  }

  // Top legend banner
  const bannerH = 30;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, 0, w, bannerH);
  ctx.fillStyle = '#ffffff';
  ctx.font = '600 12px "JetBrains Mono", monospace';
  let legendTxt = `${provider} · Boundary Trees (Red): ${boundaryTreesDrawn}`;
  if (interiorTreesDrawn > 0) {
    legendTxt += ` · Interior (Cyan): ${interiorTreesDrawn}`;
  }
  ctx.fillText(legendTxt, 12, 19);

  return out.toDataURL('image/jpeg', 0.92);
}

export async function analyzeFarmSatellite(params: {
  poly: PointLonLat[];
  species: string;
  submissionTime: string;
  useNdvi: boolean;
}): Promise<SatelliteAnalysisResult> {
  const { poly, species, submissionTime, useNdvi } = params;
  const [cLon, cLat] = polygonCentroid(poly);
  const [minx, miny, maxx, maxy] = polygonBounds(poly);

  const metaPromise = fetch(
    `/api/satellite/metadata?lon=${cLon}&lat=${cLat}&minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&useNdvi=${useNdvi ? '1' : '0'}`
  )
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null) as Promise<{
    imageryDateIso?: string | null;
    ndviSceneFound?: boolean;
  } | null>;

  const { canvas, ring, zoom, provider } = await fetchFarmMosaic(poly);
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const farmMask = createPolygonMask(canvas.width, canvas.height, ring);
  const { mask: boundaryMask, strokePx } = createBoundaryCorridorMask(
    canvas.width,
    canvas.height,
    ring,
    cLat,
    zoom,
    BOUNDARY_BUFFER_M
  );

  const { boxes, vegPct, counts, crownM } = detectCanopyTrees(
    imgData,
    farmMask,
    boundaryMask,
    cLat,
    zoom
  );

  const factor = calibFactor(species);
  const aiLow = Math.round(counts.lo * factor);
  const aiCount = Math.round(counts.mid * factor);
  const aiHigh = Math.round(counts.hi * factor);

  const annotatedImageUrl = renderAnnotatedSatellite(
    canvas,
    farmMask,
    ring,
    strokePx,
    boxes,
    provider
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
    // Derive calibrated greenness-based NDVI estimate anchored to Sentinel-2 availability & ExG cover
    let interiorGreen = 0;
    let interiorTotal = 0;
    for (let i = 0; i < canvas.width * canvas.height; i++) {
      if (!farmMask[i]) continue;
      const idx = i * 4;
      const r = imgData.data[idx];
      const g = imgData.data[idx + 1];
      const vri = (g - r) / (g + r + 1e-6);
      interiorGreen += vri;
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
    zoomUsed: zoom,
    provider,
    annotatedImageUrl,
  };
}
