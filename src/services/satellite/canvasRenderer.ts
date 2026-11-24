import { createPolygonMask } from './imageProcessing';
import type { FetchedMosaic } from './tileFetcher';
import type { TreeClassName, TreeDetectionBox } from '../../types/qa';

const CLASS_COLORS: Record<TreeClassName, string> = {
  orchard_tree: '#10b981',
  palm_tree: '#f59e0b',
  windbreak_tree: '#06b6d4',
};

export function renderCleanAndAnnotatedCanvases(
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

export function renderComparisonCanvasUrl(mosaic: FetchedMosaic, inwardBufferM: number): string {
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
