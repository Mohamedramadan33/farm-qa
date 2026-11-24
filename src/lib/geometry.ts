import { normalizeDigits, present } from './arabicNorm';
import type { PointLonLat } from '../types/qa';

export const FEDDAN_M2 = 4200.83;
export const BOUNDARY_BUFFER_M = 8.0;
export const DUP_IOU = 0.80;

const DEC_RE = /-?\d+\.\d+/g;

export function parsePoints(raw: unknown): PointLonLat[] | null {
  if (!present(raw)) return null;
  const s = normalizeDigits(String(raw));
  let pts: PointLonLat[] = [];

  for (const chunk of s.split(/[;|\n]|P/)) {
    const nums = chunk.match(DEC_RE);
    if (nums && nums.length >= 2) {
      let a = parseFloat(nums[0]);
      let b = parseFloat(nums[1]);
      if (Math.abs(a) > 90 && Math.abs(b) <= 90) {
        [a, b] = [b, a];
      }
      pts.push([b, a]); // (lon, lat)
    }
  }

  if (pts.length < 3) {
    const allNums = (s.match(DEC_RE) ?? []).map(parseFloat);
    const step = allNums.length % 4 === 0 ? 4 : 2;
    pts = [];
    for (let i = 0; i < allNums.length - 1; i += step) {
      let lat = allNums[i];
      let lon = allNums[i + 1];
      if (Math.abs(lat) > 90 && Math.abs(lon) <= 90) {
        [lat, lon] = [lon, lat];
      }
      pts.push([lon, lat]);
    }
  }

  const clean: PointLonLat[] = [];
  for (const p of pts) {
    if (p[0] >= -180 && p[0] <= 180 && p[1] >= -90 && p[1] <= 90) {
      const prev = clean[clean.length - 1];
      if (!prev || prev[0] !== p[0] || prev[1] !== p[1]) {
        clean.push(p);
      }
    }
  }

  return clean.length >= 3 ? clean : null;
}

function ccw(A: PointLonLat, B: PointLonLat, C: PointLonLat): number {
  return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
}

function segmentsProperlyIntersect(
  a1: PointLonLat,
  a2: PointLonLat,
  b1: PointLonLat,
  b2: PointLonLat
): boolean {
  const d1 = ccw(a1, a2, b1);
  const d2 = ccw(a1, a2, b2);
  const d3 = ccw(b1, b2, a1);
  const d4 = ccw(b1, b2, a2);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }
  return false;
}

export function isPolygonSelfIntersecting(ring: PointLonLat[]): boolean {
  const closed =
    ring.length > 0 &&
    (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1])
      ? [...ring, ring[0]]
      : ring;
  const n = closed.length - 1;
  if (n < 3) return true;

  for (let i = 0; i < n; i++) {
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;
      if (segmentsProperlyIntersect(closed[i], closed[i + 1], closed[j], closed[j + 1])) {
        return true;
      }
    }
  }
  return false;
}

function signedPlanarArea(ring: PointLonLat[]): number {
  let area = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  }
  return area / 2;
}

export function pointsToPolygon(pts: PointLonLat[]): PointLonLat[] | null {
  if (!pts || pts.length < 3) return null;
  const closed: PointLonLat[] =
    pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]
      ? [...pts]
      : [...pts, pts[0]];

  if (!isPolygonSelfIntersecting(closed) && Math.abs(signedPlanarArea(closed)) > 1e-13) {
    return closed;
  }

  // Untangle self-intersecting ("bowtie / X-shaped") point orders
  const seen = new Set<string>();
  const uniquePts: PointLonLat[] = [];
  for (const p of closed.slice(0, -1)) {
    const key = `${p[0].toFixed(9)},${p[1].toFixed(9)}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniquePts.push(p);
    }
  }

  if (uniquePts.length < 3) return null;

  const cx = uniquePts.reduce((acc, p) => acc + p[0], 0) / uniquePts.length;
  const cy = uniquePts.reduce((acc, p) => acc + p[1], 0) / uniquePts.length;
  const sorted = [...uniquePts].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)
  );
  const fixedRing: PointLonLat[] = [...sorted, sorted[0]];
  if (Math.abs(signedPlanarArea(fixedRing)) <= 1e-13) return null;
  return fixedRing;
}

export function parsePolygon(raw: unknown): PointLonLat[] | null {
  const pts = parsePoints(raw);
  return pts ? pointsToPolygon(pts) : null;
}

export function polygonCentroid(poly: PointLonLat[]): PointLonLat {
  const pts =
    poly.length > 1 && poly[0][0] === poly[poly.length - 1][0] && poly[0][1] === poly[poly.length - 1][1]
      ? poly.slice(0, -1)
      : poly;
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    cx += p[0];
    cy += p[1];
  }
  return [cx / Math.max(1, pts.length), cy / Math.max(1, pts.length)];
}

export function polygonBounds(poly: PointLonLat[]): [number, number, number, number] {
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  for (const [x, y] of poly) {
    if (x < minx) minx = x;
    if (y < miny) miny = y;
    if (x > maxx) maxx = x;
    if (y > maxy) maxy = y;
  }
  return [minx, miny, maxx, maxy];
}

/**
 * WGS84 Ellipsoidal UTM forward projection matching pyproj EPSG:326xx / 327xx
 */
export function projectLonLatToUtm(
  lon: number,
  lat: number,
  zone: number,
  south: boolean
): [number, number] {
  const a = 6378137.0;
  const f = 1 / 298.257223563;
  const k0 = 0.9996;
  const e2 = 2 * f - f * f;
  const ep2 = e2 / (1 - e2);

  const phi = (lat * Math.PI) / 180;
  const lam = (lon * Math.PI) / 180;
  const lam0 = (((zone - 1) * 6 - 180 + 3) * Math.PI) / 180;

  const sinPhi = Math.sin(phi);
  const cosPhi = Math.cos(phi);
  const tanPhi = Math.tan(phi);

  const N = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
  const T = tanPhi * tanPhi;
  const C = ep2 * cosPhi * cosPhi;
  const A = cosPhi * (lam - lam0);

  const M =
    a *
    ((1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 * e2 * e2) / 256) * phi -
      ((3 * e2) / 8 + (3 * e2 * e2) / 32 + (45 * e2 * e2 * e2) / 1024) * Math.sin(2 * phi) +
      ((15 * e2 * e2) / 256 + (45 * e2 * e2 * e2) / 1024) * Math.sin(4 * phi) -
      ((35 * e2 * e2 * e2) / 3072) * Math.sin(6 * phi));

  const easting =
    500000.0 +
    k0 *
      N *
      (A +
        ((1 - T + C) * Math.pow(A, 3)) / 6 +
        ((5 - 18 * T + T * T + 72 * C - 58 * ep2) * Math.pow(A, 5)) / 120);

  let northing =
    k0 *
    (M +
      N *
        tanPhi *
        ((A * A) / 2 +
          ((5 - T + 9 * C + 4 * C * C) * Math.pow(A, 4)) / 24 +
          ((61 - 58 * T + T * T + 600 * C - 330 * ep2) * Math.pow(A, 6)) / 720));

  if (south) {
    northing += 10000000.0;
  }
  return [easting, northing];
}

export function geomMetricsUtm(poly: PointLonLat[]): { areaFeddan: number; perimeterM: number } {
  const [cx, cy] = polygonCentroid(poly);
  const zone = Math.min(60, Math.max(1, Math.floor((cx + 180) / 6) + 1));
  const south = cy < 0;

  const utmRing = poly.map(([lon, lat]) => projectLonLatToUtm(lon, lat, zone, south));
  if (
    utmRing.length > 0 &&
    (utmRing[0][0] !== utmRing[utmRing.length - 1][0] ||
      utmRing[0][1] !== utmRing[utmRing.length - 1][1])
  ) {
    utmRing.push(utmRing[0]);
  }

  let area2 = 0;
  let perim = 0;
  for (let i = 0; i < utmRing.length - 1; i++) {
    const [x1, y1] = utmRing[i];
    const [x2, y2] = utmRing[i + 1];
    area2 += x1 * y2 - x2 * y1;
    perim += Math.hypot(x2 - x1, y2 - y1);
  }

  const areaM2 = Math.abs(area2) / 2;
  return {
    areaFeddan: areaM2 / FEDDAN_M2,
    perimeterM: perim,
  };
}

export function gpsAccuracy(raw: unknown): number | null {
  if (!present(raw)) return null;
  const accs: number[] = [];
  for (const chunk of normalizeDigits(String(raw)).split(/[;|\n]|P/)) {
    const nums = chunk.match(/-?\d+(?:\.\d+)?/g);
    if (!nums) continue;
    const idx = nums.findIndex((n) => n.includes('.'));
    if (idx !== -1 && nums.length > idx + 3) {
      const val = parseFloat(nums[idx + 3]);
      if (Number.isFinite(val)) accs.push(val);
    }
  }
  if (!accs.length) return null;
  return accs.reduce((a, b) => a + b, 0) / accs.length;
}

export function gpsQuality(raw: unknown, pts: PointLonLat[]): string[] {
  const warns: string[] = [];
  if (pts.length < 4) {
    warns.push('Too few points (<4)');
  }

  const ring: PointLonLat[] =
    pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]
      ? pts
      : [...pts, pts[0]];

  if (isPolygonSelfIntersecting(ring)) {
    warns.push('Points were recorded in crossed order (Auto-corrected to full boundary)');
  } else {
    const areaDeg = Math.abs(signedPlanarArea(ring));
    let lenDeg = 0;
    for (let i = 0; i < ring.length - 1; i++) {
      lenDeg += Math.hypot(ring[i + 1][0] - ring[i][0], ring[i + 1][1] - ring[i][1]);
    }
    if (areaDeg > 0 && lenDeg > 0) {
      const compactness = (4 * Math.PI * areaDeg) / (lenDeg * lenDeg);
      if (compactness < 0.15) {
        warns.push(
          `Extremely thin shape (Compactness ${compactness.toFixed(2)}) - Likely a road/path, not a farm`
        );
      } else if (compactness < 0.30) {
        warns.push(`Irregular shape (Compactness ${compactness.toFixed(2)})`);
      }
    }
  }

  const acc = gpsAccuracy(raw);
  if (acc !== null && acc > 15) {
    warns.push(`Poor GPS accuracy during capture (Avg ${Math.round(acc)}m)`);
  }
  return warns;
}

export function pointInRingPx(x: number, y: number, ring: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointInRingXY(x: number, y: number, ring: PointLonLat[]): boolean {
  return pointInRingPx(x, y, ring);
}

export function polygonIoU(polyA: PointLonLat[], polyB: PointLonLat[]): number {
  const [minxA, minyA, maxxA, maxyA] = polygonBounds(polyA);
  const [minxB, minyB, maxxB, maxyB] = polygonBounds(polyB);
  if (maxxA < minxB || maxxB < minxA || maxyA < minyB || maxyB < minyA) {
    return 0;
  }

  const minx = Math.min(minxA, minxB);
  const miny = Math.min(minyA, minyB);
  const maxx = Math.max(maxxA, maxxB);
  const maxy = Math.max(maxyA, maxyB);
  const steps = 48;
  const dx = (maxx - minx) / steps;
  const dy = (maxy - miny) / steps;
  if (dx <= 0 || dy <= 0) return 0;

  let inter = 0;
  let union = 0;
  for (let ix = 0; ix < steps; ix++) {
    const x = minx + (ix + 0.5) * dx;
    for (let iy = 0; iy < steps; iy++) {
      const y = miny + (iy + 0.5) * dy;
      const inA = pointInRingXY(x, y, polyA);
      const inB = pointInRingXY(x, y, polyB);
      if (inA && inB) inter++;
      if (inA || inB) union++;
    }
  }
  return union > 0 ? inter / union : 0;
}

/**
 * Translates a lon/lat polygon in meters (+offsetX_m East, +offsetY_m North)
 * to calibrate imagery-to-GPS alignment shifts (e.g. 2-6m offset on Google Satellite).
 */
export function shiftPolygonMeters(
  poly: PointLonLat[],
  offsetX_m: number,
  offsetY_m: number
): PointLonLat[] {
  if (!poly.length || (Math.abs(offsetX_m) < 1e-6 && Math.abs(offsetY_m) < 1e-6)) {
    return poly;
  }
  const [, cLat] = polygonCentroid(poly);
  const dLat = offsetY_m / 111139.0;
  const dLon = offsetX_m / (111139.0 * Math.max(0.2, Math.cos((cLat * Math.PI) / 180)));
  return poly.map(([lon, lat]) => [lon + dLon, lat + dLat]);
}

/**
 * Buffers a closed polygon inward (bufferM < 0, e.g. -1.5m) or outward (bufferM > 0)
 * using mitered edge-normal offsets in local meters. Clamps miter spikes cleanly.
 */
export function bufferPolygonMeters(poly: PointLonLat[], bufferM: number): PointLonLat[] {
  if (!poly || poly.length < 3 || Math.abs(bufferM) < 0.05) {
    return poly;
  }

  const open =
    poly[0][0] === poly[poly.length - 1][0] && poly[0][1] === poly[poly.length - 1][1]
      ? poly.slice(0, -1)
      : [...poly];
  if (open.length < 3) return poly;

  const [cLon, cLat] = polygonCentroid(open);
  const mPerDegLat = 111139.0;
  const mPerDegLon = 111139.0 * Math.max(0.2, Math.cos((cLat * Math.PI) / 180));

  const ptsM: Array<[number, number]> = open.map(([lon, lat]) => [
    (lon - cLon) * mPerDegLon,
    (lat - cLat) * mPerDegLat,
  ]);

  // Signed area in m^2: >0 => CCW, <0 => CW
  let area2 = 0;
  for (let i = 0; i < ptsM.length; i++) {
    const [x1, y1] = ptsM[i];
    const [x2, y2] = ptsM[(i + 1) % ptsM.length];
    area2 += x1 * y2 - x2 * y1;
  }
  if (Math.abs(area2) < 1.0) return poly;
  const ccwSign = area2 > 0 ? 1 : -1;

  // For CCW polygon, outward right-hand normal is (dy, -dx)/len; inward normal is (-dy, dx)/len
  // So moving by `bufferM` (>0 outward, <0 inward) multiplies inward normal by (-bufferM * ccwSign)
  const n = ptsM.length;
  const offsetEdges: Array<{ p: [number, number]; d: [number, number] }> = [];
  for (let i = 0; i < n; i++) {
    const [x1, y1] = ptsM[i];
    const [x2, y2] = ptsM[(i + 1) % n];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) continue;
    // Outward unit normal for CCW is (dy/len, -dx/len)
    const nx = (dy / len) * ccwSign;
    const ny = (-dx / len) * ccwSign;
    offsetEdges.push({
      p: [x1 + nx * bufferM, y1 + ny * bufferM],
      d: [dx / len, dy / len],
    });
  }
  if (offsetEdges.length < 3) return poly;

  const outM: Array<[number, number]> = [];
  const m = offsetEdges.length;
  const maxMiter = Math.abs(bufferM) * 3.0;

  for (let i = 0; i < m; i++) {
    const e1 = offsetEdges[(i - 1 + m) % m];
    const e2 = offsetEdges[i];
    const denom = e1.d[0] * e2.d[1] - e1.d[1] * e2.d[0];
    const origV = ptsM[i % ptsM.length];
    if (Math.abs(denom) < 1e-3) {
      outM.push(e2.p);
      continue;
    }
    const rx = e2.p[0] - e1.p[0];
    const ry = e2.p[1] - e1.p[1];
    const t = (rx * e2.d[1] - ry * e2.d[0]) / denom;
    const ix = e1.p[0] + t * e1.d[0];
    const iy = e1.p[1] + t * e1.d[1];
    const dist = Math.hypot(ix - origV[0], iy - origV[1]);
    if (dist > maxMiter && dist > 1e-4) {
      const clampScale = maxMiter / dist;
      outM.push([
        origV[0] + (ix - origV[0]) * clampScale,
        origV[1] + (iy - origV[1]) * clampScale,
      ]);
    } else {
      outM.push([ix, iy]);
    }
  }

  // Ensure shrunk polygon did not invert
  let newArea2 = 0;
  for (let i = 0; i < outM.length; i++) {
    const [x1, y1] = outM[i];
    const [x2, y2] = outM[(i + 1) % outM.length];
    newArea2 += x1 * y2 - x2 * y1;
  }
  if (newArea2 * area2 <= 0) {
    return poly;
  }

  const outPoly: PointLonLat[] = outM.map(([mx, my]) => [
    cLon + mx / mPerDegLon,
    cLat + my / mPerDegLat,
  ]);
  outPoly.push(outPoly[0]);
  return outPoly;
}
