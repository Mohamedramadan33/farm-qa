import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import express from 'express';
import rateLimit from 'express-rate-limit';
import { SEED_LABELS, SEED_SUBMISSIONS } from './src/server/seedSubmissions.ts';
import { divideWorkFairly, type UnvalidatedFarmItem } from './src/lib/workDivider.ts';
import {
  TEAM_KEYS,
  type AssignmentRecord,
  type FieldDecisionRow,
  type LabelMap,
  type MasterReviewRecord,
  type RawKoboSubmission,
  type TeamMemberKey,
  type TreeClassName,
  type TreeHealthStatus,
} from './src/types/qa.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execFileAsync = promisify(execFile);

const PORT = Number(process.env.PORT || 3000);
const KOBO_SERVER = (process.env.KOBO_SERVER || 'https://kf.kobotoolbox.org').replace(/\/+$/, '');
const KOBO_ASSET_UID = process.env.KOBO_ASSET_UID || 'aX4NJWgge6tooXjfSYXhrq';
const ALLOWED_MEDIA_HOST = new URL(KOBO_SERVER).hostname;
const MAX_MEDIA_BYTES = 15 * 1024 * 1024;
const SESSION_SECRET =
  process.env.SESSION_SECRET?.trim() || crypto.randomBytes(32).toString('hex');
const SESSION_TTL_SEC = 12 * 3600;

interface ServerDetectionConfig {
  version: string;
  classes: TreeClassName[];
  aiModelConfThresh: number;
  confIntervalDelta: number;
  sahiTileSize: number;
  sahiOverlap: number;
  highResTileSize?: number;
  highResOverlap?: number;
  highAccuracyTileSize?: number;
  highAccuracyOverlap?: number;
  nmsIouThresh: number;
  wbfIouThresh?: number;
  classAwareNms: boolean;
  crossClassSuppressionIou: number;
  minCrownDiamM: number;
  maxCrownDiamM: number;
}

const DEFAULT_DETECTION_CONFIG: ServerDetectionConfig = {
  version: '1.3.0',
  classes: ['orchard_tree', 'palm_tree', 'windbreak_tree'],
  aiModelConfThresh: 0.55,
  confIntervalDelta: 0.1,
  sahiTileSize: 512,
  sahiOverlap: 0.3,
  highResTileSize: 288,
  highResOverlap: 0.4,
  highAccuracyTileSize: 288,
  highAccuracyOverlap: 0.4,
  nmsIouThresh: 0.4,
  wbfIouThresh: 0.42,
  classAwareNms: true,
  crossClassSuppressionIou: 0.85,
  minCrownDiamM: 1.8,
  maxCrownDiamM: 7.0,
};

function loadServerDetectionConfig(): ServerDetectionConfig {
  const cfgPath = path.resolve(__dirname, 'config', 'detection.config.json');
  try {
    if (fs.existsSync(cfgPath)) {
      const parsed = JSON.parse(fs.readFileSync(cfgPath, 'utf-8')) as Partial<ServerDetectionConfig>;
      return { ...DEFAULT_DETECTION_CONFIG, ...parsed };
    }
  } catch {
    // Fallback to defaults if config JSON cannot be read
  }
  return DEFAULT_DETECTION_CONFIG;
}

const detectionConfig = loadServerDetectionConfig();
const TREE_CLASSES = ['orchard_tree', 'palm_tree', 'windbreak_tree'] as const;
const FEEDBACK_ROOT = path.resolve(__dirname, 'dataset', 'feedback_queue');

function isAllowedMediaUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    return u.protocol === 'https:' && u.hostname === ALLOWED_MEDIA_HOST;
  } catch {
    return false;
  }
}

function resolveConfinedModel(relCandidates: string[]): string | null {
  for (const rel of relCandidates) {
    const resolved = path.resolve(__dirname, rel);
    if (!resolved.startsWith(__dirname + path.sep)) continue;
    if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
      return resolved;
    }
  }
  return null;
}

let cachedPyYoloAvail: boolean | null = null;
async function checkPythonYoloAvailable(): Promise<boolean> {
  if (cachedPyYoloAvail !== null) return cachedPyYoloAvail;
  try {
    await execFileAsync('python3', ['-c', 'import ultralytics'], { timeout: 4000 });
    cachedPyYoloAvail = true;
  } catch {
    cachedPyYoloAvail = false;
  }
  return cachedPyYoloAvail;
}

function signDetectionToken(userKey: TeamMemberKey): string {
  const nowSec = Math.floor(Date.now() / 1000);
  const payload = Buffer.from(
    JSON.stringify({ sub: userKey, iat: nowSec, exp: nowSec + SESSION_TTL_SEC }),
    'utf-8'
  ).toString('base64url');
  const sig = crypto.createHmac('sha256', SESSION_SECRET).update(payload).digest('base64url');
  return `${payload}.${sig}`;
}

function verifyDetectionToken(token: string): boolean {
  const parts = token.split('.');
  if (parts.length !== 2) return false;
  const [payloadB64, sigB64] = parts;
  if (!payloadB64 || !sigB64) return false;

  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payloadB64)
    .digest('base64url');
  const a = Buffer.from(sigB64);
  const b = Buffer.from(expectedSig);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return false;

  try {
    const decoded = JSON.parse(Buffer.from(payloadB64, 'base64url').toString('utf-8')) as {
      sub?: string;
      exp?: number;
    };
    if (!decoded.sub || !TEAM_KEYS.includes(decoded.sub as TeamMemberKey)) return false;
    if (!decoded.exp || decoded.exp < Math.floor(Date.now() / 1000)) return false;
    return true;
  } catch {
    return false;
  }
}

function requireDetectionAuth(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
): void {
  const authHeader = String(req.headers.authorization || '').trim();
  if (authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice(7).trim();
    if (token && verifyDetectionToken(token)) {
      next();
      return;
    }
  }

  const configuredKey = process.env.DETECTION_API_KEY?.trim();
  const apiKeyHeader = String(req.headers['x-api-key'] || '').trim();
  if (configuredKey && apiKeyHeader) {
    const a = Buffer.from(apiKeyHeader);
    const b = Buffer.from(configuredKey);
    if (a.length === b.length && crypto.timingSafeEqual(a, b)) {
      next();
      return;
    }
  }

  res.status(401).json({
    ok: false,
    error: 'Authentication required (valid Bearer session token or X-API-Key).',
  });
}

function pointInRing(x: number, y: number, ring: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function worldPxToLonLatServer(px: number, py: number, zoom: number): [number, number] {
  const worldSize = 256 * Math.pow(2, zoom);
  const lon = (px / worldSize) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * py) / worldSize;
  const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  return [lon, lat];
}

interface ServerDetectionBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  label: TreeClassName;
  confidence: number;
  health: TreeHealthStatus;
  lat?: number;
  lon?: number;
}

function serverBoxIoU(a: ServerDetectionBox, b: ServerDetectionBox): number {
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

/**
 * Class-Aware Consensus Weighted Box Fusion (WBF) for overlapping SAHI tiles.
 * Quadratic confidence-weighting tightens crown centers across tile seams while
 * rewarding detections corroborated across 2+ overlapping SAHI windows.
 */
function fuseServerDetectionsWBF(
  boxes: ServerDetectionBox[],
  iouThresh: number,
  minCenterDistPx: number,
  crossClassIou: number
): ServerDetectionBox[] {
  const ordered = [...boxes].sort((a, b) => b.confidence - a.confidence);
  interface Cluster {
    wSum: number;
    maxConf: number;
    votes: number;
    wxmin: number;
    wymin: number;
    wxmax: number;
    wymax: number;
    label: TreeClassName;
    health: TreeHealthStatus;
    rep: ServerDetectionBox;
  }

  const clusters: Cluster[] = [];
  for (const b of ordered) {
    const cx = (b.xmin + b.xmax) / 2;
    const cy = (b.ymin + b.ymax) / 2;
    let target: Cluster | null = null;

    for (const cl of clusters) {
      const iou = serverBoxIoU(b, cl.rep);
      const pcx = (cl.rep.xmin + cl.rep.xmax) / 2;
      const pcy = (cl.rep.ymin + cl.rep.ymax) / 2;
      const dist = Math.hypot(cx - pcx, cy - pcy);

      if (b.label === cl.label) {
        if (iou >= iouThresh || dist < minCenterDistPx) {
          target = cl;
          break;
        }
      } else if (iou >= crossClassIou) {
        target = cl;
        break;
      }
    }

    const w = Math.max(0.04, b.confidence * b.confidence);
    if (!target) {
      clusters.push({
        wSum: w,
        maxConf: b.confidence,
        votes: 1,
        wxmin: b.xmin * w,
        wymin: b.ymin * w,
        wxmax: b.xmax * w,
        wymax: b.ymax * w,
        label: b.label,
        health: b.health,
        rep: { ...b },
      });
    } else {
      if (b.confidence > target.maxConf) {
        target.label = b.label;
        target.health = b.health;
      }
      target.votes += 1;
      target.wSum += w;
      target.maxConf = Math.max(target.maxConf, b.confidence);
      target.wxmin += b.xmin * w;
      target.wymin += b.ymin * w;
      target.wxmax += b.xmax * w;
      target.wymax += b.ymax * w;
      target.rep = {
        xmin: target.wxmin / target.wSum,
        ymin: target.wymin / target.wSum,
        xmax: target.wxmax / target.wSum,
        ymax: target.wymax / target.wSum,
        label: target.label,
        confidence: target.maxConf,
        health: target.health,
      };
    }
  }

  return clusters.slice(0, 1200).map((cl) => {
    const consensusBoost = Math.min(0.08, Math.max(0, cl.votes - 1) * 0.04);
    const calibratedConf = Math.min(0.98, cl.maxConf + consensusBoost);
    return {
      xmin: Math.round(cl.rep.xmin),
      ymin: Math.round(cl.rep.ymin),
      xmax: Math.round(cl.rep.xmax),
      ymax: Math.round(cl.rep.ymax),
      label: cl.label,
      confidence: Number(calibratedConf.toFixed(3)),
      health: cl.health,
    };
  });
}

const TREE_VISION_SYSTEM_INSTRUCTION = [
  'You are an ultra-precise geospatial computer vision detector trained on Egyptian Nile Delta and desert-reclamation orchards (Zoom 19-20 RGB satellite imagery).',
  'Your sole objective is exact per-tree localization with near-zero false positives on crops, soil, buildings, or water.',
  'Always emit separate tight bounding boxes centered on sunlit leafy crowns ONLY (never include long cast shadow tails inside box_2d).',
  'Return strictly valid JSON matching the schema.',
].join(' ');

function buildTreeVisionPrompt(tileW: number, tileH: number, mpp: number, isHighRes = false): string {
  const pxPerM = 1 / Math.max(mpp, 0.08);
  const youngPx = Math.max(6, Math.round((isHighRes ? 1.2 : 1.6) * pxPerM));
  const maturePx = Math.round(6.0 * pxPerM);
  return [
    `Tile size: ${tileW}x${tileH} px at ~${mpp.toFixed(3)} m/px (expect ~${youngPx}-${maturePx} px crown diameter for ${isHighRes ? '1.2m-6.0m' : '1.6m-6.0m'} canopies).`,
    ...(isHighRes
      ? [
          `HIGH-RES SUB-TILE MODE (${tileW}x${tileH} tight crop): Look carefully for small/young saplings (>=1.2m crown) along drip lines, split tightly clustered crowns into individual boxes, and ignore any flat dark #0f172a edge padding borders.`,
        ]
      : []),
    '',
    'INCLUDE (Individual Trees & Palms ONLY):',
    '1. "orchard_tree": Distinct circular/lobed foliage crowns (citrus, mango, olive, guava, pomegranate, young saplings along drip-irrigation lines), typically showing canopy leaf texture and an adjacent directional cast shadow.',
    '2. "palm_tree": Star-shaped / spiky rosette date palm crowns and tall palm shadows.',
    '3. "windbreak_tree": Individual tall linear trees along farm boundaries, fences, or canal edges (emit one box per distinct crown peak along hedgerows rather than one giant box).',
    '4. Touching canopies: If two orchard crowns touch slightly, split them into TWO separate tight boxes centered on each canopy peak.',
    '5. Slightly dimmed outer margins provide context only—detect trees visible in the bright interior parcel and along its immediate border.',
    '',
    'STRICTLY EXCLUDE (NEVER DETECT):',
    '- Flat uniform green field crops (berseem/clover, alfalfa, wheat, vegetable beds), grass strips, or plowed furrow stripes.',
    '- Buildings, concrete/metal roofs, greenhouses, shade nets, water basins, canal water, roads, cars, or standalone dark shadows without a crown.',
    '- Return detections: [] if no real trees or palms are present.',
    '',
    'Output normalized coordinates [ymin, xmin, ymax, xmax] in 0..1000.',
  ].join('\n');
}

function nowStr(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// In-memory store backing Kobo validation statuses, reviews, assignments, and tile LRU cache
const localValidationStatuses = new Map<string, { uid: string; label: string }>();
const masterReviews = new Map<string, MasterReviewRecord>();
const fieldReviews = new Map<string, FieldDecisionRow[]>();
const assignments = new Map<string, AssignmentRecord>();

SEED_SUBMISSIONS.forEach((sub, idx) => {
  const subId = String(sub._id);
  const member = TEAM_KEYS[idx % TEAM_KEYS.length];
  const atHouse = sub.at_house as Record<string, string> | undefined;
  assignments.set(subId, {
    to: member,
    by: 'auto',
    at: nowStr(),
    farmId: atHouse?.farm_id || subId,
  });
});

let cachedKobo: {
  ts: number;
  rows: RawKoboSubmission[];
  labels: LabelMap;
  live: boolean;
} | null = null;

const tileCache = new Map<string, { buf: Buffer; contentType: string }>();
const imageryDateCache = new Map<string, string | null>();

async function fetchLiveKobo(): Promise<{ rows: RawKoboSubmission[]; labels: LabelMap }> {
  const token = process.env.KOBO_TOKEN?.trim();
  if (!token) {
    throw new Error('KOBO_TOKEN_NOT_SET');
  }

  const headers = { Authorization: `Token ${token}` };
  let url: string | null = `${KOBO_SERVER}/api/v2/assets/${KOBO_ASSET_UID}/data.json?limit=3000`;
  const rows: RawKoboSubmission[] = [];

  while (url) {
    const resp = await fetch(url, { headers, signal: AbortSignal.timeout(45000) });
    if (resp.status === 401) {
      throw new Error('Invalid or expired Kobo Token (401)');
    }
    if (!resp.ok) {
      throw new Error(`Kobo API HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as { results?: RawKoboSubmission[]; next?: string | null };
    if (Array.isArray(data.results)) {
      rows.push(...data.results);
    }
    url = data.next || null;
  }

  const labels: LabelMap = { ...SEED_LABELS };
  try {
    const assetResp = await fetch(`${KOBO_SERVER}/api/v2/assets/${KOBO_ASSET_UID}.json`, {
      headers,
      signal: AbortSignal.timeout(25000),
    });
    if (assetResp.ok) {
      const assetJson = (await assetResp.json()) as {
        content?: {
          survey?: Array<{
            name?: string;
            $autoname?: string;
            $xpath?: string;
            label?: string | string[];
          }>;
        };
      };
      for (const item of assetJson.content?.survey || []) {
        const name = item.name || item.$autoname;
        const lbl = item.label;
        if (!name || !lbl) continue;
        let entry: [string, string] | null = null;
        if (Array.isArray(lbl)) {
          const parts = lbl.filter(Boolean).map(String);
          if (parts.length > 0) {
            entry = [parts[0], parts.join(' / ')];
          }
        } else {
          entry = [String(lbl), String(lbl)];
        }
        if (!entry) continue;
        const lowerName = name.toLowerCase();
        if (!labels[lowerName]) {
          labels[lowerName] = entry;
        }
        if (item.$xpath) {
          labels[String(item.$xpath).toLowerCase()] = entry;
        }
      }
    }
  } catch {
    // Keep SEED_LABELS fallback if asset metadata request fails
  }

  return { rows, labels };
}

async function getSubmissionsBundle(forceRefresh = false) {
  const token = process.env.KOBO_TOKEN?.trim();
  if (token) {
    if (!forceRefresh && cachedKobo && Date.now() - cachedKobo.ts < 600_000) {
      return cachedKobo;
    }
    const { rows, labels } = await fetchLiveKobo();
    cachedKobo = { ts: Date.now(), rows, labels, live: true };
    return cachedKobo;
  }

  const seeded = SEED_SUBMISSIONS.map((s) => {
    const sid = String(s._id);
    const localStatus = localValidationStatuses.get(sid);
    return localStatus ? { ...s, _validation_status: localStatus } : s;
  });
  return {
    ts: Date.now(),
    rows: seeded,
    labels: SEED_LABELS,
    live: false,
  };
}

async function startServer() {
  const app = express();
  app.set('trust proxy', 1);
  app.use(express.json({ limit: '25mb' }));

  const detectionLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 90,
    standardHeaders: true,
    legacyHeaders: false,
    message: { ok: false, error: 'Too many detection requests; please wait a moment.' },
  });

  const feedbackLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 20,
    standardHeaders: true,
    legacyHeaders: false,
    message: { ok: false, error: 'Too many dataset feedback uploads; please wait a moment.' },
  });

  // 0. Session Token Issuance for Authenticated Auditor Detection Calls
  app.post('/api/auth/session', (req, res) => {
    const { userKey } = (req.body || {}) as { userKey?: unknown };
    if (typeof userKey !== 'string' || !TEAM_KEYS.includes(userKey as TeamMemberKey)) {
      res.status(400).json({ ok: false, error: 'Valid team member userKey is required.' });
      return;
    }
    const token = signDetectionToken(userKey as TeamMemberKey);
    res.json({ ok: true, token, expiresInSec: SESSION_TTL_SEC });
  });

  // 1. Kobo Submissions & Survey Labels
  app.get('/api/kobo/submissions', async (req, res) => {
    try {
      const refresh = req.query.refresh === '1';
      const bundle = await getSubmissionsBundle(refresh);

      let nextIdx = assignments.size;
      for (const r of bundle.rows) {
        const subId = String(r._id ?? '');
        if (subId && !assignments.has(subId)) {
          const member = TEAM_KEYS[nextIdx % TEAM_KEYS.length];
          nextIdx++;
          assignments.set(subId, {
            to: member,
            by: 'auto',
            at: nowStr(),
          });
        }
      }

      res.json({
        rows: bundle.rows,
        labels: bundle.labels,
        liveKobo: bundle.live,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed fetching Kobo submissions';
      res.status(500).json({ error: msg });
    }
  });

  // 2. Kobo Media Proxy (Images & Videos)
  app.get('/api/kobo/media', async (req, res) => {
    const rawUrl = String(req.query.url || '').trim();
    if (!rawUrl || !isAllowedMediaUrl(rawUrl)) {
      res.status(400).json({ error: 'Media URL must point to the configured Kobo server' });
      return;
    }

    try {
      const token = process.env.KOBO_TOKEN?.trim();
      const headers: Record<string, string> = {};
      if (token) {
        headers.Authorization = `Token ${token}`;
      }
      const upstream = await fetch(rawUrl, {
        headers,
        signal: AbortSignal.timeout(45000),
      });
      if (!upstream.ok) {
        res.status(upstream.status).json({ error: `Media fetch failed (${upstream.status})` });
        return;
      }
      const len = Number(upstream.headers.get('content-length') || 0);
      if (len > 0 && len > MAX_MEDIA_BYTES) {
        res.status(413).json({ error: 'Attachment exceeds 15MB limit' });
        return;
      }
      const contentType = upstream.headers.get('content-type') || 'application/octet-stream';
      const arrBuf = await upstream.arrayBuffer();
      if (arrBuf.byteLength > MAX_MEDIA_BYTES) {
        res.status(413).json({ error: 'Attachment exceeds 15MB limit' });
        return;
      }
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'public, max-age=3600');
      res.send(Buffer.from(arrBuf));
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Media download error';
      res.status(502).json({ error: msg });
    }
  });

  // 3. Kobo Validation Status Update
  app.patch('/api/kobo/validation-status', async (req, res) => {
    const { submissionId, statusCode } = req.body as {
      submissionId?: string;
      statusCode?: string;
    };
    if (!submissionId || !statusCode) {
      res.status(400).json({ ok: false, error: 'Missing submissionId or statusCode' });
      return;
    }

    const labelMap: Record<string, string> = {
      validation_status_approved: 'Approved',
      validation_status_not_approved: 'Not Approved',
      validation_status_on_hold: 'On Hold',
    };
    localValidationStatuses.set(String(submissionId), {
      uid: statusCode,
      label: labelMap[statusCode] || statusCode,
    });

    const token = process.env.KOBO_TOKEN?.trim();
    if (!token) {
      res.json({ ok: true, syncedMode: 'session' });
      return;
    }

    try {
      const numId = Number(submissionId);
      const upstream = await fetch(
        `${KOBO_SERVER}/api/v2/assets/${KOBO_ASSET_UID}/data/validation_statuses/`,
        {
          method: 'PATCH',
          headers: {
            Authorization: `Token ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            payload: {
              submission_ids: [Number.isFinite(numId) ? numId : submissionId],
              'validation_status.uid': statusCode,
            },
          }),
          signal: AbortSignal.timeout(20000),
        }
      );
      if (upstream.ok && cachedKobo) {
        for (const r of cachedKobo.rows) {
          if (String(r._id) === String(submissionId)) {
            r._validation_status = {
              uid: statusCode,
              label: labelMap[statusCode] || statusCode,
            };
          }
        }
      }
      res.json({ ok: upstream.ok, syncedMode: 'kobo' });
    } catch {
      res.json({ ok: false, syncedMode: 'kobo' });
    }
  });

  // 4. CORS-Safe Satellite Tile Proxy (Google Earth & ESRI World Imagery with auto-fallback)
  async function fetchUpstreamTile(
    prov: 'google' | 'esri',
    zn: number,
    xn: number,
    yn: number
  ): Promise<{ buf: Buffer; contentType: string; isLikelyBlank: boolean } | null> {
    const shard = Math.abs(xn + yn) % 4;
    const tileUrl =
      prov === 'google'
        ? `https://mt${shard}.google.com/vt/lyrs=s&hl=en&x=${xn}&y=${yn}&z=${zn}`
        : `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zn}/${yn}/${xn}`;

    try {
      const upstream = await fetch(tileUrl, {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
          Accept: 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        },
        signal: AbortSignal.timeout(9000),
      });
      if (!upstream.ok) return null;
      const arrBuf = await upstream.arrayBuffer();
      const buf = Buffer.from(arrBuf);
      if (buf.length < 400) return null;
      const contentType = upstream.headers.get('content-type') || 'image/jpeg';
      if (!contentType.startsWith('image/')) return null;
      const isLikelyBlank = prov === 'esri' && buf.length <= 2750;
      return { buf, contentType, isLikelyBlank };
    } catch {
      return null;
    }
  }

  app.get('/api/satellite/tile/:provider/:z/:x/:y', async (req, res) => {
    const { provider, z, x, y } = req.params;
    const zn = Number(z);
    const xn = Number(x);
    const yn = Number(y);
    if (!Number.isInteger(zn) || !Number.isInteger(xn) || !Number.isInteger(yn) || zn < 0 || zn > 22) {
      res.status(400).end();
      return;
    }

    const allowFallback = req.query.fallback !== '0';
    const primary: 'google' | 'esri' = provider === 'esri' ? 'esri' : 'google';
    const secondary: 'google' | 'esri' = primary === 'google' ? 'esri' : 'google';
    const cacheKey = `${primary}:${zn}:${xn}:${yn}`;

    const hit = tileCache.get(cacheKey);
    if (hit) {
      res.setHeader('Content-Type', hit.contentType);
      res.setHeader('Cache-Control', 'public, max-age=86400');
      res.send(hit.buf);
      return;
    }

    let result = await fetchUpstreamTile(primary, zn, xn, yn);
    if ((!result || result.isLikelyBlank) && allowFallback) {
      const fb = await fetchUpstreamTile(secondary, zn, xn, yn);
      if (fb && (!result || !fb.isLikelyBlank)) {
        result = fb;
      }
    }

    if (!result) {
      res.status(404).end();
      return;
    }

    if (tileCache.size > 1500) {
      const oldest = tileCache.keys().next().value;
      if (oldest) tileCache.delete(oldest);
    }
    tileCache.set(cacheKey, { buf: result.buf, contentType: result.contentType });
    res.setHeader('Content-Type', result.contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.send(result.buf);
  });

  // 5. Satellite Capture Date (ArcGIS SRC_DATE2) & Recent Sentinel-2 STAC Metadata
  app.get('/api/satellite/metadata', async (req, res) => {
    const lon = Number(req.query.lon);
    const lat = Number(req.query.lat);
    const useNdvi = req.query.useNdvi === '1' || req.query.useNdvi === 'true';
    const minx = Number(req.query.minx ?? lon - 0.001);
    const miny = Number(req.query.miny ?? lat - 0.001);
    const maxx = Number(req.query.maxx ?? lon + 0.001);
    const maxy = Number(req.query.maxy ?? lat + 0.001);

    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      res.status(400).json({ imageryDate: null, pseudoNdvi: null });
      return;
    }

    const key = `${lon.toFixed(3)},${lat.toFixed(3)}`;
    let imageryDateIso: string | null = imageryDateCache.get(key) ?? null;

    if (!imageryDateCache.has(key)) {
      try {
        const qs = new URLSearchParams({
          geometry: `${lon},${lat}`,
          geometryType: 'esriGeometryPoint',
          inSR: '4326',
          spatialRel: 'esriSpatialRelIntersects',
          outFields: 'SRC_DATE2',
          returnGeometry: 'false',
          f: 'json',
        });
        const arcResp = await fetch(
          `https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/0/query?${qs.toString()}`,
          { signal: AbortSignal.timeout(12000) }
        );
        if (arcResp.ok) {
          const arcJson = (await arcResp.json()) as {
            features?: Array<{ attributes?: { SRC_DATE2?: number } }>;
          };
          const timestamps = (arcJson.features || [])
            .map((f) => f.attributes?.SRC_DATE2)
            .filter((v): v is number => typeof v === 'number' && v > 0);
          if (timestamps.length > 0) {
            const maxTs = Math.max(...timestamps);
            imageryDateIso = new Date(maxTs).toISOString();
          }
        }
      } catch {
        imageryDateIso = null;
      }
      imageryDateCache.set(key, imageryDateIso);
    }

    let ndviSceneFound = false;
    let ndviSceneCloud: number | null = null;
    if (useNdvi) {
      try {
        const end = new Date();
        const start = new Date(end.getTime() - 120 * 24 * 3600 * 1000);
        const stacResp = await fetch('https://planetarycomputer.microsoft.com/api/stac/v1/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collections: ['sentinel-2-l2a'],
            bbox: [minx, miny, maxx, maxy],
            datetime: `${start.toISOString().slice(0, 10)}/${end.toISOString().slice(0, 10)}`,
            limit: 5,
            query: { 'eo:cloud_cover': { lt: 20 } },
          }),
          signal: AbortSignal.timeout(12000),
        });
        if (stacResp.ok) {
          const stacJson = (await stacResp.json()) as {
            features?: Array<{ properties?: { 'eo:cloud_cover'?: number } }>;
          };
          if (stacJson.features && stacJson.features.length > 0) {
            ndviSceneFound = true;
            ndviSceneCloud = stacJson.features[0].properties?.['eo:cloud_cover'] ?? null;
          }
        }
      } catch {
        ndviSceneFound = false;
      }
    }

    res.json({
      imageryDateIso,
      ndviSceneFound,
      ndviSceneCloud,
    });
  });

  // 5b. Satellite AI Model Status & ONNX Serving
  app.get('/api/satellite/model-status', async (_req, res) => {
    const geminiReady = Boolean(process.env.GEMINI_API_KEY?.trim());
    const onnxPath = resolveConfinedModel(['public/models/best.onnx', 'models/best.onnx']);
    const ptPath = resolveConfinedModel(['models/best.pt', 'best.pt']);
    const remoteUrl = (process.env.YOLO_INFER_URL || '').trim();
    const pyReady = ptPath ? await checkPythonYoloAvailable() : false;

    res.json({
      ok: true,
      geminiReady,
      onnxReady: Boolean(onnxPath),
      onnxUrl: onnxPath ? '/models/best.onnx' : null,
      ptReady: Boolean(ptPath),
      backendYoloReady: Boolean(remoteUrl || pyReady),
      classes: TREE_CLASSES,
      config: detectionConfig,
    });
  });

  app.get('/models/best.onnx', (_req, res) => {
    const onnxPath = resolveConfinedModel(['public/models/best.onnx', 'models/best.onnx']);
    if (!onnxPath) {
      res.status(404).json({ error: 'ONNX model weights not exported yet' });
      return;
    }
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Cache-Control', 'public, max-age=3600');
    res.sendFile(onnxPath);
  });

  // 5c. Primary Tier-1 AI Spatial Tree Crown Detector (Gemini Vision + WBF + Geometry Filtering)
  app.post('/api/detect-trees', detectionLimiter, requireDetectionAuth, async (req, res) => {
    const geminiKey = process.env.GEMINI_API_KEY?.trim();
    if (!geminiKey) {
      res.status(503).json({
        ok: false,
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
      return;
    }

    const {
      tiles,
      polygonRingPx,
      geoTransform,
      highRes = false,
      highAccuracy = false,
      confThresh,
      minConf,
    } = (req.body || {}) as {
      tiles?: Array<{
        imageBase64?: unknown;
        offsetX?: unknown;
        offsetY?: unknown;
        width?: unknown;
        height?: unknown;
      }>;
      polygonRingPx?: unknown;
      geoTransform?: {
        worldOriginX?: unknown;
        worldOriginY?: unknown;
        zoom?: unknown;
        mpp?: unknown;
      };
      highRes?: unknown;
      highAccuracy?: unknown;
      confThresh?: unknown;
      minConf?: unknown;
    };
    const isHighRes = Boolean(highRes || highAccuracy);

    if (!Array.isArray(tiles) || tiles.length === 0 || tiles.length > 48) {
      res.status(400).json({ ok: false, error: 'tiles array (1..48 tiles) is required.' });
      return;
    }

    const validTiles: Array<{
      b64: string;
      mimeType: string;
      offsetX: number;
      offsetY: number;
      width: number;
      height: number;
    }> = [];

    for (const t of tiles) {
      if (typeof t?.imageBase64 !== 'string' || t.imageBase64.length < 32 || t.imageBase64.length > 8_000_000) {
        res.status(400).json({ ok: false, error: 'Invalid tile imageBase64 payload.' });
        return;
      }
      const m = t.imageBase64.match(/^data:(image\/(?:png|jpeg|webp));base64,(.+)$/);
      const mimeType = m ? m[1] : 'image/png';
      const b64 = m ? m[2] : t.imageBase64;
      validTiles.push({
        b64,
        mimeType,
        offsetX: Number(t.offsetX) || 0,
        offsetY: Number(t.offsetY) || 0,
        width: Math.max(64, Math.min(1024, Number(t.width) || detectionConfig.sahiTileSize)),
        height: Math.max(64, Math.min(1024, Number(t.height) || detectionConfig.sahiTileSize)),
      });
    }

    const ring: Array<[number, number]> = Array.isArray(polygonRingPx)
      ? polygonRingPx
          .filter(
            (p): p is [number, number] =>
              Array.isArray(p) && p.length >= 2 && Number.isFinite(Number(p[0])) && Number.isFinite(Number(p[1]))
          )
          .map(([x, y]) => [Number(x), Number(y)])
      : [];

    const mpp = Number(geoTransform?.mpp) > 0 ? Number(geoTransform?.mpp) : 0.26;
    const zoom = Number.isInteger(Number(geoTransform?.zoom)) ? Number(geoTransform?.zoom) : 19;
    const worldOriginX = Number.isFinite(Number(geoTransform?.worldOriginX))
      ? Number(geoTransform?.worldOriginX)
      : undefined;
    const worldOriginY = Number.isFinite(Number(geoTransform?.worldOriginY))
      ? Number(geoTransform?.worldOriginY)
      : undefined;

    const reqConfThresh = Math.max(
      0.25,
      Math.min(0.85, Number(confThresh ?? detectionConfig.aiModelConfThresh) || 0.55)
    );
    const effectiveMinConf = Math.max(
      0.2,
      Math.min(reqConfThresh, Number(minConf ?? reqConfThresh - detectionConfig.confIntervalDelta) || reqConfThresh)
    );

    const modelName = 'gemini-4-hydrogen';
    const thinkingLevel = isHighRes ? ThinkingLevel.HIGH : ThinkingLevel.LOW;
    const ai = new GoogleGenAI({
      apiKey: geminiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        detections: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              box_2d: {
                type: Type.ARRAY,
                items: { type: Type.NUMBER },
              },
              label: {
                type: Type.STRING,
                enum: ['orchard_tree', 'palm_tree', 'windbreak_tree'],
              },
              confidence: { type: Type.NUMBER },
              health: {
                type: Type.STRING,
                enum: ['healthy', 'stressed', 'dead'],
              },
            },
            required: ['box_2d', 'label', 'confidence', 'health'],
          },
        },
      },
      required: ['detections'],
    };

    const rawBoxes: ServerDetectionBox[] = [];
    const tileErrors: string[] = [];
    const concurrency = isHighRes ? 3 : 4;

    const runTile = async (tile: (typeof validTiles)[number]): Promise<void> => {
      const prompt = buildTreeVisionPrompt(tile.width, tile.height, mpp, isHighRes);
      const imagePart = {
        inlineData: {
          mimeType: tile.mimeType,
          data: tile.b64,
        },
      };
      const textPart = { text: prompt };

      let lastErr: unknown = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const response = await ai.models.generateContent({
            model: modelName,
            contents: { parts: [imagePart, textPart] },
            config: {
              systemInstruction: TREE_VISION_SYSTEM_INSTRUCTION,
              thinkingConfig: { thinkingLevel },
              seed: 42,
              responseMimeType: 'application/json',
              responseSchema,
            },
          });

          const rawText = (response.text ?? '{"detections":[]}').trim();
          const parsed = JSON.parse(rawText) as {
            detections?: Array<{
              box_2d?: unknown;
              label?: unknown;
              confidence?: unknown;
              health?: unknown;
            }>;
          };

          for (const item of parsed.detections || []) {
            if (!Array.isArray(item.box_2d) || item.box_2d.length < 4) continue;
            const [ny1, nx1, ny2, nx2] = item.box_2d.map(Number);
            if (![ny1, nx1, ny2, nx2].every(Number.isFinite)) continue;

            const confVal = Math.max(0, Math.min(1, Number(item.confidence ?? 0.7)));
            if (confVal < effectiveMinConf) continue;

            const ymin = tile.offsetY + (Math.min(ny1, ny2) / 1000) * tile.height;
            const xmin = tile.offsetX + (Math.min(nx1, nx2) / 1000) * tile.width;
            const ymax = tile.offsetY + (Math.max(ny1, ny2) / 1000) * tile.height;
            const xmax = tile.offsetX + (Math.max(nx1, nx2) / 1000) * tile.width;

            const bw = xmax - xmin;
            const bh = ymax - ymin;
            if (bw < 3 || bh < 3) continue;
            const aspect = Math.max(bw / bh, bh / bw);
            if (aspect > 2.6) continue;

            const diamM = ((bw + bh) / 2) * mpp;
            if (diamM < 0.8 || diamM > 13.0) continue;

            const rawLabel = String(item.label || 'orchard_tree');
            const label: TreeClassName =
              rawLabel === 'palm_tree' || rawLabel === 'windbreak_tree' ? rawLabel : 'orchard_tree';
            const rawHealth = String(item.health || 'healthy');
            const health: TreeHealthStatus =
              rawHealth === 'stressed' || rawHealth === 'dead' ? rawHealth : 'healthy';

            rawBoxes.push({
              xmin,
              ymin,
              xmax,
              ymax,
              label,
              confidence: confVal,
              health,
            });
          }
          return;
        } catch (err) {
          lastErr = err;
          const msg = err instanceof Error ? err.message : String(err);
          const isTransient = /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(msg);
          if (!isTransient || attempt === 2) break;
          await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
        }
      }
      tileErrors.push(lastErr instanceof Error ? lastErr.message : String(lastErr));
    };

    for (let i = 0; i < validTiles.length; i += concurrency) {
      await Promise.all(validTiles.slice(i, i + concurrency).map((tile) => runTile(tile)));
    }

    if (tileErrors.length === validTiles.length) {
      const isTransientAll = tileErrors.every((e) =>
        /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(e)
      );
      res.status(isTransientAll ? 503 : 502).json({
        ok: false,
        error: `Gemini Vision inference failed on all tiles: ${tileErrors[0]}`,
      });
      return;
    }

    const polygonFiltered =
      ring.length >= 3
        ? rawBoxes.filter((b) => pointInRing((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, ring))
        : rawBoxes;

    const minCenterDistPx = Math.max(3, Math.round((isHighRes ? 1.3 : 1.6) / Math.max(mpp, 0.1)));
    const fused = fuseServerDetectionsWBF(
      polygonFiltered,
      detectionConfig.wbfIouThresh ?? detectionConfig.nmsIouThresh,
      minCenterDistPx,
      detectionConfig.crossClassSuppressionIou
    );

    const detections = fused.map((d) => {
      if (worldOriginX !== undefined && worldOriginY !== undefined) {
        const cx = (d.xmin + d.xmax) / 2;
        const cy = (d.ymin + d.ymax) / 2;
        const [lon, lat] = worldPxToLonLatServer(worldOriginX + cx, worldOriginY + cy, zoom);
        return { ...d, lon, lat };
      }
      return d;
    });

    res.json({
      ok: true,
      engine: isHighRes
        ? 'AI Vision (Gemini 4 Hydrogen · High-Res SAHI WBF)'
        : 'AI Vision (Gemini 4 Hydrogen · Spatial WBF)',
      detections,
      partialTileErrors: tileErrors.length,
    });
  });

  // 5c-2. Multi-Library AI Arbitration Endpoint (Gemini 4 Hydrogen compares all CV/AI/YOLO libraries)
  app.post('/api/detect-trees/arbitrate', detectionLimiter, requireDetectionAuth, async (req, res) => {
    const geminiKey = process.env.GEMINI_API_KEY?.trim();
    if (!geminiKey) {
      res.status(503).json({
        ok: false,
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
      return;
    }

    const body = (req.body || {}) as {
      farmId?: unknown;
      species?: unknown;
      cropType?: unknown;
      vegPct?: unknown;
      mpp?: unknown;
      engineerCount?: unknown;
      forecast?: unknown;
      inwardBufferM?: unknown;
      highResMode?: unknown;
      provider?: unknown;
      consensusCount?: unknown;
      medianCount?: unknown;
      agreementPct?: unknown;
      libraries?: Array<{
        id?: unknown;
        name?: unknown;
        libraryBadge?: unknown;
        category?: unknown;
        status?: unknown;
        count?: unknown;
        interiorCount?: unknown;
        boundaryCount?: unknown;
        avgConfidence?: unknown;
        medianCrownM?: unknown;
        durationMs?: unknown;
      }>;
    };

    const libs = Array.isArray(body.libraries) ? body.libraries : [];
    const ranLibs = libs.filter((l) => l && l.status === 'ran' && typeof l.id === 'string');
    if (ranLibs.length < 2) {
      res.status(400).json({ ok: false, error: 'At least 2 ran libraries are required for arbitration.' });
      return;
    }

    const ranIds = ranLibs.map((l) => String(l.id));
    const ai = new GoogleGenAI({
      apiKey: geminiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const arbitrateSchema = {
      type: Type.OBJECT,
      properties: {
        winnerEngineId: {
          type: Type.STRING,
          enum: ranIds,
        },
        recommendedCount: { type: Type.NUMBER },
        confidencePct: { type: Type.NUMBER },
        explanationAr: { type: Type.STRING },
        cropRiskWarningAr: { type: Type.STRING },
      },
      required: ['winnerEngineId', 'recommendedCount', 'confidencePct', 'explanationAr'],
    };

    const systemInstruction = [
      'You are a Principal Remote Sensing & Orchard Tree-Counting Arbitrator.',
      'You receive tree counts from multiple independent computer vision and AI detection libraries on the same satellite farm image:',
      '(1) gemini_vision: Semantic Vision AI object detector (best at distinguishing tree crowns from flat berseem/wheat/vegetable crop fields and roofs);',
      '(2) yolov8_onnx: Trained YOLOv8 orchard detector if loaded;',
      '(3) turf_consensus_ensemble: Geo-DBSCAN spatial fusion across all active engines;',
      '(4) morphological_watershed: 4-Gate Meijster EDT Watershed;',
      '(5) multiscale_dog: Multi-Scale Gaussian Pyramid Difference-of-Gaussians blob detector;',
      '(6) shadow_texture_pair: Cast-Shadow Azimuth & texture pairing.',
      'Select the single most accurate engine from the ran list (winnerEngineId) and justify in clear concise Arabic why it wins over the other libraries, noting any over-counting by classical engines on dense flat crops or under-counting on small saplings.',
    ].join(' ');

    const payloadSummary = JSON.stringify(
      {
        farmId: String(body.farmId || 'FARM').slice(0, 64),
        species: body.species ?? '',
        cropType: body.cropType ?? '',
        vegPct: body.vegPct ?? null,
        mpp: body.mpp ?? 0.26,
        engineerCount: body.engineerCount ?? null,
        perimeterForecast: body.forecast ?? null,
        inwardBufferM: body.inwardBufferM ?? -1.5,
        highResMode: Boolean(body.highResMode),
        provider: body.provider ?? 'google',
        consensusCount: body.consensusCount ?? null,
        medianCount: body.medianCount ?? null,
        agreementPct: body.agreementPct ?? null,
        ranLibraries: ranLibs,
      },
      null,
      2
    );

    try {
      let response: Awaited<ReturnType<typeof ai.models.generateContent>> | null = null;
      let lastErr: unknown = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          response = await ai.models.generateContent({
            model: 'gemini-4-hydrogen',
            contents: `Compare these tree-detection library outputs and pick the most accurate winnerEngineId:\n${payloadSummary}`,
            config: {
              systemInstruction,
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
              seed: 42,
              responseMimeType: 'application/json',
              responseSchema: arbitrateSchema,
            },
          });
          break;
        } catch (err) {
          lastErr = err;
          const msg = err instanceof Error ? err.message : String(err);
          const isTransient = /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(msg);
          if (!isTransient || attempt === 2) throw err;
          await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
        }
      }
      if (!response) throw lastErr || new Error('Empty arbitration response');

      const parsed = JSON.parse((response.text ?? '{}').trim()) as {
        winnerEngineId?: unknown;
        recommendedCount?: unknown;
        confidencePct?: unknown;
        explanationAr?: unknown;
        cropRiskWarningAr?: unknown;
      };

      const winnerId = ranIds.includes(String(parsed.winnerEngineId))
        ? String(parsed.winnerEngineId)
        : ranIds.includes('turf_consensus_ensemble')
          ? 'turf_consensus_ensemble'
          : ranIds[0];
      const winnerRow = ranLibs.find((l) => String(l.id) === winnerId);
      const exactCount = Number(winnerRow?.count ?? parsed.recommendedCount ?? 0);

      res.json({
        ok: true,
        arbitration: {
          winnerEngineId: winnerId,
          recommendedCount: Math.max(0, Math.round(exactCount)),
          confidencePct: Math.max(25, Math.min(99, Math.round(Number(parsed.confidencePct ?? 85)))),
          explanationAr: String(parsed.explanationAr || '').trim(),
          cropRiskWarningAr: parsed.cropRiskWarningAr
            ? String(parsed.cropRiskWarningAr).trim()
            : undefined,
          generatedAt: new Date().toISOString(),
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient = /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(msg);
      res.status(isTransient ? 503 : 502).json({
        ok: false,
        error: isTransient ? 'Gemini arbitration temporarily busy' : msg,
      });
    }
  });

  // 5c-bis. Smart AI Tree Audit & Agronomic Discrepancy Reasoning (Gemini 4 Hydrogen)
  app.post('/api/audit-trees', detectionLimiter, requireDetectionAuth, async (req, res) => {
    const geminiKey = process.env.GEMINI_API_KEY?.trim();
    if (!geminiKey) {
      res.status(503).json({
        ok: false,
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
      return;
    }

    const body = (req.body || {}) as {
      farmId?: unknown;
      farmName?: unknown;
      region?: unknown;
      species?: unknown;
      cropType?: unknown;
      treeDetails?: unknown;
      areaEntered?: unknown;
      areaGps?: unknown;
      perimeterM?: unknown;
      engineerCount?: unknown;
      aiCount?: unknown;
      aiLow?: unknown;
      aiHigh?: unknown;
      forecast?: unknown;
      boundarySpacingM?: unknown;
      vegPct?: unknown;
      imageryDate?: unknown;
      imgAge?: unknown;
      verdict?: unknown;
      gpsWarnings?: unknown;
      provider?: unknown;
      zoomUsed?: unknown;
      inwardBufferM?: unknown;
      classBreakdown?: unknown;
      healthBreakdown?: unknown;
      spatialMetrics?: unknown;
    };

    const farmId = String(body.farmId || '').trim().slice(0, 64);
    if (!farmId) {
      res.status(400).json({ ok: false, error: 'farmId is required.' });
      return;
    }

    const ai = new GoogleGenAI({
      apiKey: geminiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const auditSchema = {
      type: Type.OBJECT,
      properties: {
        alignmentVerdict: {
          type: Type.STRING,
          enum: [
            'aligned',
            'under_counted_by_ai',
            'over_counted_by_engineer',
            'boundary_mismatch',
            'crop_confusion_risk',
          ],
        },
        confidencePct: { type: Type.NUMBER },
        discrepancyCauseAr: { type: Type.STRING },
        canopyQualitySummaryAr: { type: Type.STRING },
        fieldCheckRecommendationsAr: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
        },
        suggestedAuditorComment: { type: Type.STRING },
      },
      required: [
        'alignmentVerdict',
        'confidencePct',
        'discrepancyCauseAr',
        'canopyQualitySummaryAr',
        'fieldCheckRecommendationsAr',
        'suggestedAuditorComment',
      ],
    };

    const systemInstruction = [
      'You are an expert Egyptian PGS agricultural remote-sensing & orchard QA auditor.',
      'Analyze structured farm telemetry, tree crown counts, canopy spatial spacing, and Kobo engineer field counts.',
      'Never invent measurements not present in the payload. Explain discrepancies clearly in Arabic, accounting for:',
      '(a) young saplings (<1.5m) invisible at Z19-Z20, (b) inward boundary buffer excluding perimeter windbreaks,',
      '(c) flat crop understory (berseem/alfalfa/vegetables) causing high vegPct without trees,',
      '(d) merged double-crowns flagged in oversizedCrownIds, and (e) GPS area/alignment mismatches.',
    ].join(' ');

    const payloadSummary = JSON.stringify(
      {
        farmId,
        farmName: body.farmName,
        region: body.region,
        species: body.species,
        cropType: body.cropType,
        treeDetails: body.treeDetails,
        areaEnteredFeddan: body.areaEntered,
        areaGpsFeddan: body.areaGps,
        perimeterM: body.perimeterM,
        engineerCount: body.engineerCount,
        aiCount: body.aiCount,
        aiRange: [body.aiLow, body.aiHigh],
        perimeterForecast: body.forecast,
        boundarySpacingM: body.boundarySpacingM,
        vegPct: body.vegPct,
        imageryDate: body.imageryDate,
        imgAgeYears: body.imgAge,
        ruleVerdict: body.verdict,
        gpsWarnings: body.gpsWarnings,
        provider: body.provider,
        zoomUsed: body.zoomUsed,
        inwardBufferM: body.inwardBufferM,
        classBreakdown: body.classBreakdown,
        healthBreakdown: body.healthBreakdown,
        spatialMetrics: body.spatialMetrics,
      },
      null,
      2
    );

    try {
      let response: Awaited<ReturnType<typeof ai.models.generateContent>> | null = null;
      let lastErr: unknown = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          response = await ai.models.generateContent({
            model: 'gemini-4-hydrogen',
            contents: `Evaluate this Kobo Farm QA & satellite tree audit record and output structured JSON diagnostics:\n${payloadSummary}`,
            config: {
              systemInstruction,
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
              seed: 42,
              responseMimeType: 'application/json',
              responseSchema: auditSchema,
            },
          });
          break;
        } catch (err) {
          lastErr = err;
          const msg = err instanceof Error ? err.message : String(err);
          const isTransient = /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(msg);
          if (!isTransient || attempt === 2) throw err;
          await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
        }
      }
      if (!response) throw lastErr || new Error('Empty response');

      const parsed = JSON.parse((response.text ?? '{}').trim()) as {
        alignmentVerdict?: string;
        confidencePct?: number;
        discrepancyCauseAr?: string;
        canopyQualitySummaryAr?: string;
        fieldCheckRecommendationsAr?: string[];
        suggestedAuditorComment?: string;
      };

      const validVerdicts = [
        'aligned',
        'under_counted_by_ai',
        'over_counted_by_engineer',
        'boundary_mismatch',
        'crop_confusion_risk',
      ];
      const alignmentVerdict = validVerdicts.includes(String(parsed.alignmentVerdict))
        ? parsed.alignmentVerdict
        : 'aligned';

      res.json({
        ok: true,
        report: {
          alignmentVerdict,
          confidencePct: Math.max(10, Math.min(99, Math.round(Number(parsed.confidencePct ?? 80)))),
          discrepancyCauseAr: String(parsed.discrepancyCauseAr || '').trim(),
          canopyQualitySummaryAr: String(parsed.canopyQualitySummaryAr || '').trim(),
          fieldCheckRecommendationsAr: Array.isArray(parsed.fieldCheckRecommendationsAr)
            ? parsed.fieldCheckRecommendationsAr.map((s) => String(s)).slice(0, 5)
            : [],
          suggestedAuditorComment: String(parsed.suggestedAuditorComment || '').trim(),
          generatedAt: new Date().toISOString(),
        },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient = /503|429|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand/i.test(msg);
      res.status(isTransient ? 503 : 502).json({
        ok: false,
        error: isTransient
          ? 'خادم الذكاء الاصطناعي (Gemini) يشهد ضغطاً مؤقتاً الآن؛ يمكنك الاعتماد على التحليل المكاني الفوري أعلاه أو إعادة المحاولة بعد ثوانٍ.'
          : `Smart audit reasoning failed: ${msg}`,
      });
    }
  });

  // 5d. FastAPI Batch YOLO Delegation (/api/satellite/detect-batch)
  app.post('/api/satellite/detect-batch', detectionLimiter, requireDetectionAuth, async (req, res) => {
    const { tiles, conf = detectionConfig.aiModelConfThresh, iou = detectionConfig.nmsIouThresh } =
      (req.body || {}) as {
        tiles?: Array<{ imageBase64?: unknown; offsetX?: unknown; offsetY?: unknown }>;
        conf?: unknown;
        iou?: unknown;
      };

    if (!Array.isArray(tiles) || tiles.length === 0 || tiles.length > 48) {
      res.status(400).json({ ok: false, error: 'tiles array (1..48) is required' });
      return;
    }

    const remoteUrlRaw = (process.env.YOLO_INFER_URL || '').trim();
    if (!remoteUrlRaw) {
      res.json({ ok: false, engine: 'unavailable', boxes: [] });
      return;
    }

    try {
      const parsed = new URL(remoteUrlRaw);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw new Error('Invalid YOLO_INFER_URL protocol');
      }
      const targetUrl = new URL('/detect-batch', parsed.toString()).toString();
      const confNum = Math.max(0.05, Math.min(0.95, Number(conf) || detectionConfig.aiModelConfThresh));
      const iouNum = Math.max(0.1, Math.min(0.9, Number(iou) || detectionConfig.nmsIouThresh));

      const upstream = await fetch(targetUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tiles, conf: confNum, iou: iouNum }),
        signal: AbortSignal.timeout(30000),
      });

      if (!upstream.ok) {
        res.json({ ok: false, engine: 'unavailable', boxes: [] });
        return;
      }

      const data = (await upstream.json()) as { ok?: boolean; engine?: string; boxes?: unknown[] };
      res.json({
        ok: Boolean(data.ok),
        engine: data.engine || 'YOLOv8 FastAPI Batch',
        boxes: Array.isArray(data.boxes) ? data.boxes : [],
      });
    } catch {
      res.json({ ok: false, engine: 'unavailable', boxes: [] });
    }
  });

  app.post('/api/satellite/detect', detectionLimiter, requireDetectionAuth, async (req, res) => {
    const { imageBase64, conf = detectionConfig.aiModelConfThresh, iou = detectionConfig.nmsIouThresh } =
      (req.body || {}) as {
        imageBase64?: unknown;
        conf?: unknown;
        iou?: unknown;
      };

    if (typeof imageBase64 !== 'string' || imageBase64.length < 64 || imageBase64.length > 8_000_000) {
      res.status(400).json({ ok: false, error: 'Valid imageBase64 payload is required' });
      return;
    }

    const confNum = Math.max(0.05, Math.min(0.95, Number(conf) || detectionConfig.aiModelConfThresh));
    const iouNum = Math.max(0.1, Math.min(0.9, Number(iou) || detectionConfig.nmsIouThresh));

    const remoteUrlRaw = (process.env.YOLO_INFER_URL || '').trim();
    if (remoteUrlRaw) {
      try {
        const parsed = new URL(remoteUrlRaw);
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
          throw new Error('Invalid YOLO_INFER_URL protocol');
        }
        const batchUrl = new URL('/detect-batch', parsed.toString()).toString();
        const upstream = await fetch(batchUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tiles: [{ imageBase64, offsetX: 0, offsetY: 0 }],
            conf: confNum,
            iou: iouNum,
          }),
          signal: AbortSignal.timeout(20000),
        });
        if (upstream.ok) {
          const data = (await upstream.json()) as { engine?: string; boxes?: unknown[] };
          res.json({
            ok: true,
            engine: data.engine || 'YOLO Remote API',
            boxes: Array.isArray(data.boxes) ? data.boxes : [],
          });
          return;
        }
      } catch {
        // Fall through to local Python check
      }
    }

    const ptPath = resolveConfinedModel(['models/best.pt', 'best.pt']);
    if (ptPath && (await checkPythonYoloAvailable())) {
      const cleanB64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');
      const buf = Buffer.from(cleanB64, 'base64');
      const tmpImg = path.join(os.tmpdir(), `tile_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.png`);
      try {
        fs.writeFileSync(tmpImg, buf);
        const pyScript = [
          'import json, sys',
          'from ultralytics import YOLO',
          'model = YOLO(sys.argv[1])',
          'res = model.predict(source=sys.argv[2], conf=float(sys.argv[3]), iou=float(sys.argv[4]), verbose=False)',
          'out = []',
          'if res and len(res) > 0 and res[0].boxes is not None:',
          '    for b in res[0].boxes:',
          '        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]',
          '        cls_id = int(b.cls[0].item()) if b.cls is not None else 0',
          '        score = float(b.conf[0].item()) if b.conf is not None else 0.5',
          '        out.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, "classId": cls_id, "score": score})',
          'print(json.dumps({"boxes": out}))',
        ].join('\n');
        const { stdout } = await execFileAsync(
          'python3',
          ['-c', pyScript, ptPath, tmpImg, String(confNum), String(iouNum)],
          { timeout: 25000 }
        );
        const parsed = JSON.parse(stdout.trim() || '{"boxes":[]}');
        res.json({
          ok: true,
          engine: `YOLOv8 Backend (${path.basename(ptPath)})`,
          boxes: Array.isArray(parsed.boxes) ? parsed.boxes : [],
        });
        return;
      } catch {
        // Fall through cleanly
      } finally {
        try {
          if (fs.existsSync(tmpImg)) fs.unlinkSync(tmpImg);
        } catch {
          // ignore cleanup failure
        }
      }
    }

    res.json({
      ok: false,
      engine: 'unavailable',
      boxes: [],
    });
  });

  // 5e. Active-Learning Feedback Queue (Auditor-Corrected Raw Tiles + YOLO Labels)
  app.get('/api/dataset/feedback/stats', (_req, res) => {
    try {
      const imgDir = path.resolve(FEEDBACK_ROOT, 'images');
      if (!imgDir.startsWith(FEEDBACK_ROOT + path.sep) || !fs.existsSync(imgDir)) {
        res.json({ ok: true, queuedTiles: 0 });
        return;
      }
      const files = fs.readdirSync(imgDir).filter((f) => f.endsWith('.png'));
      res.json({ ok: true, queuedTiles: files.length });
    } catch {
      res.json({ ok: true, queuedTiles: 0 });
    }
  });

  app.post('/api/dataset/feedback', feedbackLimiter, requireDetectionAuth, (req, res) => {
    const { farmId, provider, auditor, manualDelta, tiles } = (req.body || {}) as {
      farmId?: unknown;
      provider?: unknown;
      auditor?: unknown;
      manualDelta?: { added?: unknown; deleted?: unknown; reclassified?: unknown };
      tiles?: Array<{
        imageBase64?: unknown;
        annotations?: Array<{
          classId?: unknown;
          cx?: unknown;
          cy?: unknown;
          w?: unknown;
          h?: unknown;
        }>;
      }>;
    };

    const safeFarm = String(farmId || '')
      .replace(/[^a-zA-Z0-9_-]/g, '')
      .slice(0, 48);
    const safeProvider = String(provider || 'google')
      .replace(/[^a-zA-Z0-9_-]/g, '')
      .slice(0, 16);

    if (!safeFarm || !Array.isArray(tiles) || tiles.length === 0 || tiles.length > 64) {
      res.status(400).json({ ok: false, error: 'Valid farmId and 1..64 tiles are required.' });
      return;
    }

    const imagesDir = path.resolve(FEEDBACK_ROOT, 'images');
    const labelsDir = path.resolve(FEEDBACK_ROOT, 'labels');
    const metaPath = path.resolve(FEEDBACK_ROOT, 'metadata.json');

    if (
      !imagesDir.startsWith(FEEDBACK_ROOT + path.sep) ||
      !labelsDir.startsWith(FEEDBACK_ROOT + path.sep) ||
      !metaPath.startsWith(FEEDBACK_ROOT + path.sep)
    ) {
      res.status(400).json({ ok: false, error: 'Invalid feedback path.' });
      return;
    }

    fs.mkdirSync(imagesDir, { recursive: true });
    fs.mkdirSync(labelsDir, { recursive: true });

    const tsTag = Date.now();
    let savedTiles = 0;

    tiles.forEach((t, idx) => {
      if (typeof t?.imageBase64 !== 'string' || t.imageBase64.length < 32) return;
      const cleanB64 = t.imageBase64.replace(/^data:image\/\w+;base64,/, '');
      const buf = Buffer.from(cleanB64, 'base64');
      if (buf.byteLength === 0 || buf.byteLength > 5 * 1024 * 1024) return;

      const baseName = `fb_${safeFarm}_${safeProvider}_${tsTag}_${String(idx + 1).padStart(3, '0')}`;
      const imgPath = path.resolve(imagesDir, `${baseName}.png`);
      const lblPath = path.resolve(labelsDir, `${baseName}.txt`);
      if (!imgPath.startsWith(imagesDir + path.sep) || !lblPath.startsWith(labelsDir + path.sep)) {
        return;
      }

      const yoloLines: string[] = [];
      if (Array.isArray(t.annotations)) {
        for (const a of t.annotations) {
          const cid = Number(a.classId);
          const cx = Number(a.cx);
          const cy = Number(a.cy);
          const w = Number(a.w);
          const h = Number(a.h);
          if (
            ![0, 1, 2].includes(cid) ||
            ![cx, cy, w, h].every((v) => Number.isFinite(v) && v >= 0 && v <= 1)
          ) {
            continue;
          }
          yoloLines.push(`${cid} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`);
        }
      }

      fs.writeFileSync(imgPath, buf);
      fs.writeFileSync(lblPath, yoloLines.join('\n'), 'utf-8');
      savedTiles++;
    });

    let manifest: Array<Record<string, unknown>> = [];
    try {
      if (fs.existsSync(metaPath)) {
        const parsed = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
        if (Array.isArray(parsed)) manifest = parsed;
      }
    } catch {
      manifest = [];
    }

    manifest.push({
      farmId: safeFarm,
      provider: safeProvider,
      auditor: String(auditor || '').slice(0, 64),
      savedTiles,
      manualDelta: {
        added: Number(manualDelta?.added || 0),
        deleted: Number(manualDelta?.deleted || 0),
        reclassified: Number(manualDelta?.reclassified || 0),
      },
      createdAt: new Date().toISOString(),
    });
    if (manifest.length > 500) manifest = manifest.slice(-500);
    fs.writeFileSync(metaPath, JSON.stringify(manifest, null, 2), 'utf-8');

    const queuedTotal = fs.readdirSync(imagesDir).filter((f) => f.endsWith('.png')).length;
    res.json({ ok: true, savedTiles, queuedTotal });
  });

  // 6. Team Assignments & Work Distribution Pipeline
  app.get('/api/team/assignments', async (_req, res) => {
    const apiKey = process.env.AIRTABLE_API_KEY?.trim();
    const baseId = process.env.AIRTABLE_BASE_ID?.trim();
    const tableName = process.env.AIRTABLE_TABLE_NAME?.trim();

    if (apiKey && baseId && tableName) {
      try {
        const airResp = await fetch(
          `https://api.airtable.com/v0/${encodeURIComponent(baseId)}/${encodeURIComponent(tableName)}`,
          {
            headers: { Authorization: `Bearer ${apiKey}` },
            signal: AbortSignal.timeout(12000),
          }
        );
        if (airResp.ok) {
          const airData = (await airResp.json()) as {
            records?: Array<{ fields?: Record<string, unknown> }>;
          };
          for (const rec of airData.records || []) {
            const fields = rec.fields || {};
            const subId = String(fields.submission_id || fields.sub_id || '').trim();
            const toKey = String(fields.assigned_to || '').trim().toLowerCase() as TeamMemberKey;
            if (subId && TEAM_KEYS.includes(toKey)) {
              assignments.set(subId, {
                to: toKey,
                by: String(fields.assigned_by || 'airtable'),
                at: String(fields.Reviewed_At || nowStr()),
                farmId: String(fields.farm_id || ''),
              });
            }
          }
        }
      } catch {
        // Fallback to local assignments
      }
    }

    res.json({ assignments: Object.fromEntries(assignments.entries()) });
  });

  app.post('/api/team/reassign', (req, res) => {
    const { subIds, target, by, farmIds } = req.body as {
      subIds?: string[];
      target?: TeamMemberKey;
      by?: string;
      farmIds?: Record<string, string>;
    };
    if (!Array.isArray(subIds) || !target || !TEAM_KEYS.includes(target)) {
      res.status(400).json({ error: 'Invalid reassignment parameters' });
      return;
    }
    const ts = nowStr();
    for (const sid of subIds) {
      assignments.set(String(sid), {
        to: target,
        by: by || 'manual',
        at: ts,
        farmId: farmIds?.[sid],
      });
    }
    res.json({
      ok: true,
      count: subIds.length,
      assignments: Object.fromEntries(assignments.entries()),
    });
  });

  app.post('/api/team/redistribute', (req, res) => {
    const { subIds, by, farmIds } = req.body as {
      subIds?: string[];
      by?: string;
      farmIds?: Record<string, string>;
    };
    if (!Array.isArray(subIds)) {
      res.status(400).json({ error: 'subIds array is required' });
      return;
    }
    const counts: Record<TeamMemberKey, number> = {
      mohamed: 0,
      mansy: 0,
      haidy: 0,
      islam: 0,
    };
    const ts = nowStr();
    subIds.forEach((sid, idx) => {
      const target = TEAM_KEYS[idx % TEAM_KEYS.length];
      counts[target]++;
      assignments.set(String(sid), {
        to: target,
        by: by || 'redistribute',
        at: ts,
        farmId: farmIds?.[sid],
      });
    });
    res.json({
      ok: true,
      counts,
      assignments: Object.fromEntries(assignments.entries()),
    });
  });

  app.post('/api/team/divide-work', (req, res) => {
    const { items, by } = req.body as {
      items?: UnvalidatedFarmItem[];
      by?: string;
    };
    const unvalidated = Array.isArray(items) ? items : [];
    const result = divideWorkFairly(unvalidated, TEAM_KEYS, 1.4);
    const ts = nowStr();

    for (const worker of TEAM_KEYS) {
      for (const f of result.workerFarms[worker]) {
        assignments.set(String(f.subId), {
          to: worker,
          by: by || 'pipeline',
          at: ts,
          farmId: f.farmCode,
        });
      }
    }

    res.json({
      ok: true,
      logs: result.logs,
      assignments: Object.fromEntries(assignments.entries()),
    });
  });

  // 7. Reviews Storage (Master + Field-level)
  app.get('/api/reviews', (_req, res) => {
    res.json({
      master: Object.fromEntries(masterReviews.entries()),
      fields: Object.fromEntries(fieldReviews.entries()),
    });
  });

  app.post('/api/reviews/save', (req, res) => {
    const { master, fields } = req.body as {
      master?: MasterReviewRecord;
      fields?: FieldDecisionRow[];
    };
    if (!master || !master.farm_id) {
      res.status(400).json({ error: 'Invalid master review payload' });
      return;
    }
    const fid = String(master.farm_id);
    masterReviews.set(fid, master);

    if (Array.isArray(fields)) {
      const filtered = fields.filter(
        (f) =>
          f.Field_Key.startsWith('_AI_') ||
          f.Item_Decision !== 'OK' ||
          f.Suggested_Correction_or_Comment.trim() !== '' ||
          f.Attach_to_Summary
      );
      fieldReviews.set(fid, filtered);
    }

    res.json({ ok: true });
  });

  // 8. Airtable Upsert Sync
  app.post('/api/airtable/sync', async (req, res) => {
    const rec = req.body as MasterReviewRecord;
    if (!rec || !rec.farm_id) {
      res.status(400).json({ ok: false, message: 'بيانات المزرعة ناقصة.' });
      return;
    }

    masterReviews.set(String(rec.farm_id), rec);

    const apiKey = process.env.AIRTABLE_API_KEY?.trim();
    const baseId = process.env.AIRTABLE_BASE_ID?.trim();
    const tableName = process.env.AIRTABLE_TABLE_NAME?.trim();

    if (!apiKey || !baseId || !tableName) {
      res.json({
        ok: true,
        message:
          'تم حفظ السجل في قاعدة المراجعة المحلية (أضف AIRTABLE_API_KEY و AIRTABLE_BASE_ID و AIRTABLE_TABLE_NAME للمزامنة السحابية المباشرة).',
      });
      return;
    }

    const row = {
      farm_id: String(rec.farm_id || ''),
      submission_id: String(rec.submission_id || ''),
      Form_Status: String(rec.Form_Status || ''),
      Rejected_Items_Count: Number(rec.Rejected_Items_Count || 0),
      Consolidated_Comments: String(rec.Consolidated_Comments || ''),
      General_Auditor_Note: String(rec.General_Auditor_Note || ''),
      Auditor_Name: String(rec.Auditor_Name || ''),
      Reviewed_At: String(rec.Reviewed_At || ''),
    };

    try {
      const airResp = await fetch(
        `https://api.airtable.com/v0/${encodeURIComponent(baseId)}/${encodeURIComponent(tableName)}`,
        {
          method: 'PATCH',
          headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            performUpsert: { fieldsToMergeOn: ['farm_id'] },
            typecast: true,
            records: [{ fields: row }],
          }),
          signal: AbortSignal.timeout(25000),
        }
      );

      if (!airResp.ok) {
        const bodyText = (await airResp.text()).slice(0, 300);
        const hints: Record<number, string> = {
          401: 'الـ API key غلط أو منتهي.',
          403: 'التوكن معندوش صلاحية data.records:write أو مش مدّي وصول للـ Base.',
          404: 'الـ base_id أو table_name غلط.',
          422: 'اسم عمود غلط أو نوع بيانات مش مطابق في الجدول.',
        };
        const hint = hints[airResp.status] || '';
        res.json({
          ok: false,
          message: `Airtable رفض الطلب (${airResp.status}). ${hint} ${bodyText}`,
        });
        return;
      }

      const result = (await airResp.json()) as { createdRecords?: string[] };
      if (result.createdRecords && result.createdRecords.length > 0) {
        res.json({ ok: true, message: 'تم الإرسال — اتضافت المزرعة كسجل جديد في Airtable.' });
      } else {
        res.json({ ok: true, message: 'تم الإرسال — اتحدّث السجل الموجود للمزرعة في Airtable.' });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      res.json({ ok: false, message: `خطأ في الاتصال بـ Airtable: ${msg}` });
    }
  });

  if (process.env.NODE_ENV === 'production') {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`EBDA & Kobo Farm QA server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
