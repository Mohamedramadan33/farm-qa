import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
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
} from './src/types/qa.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const KOBO_SERVER = (process.env.KOBO_SERVER || 'https://kf.kobotoolbox.org').replace(/\/+$/, '');
const KOBO_ASSET_UID = process.env.KOBO_ASSET_UID || 'aX4NJWgge6tooXjfSYXhrq';

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

// Initialize balanced initial assignments across the 4 EBDA team members for seed submissions
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
  app.use(express.json({ limit: '15mb' }));

  // 1. Kobo Submissions & Survey Labels
  app.get('/api/kobo/submissions', async (req, res) => {
    try {
      const refresh = req.query.refresh === '1';
      const bundle = await getSubmissionsBundle(refresh);

      // Auto-assign any unassigned pending submissions evenly across team members
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
    const maxMb = Number(req.query.maxMb || 15);
    if (!rawUrl || !rawUrl.startsWith('http')) {
      res.status(400).json({ error: 'Valid media URL is required' });
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
      if (len > 0 && len > maxMb * 1024 * 1024) {
        res.status(413).json({ error: `Attachment exceeds ${maxMb}MB limit` });
        return;
      }
      const contentType = upstream.headers.get('content-type') || 'application/octet-stream';
      const arrBuf = await upstream.arrayBuffer();
      if (arrBuf.byteLength > maxMb * 1024 * 1024) {
        res.status(413).json({ error: `Attachment exceeds ${maxMb}MB limit` });
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

  // 4. CORS-Safe Satellite Tile Proxy (ESRI World Imagery & Google Earth)
  app.get('/api/satellite/tile/:provider/:z/:x/:y', async (req, res) => {
    const { provider, z, x, y } = req.params;
    const cacheKey = `${provider}:${z}:${x}:${y}`;
    const hit = tileCache.get(cacheKey);
    if (hit) {
      res.setHeader('Content-Type', hit.contentType);
      res.setHeader('Cache-Control', 'public, max-age=86400');
      res.send(hit.buf);
      return;
    }

    const tileUrl =
      provider === 'google'
        ? `https://mt1.google.com/vt/lyrs=s&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&z=${encodeURIComponent(z)}`
        : `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${encodeURIComponent(z)}/${encodeURIComponent(y)}/${encodeURIComponent(x)}`;

    try {
      const upstream = await fetch(tileUrl, {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        },
        signal: AbortSignal.timeout(10000),
      });
      if (!upstream.ok) {
        res.status(upstream.status).end();
        return;
      }
      const arrBuf = await upstream.arrayBuffer();
      const buf = Buffer.from(arrBuf);
      if (buf.length < 400) {
        res.status(404).end();
        return;
      }
      const contentType = upstream.headers.get('content-type') || 'image/jpeg';
      if (tileCache.size > 600) {
        const oldest = tileCache.keys().next().value;
        if (oldest) tileCache.delete(oldest);
      }
      tileCache.set(cacheKey, { buf, contentType });
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'public, max-age=86400');
      res.send(buf);
    } catch {
      res.status(502).end();
    }
  });

  // 5. Satellite Capture Date (ArcGIS SRC_DATE2) & Recent Sentinel-2 STAC NDVI Metadata
  app.get('/api/satellite/metadata', async (req, res) => {
    const lon = Number(req.query.lon);
    const lat = Number(req.query.lat);
    const useNdvi = req.query.useNdvi === '1' || req.query.useNdvi === 'true';
    const minx = Number(req.query.minx ?? lon - 0.001);
    const miny = Number(req.query.miny ?? lat - 0.001);
    const maxx = Number(req.query.maxx ?? lon + 0.001);
    const maxy = Number(req.query.maxy ?? lat + 0.001);

    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      res.status(400).json({ imageryDate: null, ndviRecent: null });
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

  // 6. Team Assignments & Work Distribution Pipeline
  app.get('/api/team/assignments', async (_req, res) => {
    // Optionally hydrate from Airtable if configured
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

  // 8. Airtable Upsert Sync (Port of append_to_airtable in app.py)
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
