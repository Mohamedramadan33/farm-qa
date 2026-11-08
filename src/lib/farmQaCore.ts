import { normAr, normalizeDigits, present, toNumber } from './arabicNorm';
import {
  DUP_IOU,
  geomMetricsUtm,
  gpsQuality,
  parsePoints,
  parsePolygon,
  polygonIoU,
} from './geometry';
import {
  STATUS_PENDING,
  type AttachmentInfo,
  type FarmAuditRecord,
  type FlattenedRow,
  type LabelMap,
  type MasterReviewRecord,
  type RawKoboSubmission,
} from '../types/qa';

export const TOL = 0.2;
export const AREA_TOL = 0.1;
export const ABS_TOL = 5;
export const AI_BAND: [number, number, number] = [0.38, 0.2, 0.12];
export const VEG_EXG = 0.04;
export const VEG_REVIEW = 25.0;
export const IMAGERY_MAX_AGE_YEARS = 2.0;
export const YOUNG_CROWN_M = 2.0;
export const DEFAULT_BOUNDARY_SPACING_M = 3.0;
export const NDVI_GREEN = 0.3;
export const NDVI_BARE = 0.15;
export const NDVI_OLD_IMG_YEARS = 1.0;

export const META_PREFIXES = [
  '_',
  'meta/',
  'formhub/',
  '__version__',
  'deviceid',
  'simserial',
  'phonenumber',
  'subscriberid',
];

export const REGION_KW = [
  'المنطقة',
  'المحافظة',
  'المركز',
  'القرية',
  'الجمعية',
  'region',
  'governorate',
  'village',
];

export const FARMER_KW = [
  'اسم المزارع',
  'اسم الفلاح',
  'اسم صاحب المزرعة',
  'اسم صاحب',
  'farmer',
];

export const BOUNDARY_SPACING_M: Record<string, number> = {
  جازورين: 2.5,
  كازورينا: 2.5,
  casuarina: 2.5,
  كافور: 2.5,
  ماهوجني: 3.0,
  mahogany: 3.0,
  فيكس: 3.0,
  ficus: 3.0,
  توت: 3.5,
  نخيل: 6.0,
  نخل: 6.0,
  palm: 6.0,
  زيتون: 5.0,
  olive: 5.0,
  مانجو: 5.0,
  مانجا: 5.0,
  mango: 5.0,
  موالح: 4.0,
  citrus: 4.0,
  موز: 3.0,
  banana: 3.0,
};

interface FieldHint {
  exact: string[];
  kw: string[];
}

export const FIELD_HINTS: Record<string, FieldHint> = {
  time: { exact: ['_submission_time', 'start'], kw: [] },
  farm_id: { exact: ['farm_id'], kw: ['كود المزرعه', 'رقم المزرعه', 'farm id', 'farm code'] },
  farm_name: { exact: ['farm_name_auto', 'farm_name'], kw: ['اسم المزرعة', 'farm name'] },
  engineer: { exact: ['engineer_name'], kw: ['المهندس', 'engineer'] },
  visit_date: { exact: ['visit_date'], kw: ['تاريخ الزيارة'] },
  audit_type: { exact: ['audit_type'], kw: ['نوع الزيارة'] },
  season: { exact: ['__010', 'agri_season'], kw: ['الموسم', 'season'] },
  farm_area: { exact: ['farm_area'], kw: ['مساحه المزرعه', 'farm area'] },
  plot_area: { exact: ['plot_area', 'calculated_area', '__015'], kw: ['مساحه القطعه', 'مساحه', 'area'] },
  gps: {
    exact: ['gps_outline_of_farm'],
    kw: ['gps', 'حدود', 'مضلع', 'احداثيات', 'polygon', 'boundary'],
  },
  tree_count: {
    exact: ['tree_number', 'العدد', '__091'],
    kw: ['عدد الاشجار', 'عدد الشجر', 'tree count', 'العدد', 'عدد'],
  },
  tree_type: {
    exact: ['tree_type', 'النوع', '__016'],
    kw: ['نوع الشجر', 'نوع الاشجار', 'tree type', 'crop', 'محصول', 'النوع', 'نوع'],
  },
  plot_owner: { exact: ['plot_owner', '__011'], kw: ['صاحب الحوض', 'المالك'] },
  owner_phone: { exact: ['plot_owner_number', '__012'], kw: ['رقم تليفون', 'تليفون', 'phone'] },
  ownership: { exact: ['__009', 'ownership_type'], kw: ['ملكية', 'مملوكة'] },
  soil_type: { exact: ['__047', 'soil_type'], kw: ['نوع التربة', 'soil'] },
  irrig_type: { exact: ['__048', 'irrigation_type'], kw: ['نوع الري', 'طريقة الري'] },
  irrig_amnt: { exact: ['irrigation_amount_feddan_year'], kw: ['كمية الاستهلاك'] },
  crop_type: { exact: ['__030', 'crop_type_irrigation'], kw: ['المحصول', 'نوع المحصول'] },
  animal_type: { exact: ['_Type', 'farm_animals_type'], kw: ['نوع الحيوان', 'حيوانات'] },
  animal_count: { exact: ['__077', 'cow_numbers'], kw: ['عدد الحيوانات'] },
  non_compliance: { exact: ['non_compliance'], kw: ['عدم المطابقة', 'مخالفة'] },
  corrective_act: { exact: ['corrective_measures'], kw: ['إجراءات تصحيحية', 'تصحيح'] },
  auditor_recom: { exact: ['auditors_recommendations'], kw: ['توصيات لجنة المراجعة', 'توصيات'] },
};

export type ColsMap = Record<string, string[]>;

export function isMetaColumn(col: string): boolean {
  return META_PREFIXES.some((p) => col.startsWith(p));
}

export function flattenObject(
  obj: unknown,
  parent = '',
  sep = '/'
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (obj !== null && typeof obj === 'object' && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      if (k === '_attachments') continue;
      const nextKey = parent ? `${parent}${sep}${k}` : k;
      Object.assign(out, flattenObject(v, nextKey, sep));
    }
  } else if (Array.isArray(obj)) {
    if (obj.length > 0 && obj.every((item) => item !== null && typeof item === 'object')) {
      obj.forEach((item, i) => {
        Object.assign(out, flattenObject(item, `${parent}[${i}]`, sep));
      });
    } else {
      out[parent] = obj.map(String).join(';');
    }
  } else {
    out[parent] = typeof obj === 'string' ? normalizeDigits(obj) : obj;
  }
  return out;
}

export function buildFlattenedRows(rawRows: RawKoboSubmission[]): {
  rows: FlattenedRow[];
  columns: string[];
} {
  const colSet = new Set<string>();
  const rows = rawRows.map((r) => {
    const flat = flattenObject(r);
    for (const k of Object.keys(flat)) {
      colSet.add(k);
    }
    return flat;
  });
  return { rows, columns: Array.from(colSet) };
}

function leafName(col: string): string {
  const cleaned = col.replace(/\[\d+\]/g, '');
  const parts = cleaned.split('/');
  return (parts[parts.length - 1] || '').toLowerCase();
}

function shareMatching(rows: FlattenedRow[], col: string, pred: (v: unknown) => boolean): number {
  const vals = rows.map((r) => r[col]).filter(present);
  if (!vals.length) return 0;
  let matches = 0;
  for (const v of vals) {
    if (pred(v)) matches++;
  }
  return matches / vals.length;
}

function kwScore(col: string, kws: string[]): number {
  const name = normAr(leafName(col).replace(/[_-]/g, ' '));
  let best = 0;
  for (const k of kws) {
    const nk = normAr(k);
    if (nk && name.includes(nk) && nk.length > best) {
      best = nk.length;
    }
  }
  return best;
}

export function findCols(rows: FlattenedRow[], columns: string[], field: string): string[] {
  const hint = FIELD_HINTS[field];
  if (!hint) return [];

  const exactSet = new Set(hint.exact.map((x) => x.toLowerCase()));
  const exact = columns.filter((c) => exactSet.has(leafName(c)));
  if (exact.length > 0) return exact;

  const nonMeta = columns.filter((c) => !c.startsWith('_'));
  const isNum = (v: unknown) => toNumber(v) !== null;

  const checks: Record<string, (col: string) => boolean> = {
    gps: (col) => shareMatching(rows, col, (v) => parsePoints(v) !== null) >= 0.5,
    tree_count: (col) => shareMatching(rows, col, isNum) >= 0.5,
    farm_area: (col) => shareMatching(rows, col, isNum) >= 0.5,
    plot_area: (col) => shareMatching(rows, col, isNum) >= 0.5,
    tree_type: (col) => shareMatching(rows, col, (v) => !isNum(v)) >= 0.5,
  };

  const chk = checks[field];
  const cand = nonMeta.filter((c) => !chk || chk(c));

  if (field === 'gps') {
    cand.sort((a, b) => kwScore(b, hint.kw) - kwScore(a, hint.kw));
    return cand.slice(0, 1);
  }

  const scored = cand.map((c) => [c, kwScore(c, hint.kw)] as const);
  const top = scored.reduce((m, [, s]) => Math.max(m, s), 0);
  return top > 0 ? scored.filter(([, s]) => s === top).map(([c]) => c) : [];
}

export function mapAllColumns(rows: FlattenedRow[], columns: string[]): ColsMap {
  const out: ColsMap = {};
  for (const f of Object.keys(FIELD_HINTS)) {
    out[f] = findCols(rows, columns, f);
  }
  return out;
}

export function resolveLabel(colPath: string, labelMap: LabelMap): [string, string] {
  const clean = colPath.replace(/\[\d+\]/g, '').toLowerCase();
  const parts = clean.split('/');
  const leaf = parts[parts.length - 1] || clean;
  const idxMatches = colPath.match(/\[\d+\]/g);
  const prefix = idxMatches && idxMatches.length > 0 ? `${idxMatches[0]} ` : '';
  const entry = labelMap[clean] || labelMap[leaf];
  if (entry) {
    const [first, joined] = entry;
    return [first, `${prefix}${joined} (${leaf})`];
  }
  return [leaf, colPath];
}

export function findColsByLabel(
  columns: string[],
  labelMap: LabelMap,
  keywords: string[]
): string[] {
  const out: string[] = [];
  for (const kw of keywords) {
    const k = normAr(kw);
    for (const c of columns) {
      if (isMetaColumn(c) || out.includes(c)) continue;
      const [first] = resolveLabel(c, labelMap);
      if (normAr(first).includes(k)) {
        out.push(c);
      }
    }
  }
  return out;
}

export function firstValue(row: FlattenedRow, cols: string[]): string | null {
  for (const c of cols) {
    if (present(row[c])) {
      return String(row[c]).trim();
    }
  }
  return null;
}

export function engineerEntries(
  row: FlattenedRow,
  countCols: string[],
  typeCols: string[]
): Array<[string, number]> {
  const out: Array<[string, number]> = [];
  const getPrefix = (col: string) =>
    col.includes('/') ? col.slice(0, col.lastIndexOf('/')) : 'root_flat';

  const typesByPrefix: Record<string, unknown> = {};
  for (const c of typeCols) {
    if (present(row[c])) typesByPrefix[getPrefix(c)] = row[c];
  }
  const countsByPrefix: Record<string, unknown> = {};
  for (const c of countCols) {
    if (present(row[c])) countsByPrefix[getPrefix(c)] = row[c];
  }

  for (const [prefix, tVal] of Object.entries(typesByPrefix)) {
    if (prefix in countsByPrefix) {
      const n = toNumber(countsByPrefix[prefix]);
      if (n !== null) {
        out.push([String(tVal).trim(), n]);
      }
    }
  }

  for (const [prefix, cVal] of Object.entries(countsByPrefix)) {
    if (!(prefix in typesByPrefix)) {
      const n = toNumber(cVal);
      if (n !== null) {
        out.push(['Unspecified', n]);
      }
    }
  }

  return out;
}

export function spacingForBoundary(species: string): [number, boolean] {
  const s = present(species) ? normAr(species) : '';
  for (const [k, v] of Object.entries(BOUNDARY_SPACING_M)) {
    if (s && s.includes(normAr(k))) {
      return [v, true];
    }
  }
  return [DEFAULT_BOUNDARY_SPACING_M, false];
}

function isClose(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= Math.max(tol * Math.max(a, b), ABS_TOL);
}

export function computeVerdict(params: {
  E: number | null;
  A: number | null;
  F: number | null;
  tol?: number;
  band?: [number, number] | null;
  vegPct?: number | null;
  imgAge?: number | null;
  crownM?: number | null;
}): string {
  const { E, A, F, tol = TOL, band = null, vegPct = null, imgAge = null, crownM = null } = params;

  if (E === null || !Number.isFinite(E) || E <= 0) {
    return 'No valid boundary tree count recorded by the engineer.';
  }
  if (F === null || !Number.isFinite(F)) {
    return '[Manual Review]: Cannot calculate perimeter tree forecast (missing or invalid GPS).';
  }
  if (A === null || !Number.isFinite(A)) {
    if (E > F * (1 + tol) + ABS_TOL) {
      return '[Over Perimeter Capacity]: Reported count exceeds boundary capacity (Unverified by satellite).';
    }
    return "[Unverified by Satellite]: Count fits farm perimeter, click 'Run satellite AI' to verify.";
  }

  const [lo, hi] = band ?? [A, A];
  const inBand = (x: number) => lo * (1 - tol) - ABS_TOL <= x && x <= hi * (1 + tol) + ABS_TOL;
  const overCapacity = E > F * (1 + tol) + ABS_TOL;
  const aiMissesE = E > hi * (1 + tol) + ABS_TOL;
  const aiSeesMore = lo > E * (1 + tol) + ABS_TOL;
  const old = imgAge !== null && Number.isFinite(imgAge) && imgAge > IMAGERY_MAX_AGE_YEARS;

  if (inBand(E) && (inBand(F) || isClose(E, F, tol) || E <= F * (1 + tol) + ABS_TOL)) {
    return '[Excellent Boundary Match]: Engineer count aligns with Satellite AI on the farm perimeter.';
  }
  if (overCapacity && aiMissesE) {
    if (old) {
      return `[Review - Outdated Imagery]: Count exceeds perimeter capacity & not seen by AI, but imagery is old (~${imgAge.toFixed(1)} yrs).`;
    }
    if (vegPct !== null && Number.isFinite(vegPct) && vegPct >= VEG_REVIEW) {
      return `[Review]: Count exceeds perimeter capacity, but boundary has dense vegetation (${Math.round(vegPct)}%).`;
    }
    return '[Major Boundary Discrepancy]: Reported count exceeds perimeter capacity with no satellite evidence.';
  }
  if (aiSeesMore) {
    return '[Boundary Under-counting]: Satellite AI detected more trees along the farm boundary than reported.';
  }
  if (!overCapacity && aiMissesE) {
    let why = '';
    if (crownM !== null && Number.isFinite(crownM) && crownM < YOUNG_CROWN_M) {
      why = ` Avg crown diameter ~${crownM.toFixed(1)}m (young saplings).`;
    }
    return (
      '[Perimeter Capacity Match]: Farm perimeter fits the reported count, but AI detects fewer mature crowns (likely young saplings or partial border planting).' +
      why
    );
  }
  return '[Moderate Variance]: Noticeable difference between Engineer, Perimeter Forecast, and AI; manual review recommended.';
}

export function applyNdvi(
  verdictText: string,
  ndvi: number | null,
  veg: number | null,
  imgAge: number | null
): string {
  if (ndvi === null || !Number.isFinite(ndvi)) return verdictText;
  const oldImg = imgAge !== null && Number.isFinite(imgAge) && imgAge > NDVI_OLD_IMG_YEARS;
  const weakVeg = veg === null || !Number.isFinite(veg) || veg < 5;
  if (oldImg && weakVeg && ndvi >= NDVI_GREEN) {
    return `[Review - Recent Greening]: Recent Sentinel-2 NDVI is ${ndvi.toFixed(2)} while older satellite imagery shows low vegetation. Previous verdict may be outdated. — ${verdictText}`;
  }
  if (ndvi < NDVI_BARE) {
    return `${verdictText} (Recent NDVI ${ndvi.toFixed(2)}: Bare soil currently, supporting imagery result)`;
  }
  return verdictText;
}

export function computeConfidence(rec: {
  img_age: number | null;
  gps_warnings: string;
  area_check: string;
  density_note: string;
  ai_error: string;
}): 'High' | 'Medium' | 'Low' {
  let s = 100;
  if (rec.img_age !== null && Number.isFinite(rec.img_age) && rec.img_age > IMAGERY_MAX_AGE_YEARS) {
    s -= 30;
  }
  if (rec.gps_warnings) s -= 15;
  if (rec.area_check.startsWith('High Discrepancy') || rec.area_check.startsWith('Calculation Failed')) {
    s -= 20;
  }
  if (rec.density_note) s -= 10;
  if (rec.ai_error) s -= 40;
  return s >= 80 ? 'High' : s >= 55 ? 'Medium' : 'Low';
}

export function buildAttachmentLookup(rawRow: RawKoboSubmission): Record<string, AttachmentInfo> {
  const lookup: Record<string, AttachmentInfo> = {};
  const atts = rawRow._attachments;
  if (!Array.isArray(atts)) return lookup;

  for (const a of atts) {
    if (!a || typeof a !== 'object') continue;
    const full = String(a.filename || '');
    const parts = full.split('/');
    const short = (parts[parts.length - 1] || '').trim();
    const mime = String(a.mimetype || '').toLowerCase();
    const fullUrl = String(a.download_url || '');
    const previewUrl = String(a.download_medium_url || a.download_small_url || fullUrl);
    if (short && fullUrl) {
      const info: AttachmentInfo = {
        previewUrl,
        fullUrl,
        mimetype: mime,
        filename: short,
      };
      lookup[short] = info;
      lookup[full] = info;
    }
  }
  return lookup;
}

export function findAttachment(
  lookup: Record<string, AttachmentInfo>,
  fname: string
): AttachmentInfo | undefined {
  return lookup[fname] || lookup[fname.replace(/ /g, '_')];
}

export function isImageFilename(val: string): boolean {
  const lower = val.trim().toLowerCase();
  return ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.svg'].some((ext) => lower.endsWith(ext));
}

export function isVideoFilename(val: string): boolean {
  const lower = val.trim().toLowerCase();
  return ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp', '.m4v'].some((ext) =>
    lower.endsWith(ext)
  );
}

export function auditRow(
  row: FlattenedRow,
  dfIndex: number,
  cols: ColsMap,
  regionCols: string[],
  farmerCols: string[],
  masterReview?: MasterReviewRecord
): FarmAuditRecord {
  const subId = String(row._id ?? '');
  const farmId = firstValue(row, cols.farm_id) || `ROW-${dfIndex + 1}`;
  const gpsRaw = firstValue(row, cols.gps);

  let entered = toNumber(firstValue(row, cols.farm_area));
  if (entered === null || entered === 0) {
    entered = toNumber(firstValue(row, cols.plot_area));
  }

  const engineer = firstValue(row, cols.engineer) || 'N/A';
  const missing: string[] = [];
  if (engineer === 'N/A') missing.push('Engineer Name');
  if (entered === null || entered === 0) missing.push('Area');
  if (!gpsRaw) missing.push('GPS Coordinates');

  const poly = gpsRaw ? parsePolygon(gpsRaw) : null;
  const pts = gpsRaw ? parsePoints(gpsRaw) : null;
  const gpsWarnings = poly && pts ? gpsQuality(gpsRaw, pts).join('; ') : '';

  let areaGps: number | null = null;
  let perimM: number | null = null;
  if (poly) {
    const metrics = geomMetricsUtm(poly);
    areaGps = Number(metrics.areaFeddan.toFixed(3));
    perimM = Number(metrics.perimeterM.toFixed(1));
  }

  let areaCheck = '';
  if (areaGps === null || areaGps <= 0) {
    areaCheck = 'Calculation Failed (Invalid/Missing GPS)';
  } else if (entered === null || entered <= 0) {
    areaCheck = 'No entered area to compare';
  } else {
    const ok = Math.abs(entered - areaGps) / areaGps <= AREA_TOL;
    areaCheck = ok ? 'Matched' : 'High Discrepancy';
  }

  const entries = engineerEntries(row, cols.tree_count, cols.tree_type);
  const E = entries.length > 0 ? entries.reduce((sum, [, n]) => sum + n, 0) : null;

  const notes: string[] = [];
  let weightedSpacing = 0;
  for (const [sp, n] of entries) {
    const [spM, known] = spacingForBoundary(sp);
    weightedSpacing += spM * n;
    if (!known) {
      notes.push(`Default boundary spacing ${DEFAULT_BOUNDARY_SPACING_M}m used for '${sp}'`);
    }
  }

  const tot = entries.reduce((sum, [, n]) => sum + n, 0);
  const avgSpacingM = tot > 0 ? weightedSpacing / tot : DEFAULT_BOUNDARY_SPACING_M;
  const forecast =
    perimM !== null && perimM > 0 && avgSpacingM > 0 ? Math.round(perimM / avgSpacingM) : null;

  const byType: Record<string, number> = {};
  for (const [sp, n] of entries) {
    const spName = sp.trim();
    byType[spName] = (byType[spName] || 0) + n;
  }

  const speciesList = Object.entries(byType).sort((a, b) => b[1] - a[1]);
  const topSpecies = speciesList.length > 0 ? speciesList[0][0] : 'Unspecified';
  const treeDetails =
    speciesList.length > 0
      ? speciesList.map(([k, v]) => `${k} (${Math.round(v)})`).join(' | ')
      : 'None';
  const densityNote = Array.from(new Set(notes)).sort().join('; ');

  const initialVerdict = computeVerdict({
    E,
    A: null,
    F: forecast,
  });

  const baseRec: FarmAuditRecord = {
    _dfIndex: dfIndex,
    _submissionId: subId,
    _poly: poly,
    farm_id: farmId,
    farm_name: firstValue(row, cols.farm_name) || '—',
    engineer,
    farmer: firstValue(row, farmerCols.length ? farmerCols : cols.plot_owner) || '—',
    region: firstValue(row, regionCols) || '—',
    submission_time: firstValue(row, cols.time) || '—',
    visit_date: firstValue(row, cols.visit_date) || '—',
    audit_type: firstValue(row, cols.audit_type) || '—',
    season: firstValue(row, cols.season) || '—',
    plot_owner: firstValue(row, cols.plot_owner) || '—',
    owner_phone: firstValue(row, cols.owner_phone) || '—',
    ownership_type: firstValue(row, cols.ownership) || '—',
    soil_type: firstValue(row, cols.soil_type) || '—',
    irrig_type: firstValue(row, cols.irrig_type) || '—',
    irrig_amnt: firstValue(row, cols.irrig_amnt) || '—',
    crop_type: firstValue(row, cols.crop_type) || '—',
    animal_type: firstValue(row, cols.animal_type) || '—',
    animal_count: firstValue(row, cols.animal_count) || '—',
    non_compliance: firstValue(row, cols.non_compliance) || '—',
    corrective_act: firstValue(row, cols.corrective_act) || '—',
    auditor_recom: firstValue(row, cols.auditor_recom) || '—',
    tree_details: treeDetails,
    missing: missing.join(', '),
    gps_warnings: gpsWarnings,
    area_entered: entered,
    area_gps: areaGps,
    perimeter_m: perimM,
    area_check: areaCheck,
    engineer_count: E,
    forecast,
    boundary_spacing_m: Number(avgSpacingM.toFixed(2)),
    species: topSpecies,
    density_note: densityNote,
    ai_count: null,
    ai_low: null,
    ai_high: null,
    ai_interior_trees: null,
    ai_raw: null,
    ai_factor: null,
    veg_pct: null,
    crown_m: null,
    imagery_date: null,
    img_age: null,
    ndvi_recent: null,
    zoom_used: null,
    ai_error: '',
    verdict: initialVerdict,
    confidence: 'Medium',
    flags: [],
    Form_Status: masterReview?.Form_Status ?? STATUS_PENDING,
    Rejected_Items_Count: masterReview?.Rejected_Items_Count ?? 0,
    Consolidated_Comments: masterReview?.Consolidated_Comments ?? '',
    General_Auditor_Note: masterReview?.General_Auditor_Note ?? '',
    Auditor_Name: masterReview?.Auditor_Name ?? '',
    Reviewed_At: masterReview?.Reviewed_At ?? '',
  };

  baseRec.confidence = computeConfidence(baseRec);
  return baseRec;
}

export function runCrossChecks(records: FarmAuditRecord[]): void {
  for (const r of records) {
    r.flags = [];
  }
  const withPolys = records.filter((r) => r._poly !== null);
  for (let i = 0; i < withPolys.length; i++) {
    for (let j = i + 1; j < withPolys.length; j++) {
      const ra = withPolys[i];
      const rb = withPolys[j];
      if (ra.farm_id === rb.farm_id || !ra._poly || !rb._poly) continue;
      const iou = polygonIoU(ra._poly, rb._poly);
      if (iou >= DUP_IOU) {
        const pct = Math.round(iou * 100);
        ra.flags.push(`Polygon overlaps with Farm ${rb.farm_id} (${pct}%)`);
        rb.flags.push(`Polygon overlaps with Farm ${ra.farm_id} (${pct}%)`);
      }
    }
  }
}

export function isVerdictNegative(verdict: string): boolean {
  return (
    verdict.includes('Over Perimeter Capacity') ||
    verdict.includes('Major Boundary Discrepancy') ||
    verdict.includes('Boundary Under-counting') ||
    verdict.includes('Processing Error')
  );
}

export function isAreaCheckNegative(areaCheck: string): boolean {
  return areaCheck.startsWith('High Discrepancy') || areaCheck.startsWith('Calculation Failed');
}
