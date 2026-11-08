import type { PointLatLon } from '../types/qa';

export interface PgsRuleTableRow {
  Raw_Label: string;
  Submitted_Answer: string;
  Attachment_Link: string;
}

const AR_DIGITS_MAP: Record<string, string> = {
  '٠': '0',
  '١': '1',
  '٢': '2',
  '٣': '3',
  '٤': '4',
  '٥': '5',
  '٦': '6',
  '٧': '7',
  '٨': '8',
  '٩': '9',
  '۰': '0',
  '۱': '1',
  '۲': '2',
  '۳': '3',
  '۴': '4',
  '۵': '5',
  '۶': '6',
  '۷': '7',
  '۸': '8',
  '۹': '9',
};

const STRIP_RE = /[\u0640\u064B-\u065F\u200B-\u200F\u202A-\u202E\u2066-\u2069\ufeff]/g;

export function normPgs(input: unknown): string {
  if (input === null || input === undefined) return '';
  let s = String(input).replace(/[٠-٩۰-۹]/g, (c) => AR_DIGITS_MAP[c] ?? c);
  s = s.replace(STRIP_RE, '');
  s = s.replace(/\u00a0/g, ' ');
  s = s.replace(/[أإآ]/g, 'ا');
  s = s.replace(/ى/g, 'ي').replace(/ة/g, 'ه');
  return s.trim().split(/\s+/).filter(Boolean).join(' ');
}

function L(row: PgsRuleTableRow): string {
  return normPgs(row.Raw_Label);
}

function V(row: PgsRuleTableRow): string {
  return normPgs(row.Submitted_Answer);
}

function K(row: PgsRuleTableRow): string {
  return String(row.Attachment_Link || '').trim();
}

function toFloat(s: unknown): number | null {
  const clean = normPgs(s).replace(/٫/g, '.').replace(/,/g, '.');
  const m = clean.match(/-?\d+(?:\.\d+)?/);
  if (!m) return null;
  const num = parseFloat(m[0]);
  return Number.isFinite(num) ? num : null;
}

function firstRow(
  table: PgsRuleTableRow[],
  text: string,
  exact = false
): PgsRuleTableRow | undefined {
  const t = normPgs(text);
  return table.find((row) => {
    const l = L(row);
    return exact ? l === t : l.includes(t);
  });
}

function hasLink(table: PgsRuleTableRow[], text: string, exact = false, soft = false): boolean {
  if (soft) return true;
  const t = normPgs(text);
  return table.some((row) => {
    const l = L(row);
    const matches = exact ? l === t : l.includes(t);
    return matches && Boolean(K(row) || V(row));
  });
}

export function isValidEgyptPhone(phoneStr: string): boolean {
  let clean = Array.from(normPgs(phoneStr))
    .filter((c) => c >= '0' && c <= '9')
    .join('');
  if (clean.startsWith('0020')) {
    clean = '0' + clean.slice(4);
  } else if (clean.startsWith('20') && (clean.length === 11 || clean.length === 12)) {
    clean = '0' + clean.slice(2);
  }
  if (
    clean.length === 10 &&
    ['10', '11', '12', '15'].some((prefix) => clean.startsWith(prefix))
  ) {
    clean = '0' + clean;
  }
  if (
    clean.length === 11 &&
    ['010', '011', '012', '015'].some((prefix) => clean.startsWith(prefix))
  ) {
    return true;
  }
  const landlinePrefixes = [
    '02',
    '03',
    '040',
    '045',
    '047',
    '048',
    '050',
    '055',
    '057',
    '062',
    '064',
    '065',
    '066',
    '068',
    '069',
    '082',
    '084',
    '086',
    '088',
    '092',
    '093',
    '095',
    '096',
    '097',
  ];
  return (
    (clean.length === 9 || clean.length === 10) &&
    landlinePrefixes.some((prefix) => clean.startsWith(prefix))
  );
}

export function parseKoboDate(dateStr: string): Date | null {
  const s = String(dateStr || '').trim();
  if (!s) return null;

  // YYYY-MM-DD
  const isoMatch = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (isoMatch) {
    const d = new Date(Number(isoMatch[1]), Number(isoMatch[2]) - 1, Number(isoMatch[3]));
    return Number.isNaN(d.getTime()) ? null : d;
  }

  // DD/MM/YYYY
  const dmyMatch = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (dmyMatch) {
    const p1 = Number(dmyMatch[1]);
    const p2 = Number(dmyMatch[2]);
    const yr = Number(dmyMatch[3]);
    const day = p1 > 12 ? p1 : p1;
    const month = p1 > 12 ? p2 : p2;
    const d = new Date(yr, month - 1, day);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const parsed = new Date(s);
  if (!Number.isNaN(parsed.getTime())) {
    return new Date(parsed.getFullYear(), parsed.getMonth(), parsed.getDate());
  }
  return null;
}

export function pointInPolygonLatLon(lat: number, lng: number, poly: PointLatLon[]): boolean {
  let inside = false;
  let j = poly.length - 1;
  for (let i = 0; i < poly.length; i++) {
    const yi = poly[i][0];
    const xi = poly[i][1];
    const yj = poly[j][0];
    const xj = poly[j][1];
    if (yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

export function metersToPolygonEdge(lat: number, lng: number, poly: PointLatLon[]): number {
  const mLat = 111132.92;
  const mLng = 111412.84 * Math.cos((lat * Math.PI) / 180);
  let best = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const ax = (a[1] - lng) * mLng;
    const ay = (a[0] - lat) * mLat;
    const bx = (b[1] - lng) * mLng;
    const by = (b[0] - lat) * mLat;
    const dx = bx - ax;
    const dy = by - ay;
    const seg = dx * dx + dy * dy;
    const t = seg === 0 ? 0 : Math.max(0, Math.min(1, ((-ax) * dx + (-ay) * dy) / seg));
    const cx = ax + t * dx;
    const cy = ay + t * dy;
    best = Math.min(best, Math.hypot(cx, cy));
  }
  return best;
}

export function soilPointsFromTable(table: PgsRuleTableRow[]): PointLatLon[] {
  const target = normPgs('نقاط اخذ عينة للتربة');
  const pts: PointLatLon[] = [];
  for (const row of table) {
    if (L(row).includes(target)) {
      const nums = String(row.Submitted_Answer || '').match(/-?\d+\.\d+/g);
      if (nums && nums.length >= 2) {
        pts.push([parseFloat(nums[0]), parseFloat(nums[1])]);
      }
    }
  }
  return pts;
}

export function parseAreaFeddan(raw: unknown): number | null {
  const s = normPgs(raw).replace(/٫/g, '.').replace(/,/g, '.');
  const units: Record<string, number> = {
    [normPgs('فدان')]: 1.0,
    [normPgs('قيراط')]: 1 / 24,
    [normPgs('سهم')]: 1 / 576,
  };
  let total = 0;
  let found = false;

  if (s.includes(normPgs('فدانين'))) {
    total = 2.0;
    found = true;
  }

  const unitRegex = /(\d+(?:\.\d+)?)\s*(فدان|قيراط|سهم)/g;
  let match: RegExpExecArray | null;
  while ((match = unitRegex.exec(s)) !== null) {
    const factor = units[normPgs(match[2])] ?? 1.0;
    total += parseFloat(match[1]) * factor;
    found = true;
  }

  if (found) return total;
  if (s.includes(normPgs('فدان'))) return 1.0;
  return toFloat(s);
}

const SOIL_MIN_SAMPLES = 4;
const SOIL_EDGE_TOLERANCE_M = 5;

function fmtG(n: number): string {
  return Number(n.toFixed(2)).toString();
}

export function checkRules(
  table: PgsRuleTableRow[],
  calcArea?: number | null,
  farmPolygon?: PointLatLon[] | null
): string[] {
  const brokenRules: string[] = [];
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const n = table.length;
  const linksOk = table.some((r) => Boolean(K(r)));

  let totalFarmArea = 0;
  const areaRow = firstRow(table, 'إجمالي المساحة بالفدان');
  if (areaRow) {
    totalFarmArea = toFloat(areaRow.Submitted_Answer) ?? 0;
  }

  const calcOk = calcArea !== null && calcArea !== undefined && Number.isFinite(calcArea) && calcArea > 0;
  if (calcOk && totalFarmArea > 0) {
    if (Math.abs(calcArea - totalFarmArea) / totalFarmArea > 0.1) {
      brokenRules.push(
        `المزرعة المرسومة ${fmtG(calcArea)} فدان لكن الاستمارة مكتوب فيها ${fmtG(totalFarmArea)} فدان`
      );
    }
  }

  const soils = soilPointsFromTable(table);
  if (soils.length < SOIL_MIN_SAMPLES) {
    brokenRules.push(`عينات التربة ناقصة (${soils.length} من ${SOIL_MIN_SAMPLES})`);
  }

  const poly: PointLatLon[] = (farmPolygon || []).filter(
    (p): p is PointLatLon => Array.isArray(p) && p.length >= 2
  );
  if (poly.length >= 3 && soils.length > 0) {
    const outside: number[] = [];
    soils.forEach(([la, ln], idx) => {
      if (
        !pointInPolygonLatLon(la, ln, poly) &&
        metersToPolygonEdge(la, ln, poly) > SOIL_EDGE_TOLERANCE_M
      ) {
        outside.push(idx + 1);
      }
    });
    const insideCount = soils.length - outside.length;
    if (outside.length > 0 && insideCount < SOIL_MIN_SAMPLES) {
      const nums = outside.join('، ');
      brokenRules.push(
        outside.length === 1
          ? `عينة التربة رقم ${nums} برا حدود المزرعة`
          : `عينات التربة رقم ${nums} برا حدود المزرعة`
      );
    }
  }

  const visitRow = firstRow(table, 'تاريخ الزيارة');
  if (visitRow) {
    const vDate = parseKoboDate(visitRow.Submitted_Answer || '');
    if (vDate && vDate.getTime() > today.getTime()) {
      brokenRules.push('تاريخ الزيارة في المستقبل');
    }
  }

  const phoneRow = firstRow(table, 'رقم تليفون ممثل');
  if (phoneRow) {
    const phoneVal = String(phoneRow.Submitted_Answer || '').trim();
    if (phoneVal && !isValidEgyptPhone(phoneVal)) {
      brokenRules.push('رقم تليفون ممثل المزرعة غلط');
    }
  }

  const auditorPhotoTarget = normPgs('صورة لعضو لجنة المراجعة');
  const auditorPhotosCount = table.filter(
    (row) => L(row).includes(auditorPhotoTarget) && Boolean(K(row) || V(row))
  ).length;
  if (linksOk && auditorPhotosCount < 3) {
    brokenRules.push(`صور لجنة المراجعة ناقصة (${auditorPhotosCount} من 3)`);
  }

  let ownershipType = '';
  const ownRow =
    firstRow(table, 'مملوكة لمالك واحد') ||
    firstRow(table, 'لمجموعة مزارعين') ||
    firstRow(table, 'نوع الملكية');
  if (ownRow) {
    ownershipType = V(ownRow);
  }

  if (ownershipType.includes(normPgs('مجموعة مزارعين'))) {
    const farmerNames: string[] = [];
    const basinAreas: number[] = [];
    const farmerPhones: Array<[string, string]> = [];
    let lastOwner = '';

    for (const row of table) {
      const lbl = L(row);
      const val = V(row);
      if (lbl.includes(normPgs('اسم صاحب الحوض')) && val) {
        farmerNames.push(val);
        lastOwner = String(row.Submitted_Answer || '').trim();
      } else if (lbl.includes(normPgs('مساحة الحوض بالفدان')) && val) {
        const f = toFloat(val);
        if (f !== null) basinAreas.push(f);
      } else if (lbl === normPgs('التليفون') && val) {
        farmerPhones.push([lastOwner, String(row.Submitted_Answer || '').trim()]);
      }
    }

    if (farmerNames.length === 1) {
      brokenRules.push('قائمة المزارعين فيها مزارع واحد بس');
    }
    for (const [owner, ph] of farmerPhones) {
      if (!isValidEgyptPhone(ph)) {
        brokenRules.push(
          owner ? `رقم تليفون ${owner} غلط` : 'فيه رقم تليفون غلط في قائمة المزارعين'
        );
      }
    }
    const summedBasinArea = basinAreas.reduce((a, b) => a + b, 0);
    if (Math.abs(summedBasinArea - totalFarmArea) > 0.001) {
      brokenRules.push(
        `مساحات المزارعين (${fmtG(summedBasinArea)} فدان) مش زي مساحة المزرعة (${fmtG(totalFarmArea)} فدان)`
      );
    }
  }

  if (!hasLink(table, 'ارفق صورة الحيازة')) {
    brokenRules.push('الحيازة مش مرفقة');
  }
  if (!hasLink(table, 'الرجاء ارفاق بيان التنوع الحيوي')) {
    brokenRules.push('بيان التنوع الحيوي مش مرفق');
  }
  if (!hasLink(table, 'الرجاء ارفاق بيان GMO')) {
    brokenRules.push('بيان GMO مش مرفق');
  }

  const CROP_NAME = normPgs('اسم المحصول');
  const BOUNDARIES = new Set([CROP_NAME, normPgs('رقم الحوض'), normPgs('نوع المحصول')]);
  const CUR = normPgs('الموسم الحالي');
  const PREV = normPgs('الموسم السابق');
  const PHOTO_LABELS = new Set([normPgs('ارفق الصورة'), normPgs('ارفق صورة')]);

  const currentCropAreas: number[] = [];
  const previousCropAreas: number[] = [];
  let hasCurrentCropPhoto = true;
  let hasPreviousCropData = false;

  for (let i = 0; i < n; i++) {
    if (L(table[i]) !== CROP_NAME) continue;
    const cropName = V(table[i]);
    let end = n;
    let marker: 'cur' | 'prev' | null = null;

    for (let j = i + 1; j < n; j++) {
      const lj = L(table[j]);
      if (BOUNDARIES.has(lj)) {
        end = j;
        break;
      }
      if (lj.includes(CUR)) {
        marker = 'cur';
        end = j + 1;
        break;
      }
      if (lj.includes(PREV)) {
        marker = 'prev';
        end = j + 1;
        break;
      }
    }

    const group = table.slice(i, end);
    if (marker === null) {
      const labels = group.map((g) => L(g));
      marker = labels.includes(normPgs('الانتاجية المتوقعه'))
        ? 'cur'
        : labels.includes(normPgs('الانتاجية'))
          ? 'prev'
          : 'cur';
    }

    const areaRowInGroup = group.find((g) => L(g) === normPgs('المساحة'));
    const area = areaRowInGroup ? parseAreaFeddan(areaRowInGroup.Submitted_Answer) : null;

    if (marker === 'cur') {
      if (area !== null) currentCropAreas.push(area);
      let photoOk = group.some((g) => PHOTO_LABELS.has(L(g)) && Boolean(K(g) || V(g)));
      let k = end;
      while (!photoOk && k < n && k < end + 3 && !BOUNDARIES.has(L(table[k]))) {
        if (PHOTO_LABELS.has(L(table[k]))) {
          photoOk = Boolean(K(table[k]) || V(table[k]));
          break;
        }
        k++;
      }
      if (linksOk && !photoOk) {
        hasCurrentCropPhoto = false;
      }
    } else {
      if (cropName) hasPreviousCropData = true;
      if (area !== null) previousCropAreas.push(area);
    }
  }

  if (!hasCurrentCropPhoto) {
    brokenRules.push('صورة المحصول الحالي مش مرفقة');
  }

  const sumCurrentArea = currentCropAreas.reduce((a, b) => a + b, 0);
  if (Math.abs(sumCurrentArea - totalFarmArea) > 0.001) {
    brokenRules.push(
      `المساحة المزروعة هذا الموسم ${fmtG(totalFarmArea)} فدان حسب الاستمارة، لكن بيانات المحاصيل مجموعها ${fmtG(sumCurrentArea)} فدان`
    );
  }

  if (!hasPreviousCropData) {
    brokenRules.push('بيانات المحاصيل السابقة ناقصة');
  } else {
    const sumPrevArea = previousCropAreas.reduce((a, b) => a + b, 0);
    if (Math.abs(sumPrevArea - totalFarmArea) > 0.001) {
      brokenRules.push(
        `المساحة المزروعة الموسم السابق ${fmtG(totalFarmArea)} فدان، لكن بيانات المحاصيل المدخلة ${fmtG(sumPrevArea)} فدان`
      );
    }
  }

  if (!hasLink(table, 'ارفاق صورة الدفتر', false, !linksOk)) {
    brokenRules.push('دفتر المزرعة مش مرفق');
  }

  return brokenRules;
}
