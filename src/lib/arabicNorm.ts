const DIGIT_MAP: Record<string, string> = {
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
  '٫': '.',
  '٬': ',',
};

const DIGIT_RE = /[٠-٩۰-۹٫٬]/g;
const STRIP_MARKS_RE = /[\u0640\u064B-\u065F\u0670\u200B-\u200F\u202A-\u202E\u2066-\u2069\ufeff]/g;

export function normalizeDigits(input: unknown): string {
  if (input === null || input === undefined) return '';
  return String(input).replace(DIGIT_RE, (ch) => DIGIT_MAP[ch] ?? ch);
}

export function normAr(text: unknown): string {
  if (text === null || text === undefined) return '';
  let s = normalizeDigits(String(text)).normalize('NFKC').toLowerCase();
  s = s.replace(STRIP_MARKS_RE, '');
  s = s.replace(/\u00a0/g, ' ');
  s = s.replace(/[أإآ]/g, 'ا');
  s = s.replace(/ة/g, 'ه').replace(/ى/g, 'ي');
  return s.replace(/\s+/g, ' ').trim();
}

export function present(v: unknown): boolean {
  if (v === null || v === undefined) return false;
  if (typeof v === 'number' && Number.isNaN(v)) return false;
  const s = String(v).trim();
  return s !== '' && s !== '[]' && s !== 'nan' && s !== 'NaN' && s !== 'None' && s !== 'null';
}

export function toNumber(x: unknown): number | null {
  if (!present(x)) return null;
  if (typeof x === 'number') return Number.isFinite(x) ? x : null;
  const s = normalizeDigits(String(x)).replace(/,/g, '').replace(/\s+/g, '');
  const m = s.match(/-?\d+(?:\.\d+)?/);
  if (!m) return null;
  const parsed = parseFloat(m[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function xlSafe(v: unknown): string {
  const s = v === null || v === undefined ? '' : String(v);
  if (s.length > 0 && ['=', '+', '-', '@'].includes(s[0])) {
    return `'${s}`;
  }
  return s;
}

export function fmtVal(v: unknown, decimals?: number): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '—';
    return decimals !== undefined ? v.toFixed(decimals) : String(v);
  }
  const s = String(v).trim();
  return s ? s : '—';
}
