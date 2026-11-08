import cropPlotImg from '../assets/images/egypt_crop_plot_inspection_1793800794171.jpg';
import boundaryTreesImg from '../assets/images/farm_boundary_casuarina_trees_1793800794444.jpg';
import soilSampleImg from '../assets/images/soil_sample_field_audit_1793800795511.jpg';
import committeeVisitImg from '../assets/images/pgs_committee_site_visit_1793800797448.jpg';

function makeDocSvgDataUrl(titleAr: string, subtitleAr: string, refCode: string): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
    <rect width="800" height="600" fill="#f1f5f9"/>
    <rect x="70" y="40" width="660" height="520" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
    <rect x="70" y="40" width="660" height="72" rx="8" fill="#0f172a"/>
    <text x="400" y="84" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="700" fill="#ffffff">${titleAr}</text>
    <text x="400" y="155" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="600" fill="#334155">${subtitleAr}</text>
    <line x1="120" y1="185" x2="680" y2="185" stroke="#e2e8f0" stroke-width="2"/>
    <rect x="120" y="215" width="560" height="16" rx="4" fill="#e2e8f0"/>
    <rect x="120" y="250" width="480" height="16" rx="4" fill="#e2e8f0"/>
    <rect x="120" y="285" width="520" height="16" rx="4" fill="#e2e8f0"/>
    <rect x="120" y="320" width="410" height="16" rx="4" fill="#e2e8f0"/>
    <rect x="120" y="375" width="560" height="90" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="6 4"/>
    <text x="400" y="426" text-anchor="middle" font-family="monospace" font-size="16" fill="#0f766e">VERIFIED PGS ATTACHMENT · ${refCode}</text>
    <circle cx="200" cy="508" r="28" fill="none" stroke="#0f766e" stroke-width="3"/>
    <text x="200" y="513" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#0f766e">PGS</text>
    <text x="640" y="515" text-anchor="end" font-family="monospace" font-size="14" fill="#64748b">EBDA QA ARCHIVE</text>
  </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

const ASSET_MAP: Record<string, string> = {
  crop_plot: cropPlotImg,
  boundary_trees: boundaryTreesImg,
  soil_sample: soilSampleImg,
  committee_visit: committeeVisitImg,
  doc_tenure: makeDocSvgDataUrl(
    'مستند الحيازة الزراعية',
    'صورة ضوئية معتمدة لبطاقة الحيازة وعقد الملكية',
    'DOC-TENURE-2026'
  ),
  doc_biodiversity: makeDocSvgDataUrl(
    'بيان التنوع الحيوي بالمزرعة',
    'خطة الحفاظ على التنوع البيولوجي ومصدات الرياح',
    'DOC-BIODIV-2026'
  ),
  doc_gmo: makeDocSvgDataUrl(
    'إقرار عدم استخدام تقاوي معدلة وراثياً (GMO)',
    'تعهد المزارع ولجنة المراجعة باستخدام تقاوي عضوية أصيلة',
    'DOC-NON-GMO-2026'
  ),
};

export function resolveMediaUrl(rawUrl: string, maxMb = 15): string {
  if (!rawUrl) return '';
  if (rawUrl.startsWith('asset://')) {
    const key = rawUrl.slice('asset://'.length);
    return ASSET_MAP[key] || cropPlotImg;
  }
  if (rawUrl.startsWith('data:') || rawUrl.startsWith('blob:') || rawUrl.startsWith('/')) {
    return rawUrl;
  }
  if (rawUrl.startsWith('http://') || rawUrl.startsWith('https://')) {
    return `/api/kobo/media?url=${encodeURIComponent(rawUrl)}&maxMb=${maxMb}`;
  }
  return rawUrl;
}
