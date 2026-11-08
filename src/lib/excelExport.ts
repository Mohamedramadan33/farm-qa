import * as XLSX from 'xlsx';
import { xlSafe } from './arabicNorm';
import { STATUS_PENDING, type FarmAuditRecord } from '../types/qa';

export function downloadReviewsExcel(records: FarmAuditRecord[], onlyReviewed: boolean): void {
  const filtered = onlyReviewed
    ? records.filter((r) => r.Form_Status !== STATUS_PENDING)
    : records;

  const rows = filtered.map((r) => ({
    'Farm ID': r.farm_id,
    'Farm Name': r.farm_name || '',
    Engineer: r.engineer,
    Status: r.Form_Status,
    'Rejected Items': r.Rejected_Items_Count,
    Comment: xlSafe(r.Consolidated_Comments),
    Auditor: r.Auditor_Name,
    'Reviewed At': r.Reviewed_At,
  }));

  const worksheet = XLSX.utils.json_to_sheet(
    rows.length > 0
      ? rows
      : [
          {
            'Farm ID': '',
            'Farm Name': '',
            Engineer: '',
            Status: '',
            'Rejected Items': 0,
            Comment: '',
            Auditor: '',
            'Reviewed At': '',
          },
        ]
  );

  worksheet['!cols'] = [
    { wch: 14 },
    { wch: 24 },
    { wch: 20 },
    { wch: 32 },
    { wch: 14 },
    { wch: 90 },
    { wch: 18 },
    { wch: 20 },
  ];

  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, 'Reviews');

  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(
    now.getHours()
  )}${pad(now.getMinutes())}`;

  XLSX.writeFile(workbook, `farm_reviews_${stamp}.xlsx`);
}
