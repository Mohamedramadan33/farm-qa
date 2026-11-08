import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CloudUpload,
  Download,
  Image as ImageIcon,
  Maximize2,
  RefreshCw,
  Save,
  Search,
  Video,
  X,
  XCircle,
} from 'lucide-react';
import { fmtVal, present } from '../lib/arabicNorm';
import { downloadReviewsExcel } from '../lib/excelExport';
import {
  buildAttachmentLookup,
  findAttachment,
  isAreaCheckNegative,
  isImageFilename,
  isMetaColumn,
  isVerdictNegative,
  isVideoFilename,
  resolveLabel,
} from '../lib/farmQaCore';
import { checkRules, type PgsRuleTableRow } from '../lib/pgsRules';
import { resolveMediaUrl } from '../utils/mediaAssets';
import {
  FORM_STATUS_CHOICES,
  ITEM_DECISIONS,
  STATUS_APPROVED,
  STATUS_CONDITIONAL,
  STATUS_PENDING,
  STATUS_REJECTED,
  type FarmAuditRecord,
  type FieldDecisionRow,
  type FlattenedRow,
  type FormStatus,
  type ItemDecision,
  type LabelMap,
  type MasterReviewRecord,
  type MediaQuestionItem,
  type PointLatLon,
  type RawKoboSubmission,
} from '../types/qa';

interface ReviewViewProps {
  records: FarmAuditRecord[];
  selectedFarmId: string;
  onSelectFarmId: (farmId: string) => void;
  rawRows: RawKoboSubmission[];
  flatRows: FlattenedRow[];
  labelMap: LabelMap;
  auditorName: string;
  satelliteImages: Record<string, string>;
  runningSatelliteFarmId: string | null;
  onRunSatelliteAi: (farmId: string) => Promise<void>;
  savedFieldDecisions: Record<string, FieldDecisionRow[]>;
  onSaveDecision: (params: {
    master: MasterReviewRecord;
    fields: FieldDecisionRow[];
    syncKobo: boolean;
  }) => Promise<void>;
  onSyncAirtable: (master: MasterReviewRecord) => Promise<void>;
  onBackToList: () => void;
  onGoNext: () => void;
}

interface EditableFieldState {
  Item_Decision: ItemDecision;
  Suggested_Correction_or_Comment: string;
  Attach_to_Summary: boolean;
}

function withGeneralNote(text: string, note: string): string {
  const n = note.trim();
  return n ? `[General Note: ${n}]\n${text}` : text;
}

export const ReviewView: React.FC<ReviewViewProps> = ({
  records,
  selectedFarmId,
  onSelectFarmId,
  rawRows,
  flatRows,
  labelMap,
  auditorName,
  satelliteImages,
  runningSatelliteFarmId,
  onRunSatelliteAi,
  savedFieldDecisions,
  onSaveDecision,
  onSyncAirtable,
  onBackToList,
  onGoNext,
}) => {
  const [activeTab, setActiveTab] = useState<'form' | 'reviewed'>('form');
  const [onlyReviewedExcel, setOnlyReviewedExcel] = useState(true);

  // Field & media toggles
  const [showEmpty, setShowEmpty] = useState(false);
  const [onlyPhotos, setOnlyPhotos] = useState(false);
  const [loadPhotos, setLoadPhotos] = useState(true);
  const [loadVideos, setLoadVideos] = useState(false);
  const [fieldSearch, setFieldSearch] = useState('');

  // Editable state per farm
  const [editsByKey, setEditsByKey] = useState<Record<string, EditableFieldState>>({});
  const [customMergedComment, setCustomMergedComment] = useState<string | null>(null);
  const [generalNote, setGeneralNote] = useState('');
  const [finalStatus, setFinalStatus] = useState<FormStatus>(STATUS_PENDING);
  const [syncKobo, setSyncKobo] = useState(true);
  const [saving, setSaving] = useState(false);
  const [syncingAirtable, setSyncingAirtable] = useState(false);

  // Image lightbox state
  const [lightbox, setLightbox] = useState<{ url: string; title: string } | null>(null);
  const [brokenUrls, setBrokenUrls] = useState<Set<string>>(new Set());

  const auditRec = useMemo(
    () => records.find((r) => r.farm_id === selectedFarmId) || records[0],
    [records, selectedFarmId]
  );

  // KPI summary metrics across loaded records
  const kpis = useMemo(() => {
    let pending = 0;
    let approved = 0;
    let conditional = 0;
    let rejected = 0;
    for (const r of records) {
      if (r.Form_Status === STATUS_PENDING) pending++;
      else if (r.Form_Status === STATUS_APPROVED) approved++;
      else if (r.Form_Status === STATUS_CONDITIONAL) conditional++;
      else if (r.Form_Status === STATUS_REJECTED) rejected++;
    }
    return { total: records.length, pending, approved, conditional, rejected };
  }, [records]);

  const flatSubmission = useMemo(() => {
    if (!auditRec) return {};
    return flatRows[auditRec._dfIndex] || {};
  }, [auditRec, flatRows]);

  const attLookup = useMemo(() => {
    if (!auditRec) return {};
    const raw = rawRows[auditRec._dfIndex];
    return raw ? buildAttachmentLookup(raw) : {};
  }, [auditRec, rawRows]);

  // Build QA items, PGS rule table items, and media cards
  const { baseQaItems, mediaQuestions, rulesItems } = useMemo(() => {
    if (!auditRec) {
      return {
        baseQaItems: [] as FieldDecisionRow[],
        mediaQuestions: [] as MediaQuestionItem[],
        rulesItems: [] as PgsRuleTableRow[],
      };
    }

    const areaBad = isAreaCheckNegative(auditRec.area_check);
    const verdictBad = isVerdictNegative(auditRec.verdict);

    const qaList: FieldDecisionRow[] = [
      {
        farm_id: auditRec.farm_id,
        Field_Key: '_AI_AREA_CHECK',
        Question_Title: '[Check] Farm Area & Perimeter (Entered vs GPS)',
        Submitted_Answer: `Perimeter: ${fmtVal(auditRec.perimeter_m, 1)}m | Entered: ${fmtVal(
          auditRec.area_entered
        )} | GPS: ${fmtVal(auditRec.area_gps)} (${auditRec.area_check})`,
        Item_Decision: areaBad ? 'Reject' : 'OK',
        Suggested_Correction_or_Comment: areaBad
          ? `Area mismatch: GPS shows ${fmtVal(auditRec.area_gps)} Feddan (Perimeter: ${fmtVal(
              auditRec.perimeter_m,
              1
            )}m)`
          : '',
        Attach_to_Summary: areaBad,
      },
      {
        farm_id: auditRec.farm_id,
        Field_Key: '_AI_TREE_CHECK',
        Question_Title: '[Check] Boundary Tree Count (Engineer vs AI vs Perimeter Forecast)',
        Submitted_Answer: `Eng: ${fmtVal(auditRec.engineer_count)} | Boundary AI: ${fmtVal(
          auditRec.ai_count
        )} | Perimeter Forecast: ${fmtVal(auditRec.forecast)}`,
        Item_Decision: verdictBad ? 'Reject' : 'OK',
        Suggested_Correction_or_Comment: verdictBad ? auditRec.verdict : '',
        Attach_to_Summary: verdictBad,
      },
    ];

    const mediaList: MediaQuestionItem[] = [];
    const rItems: PgsRuleTableRow[] = [];

    for (const [colName, val] of Object.entries(flatSubmission)) {
      if (isMetaColumn(colName)) continue;
      const isPres = present(val);
      if (!isPres && !showEmpty) continue;

      const valStr = isPres ? String(val).trim() : '— (Empty)';
      const parts = valStr.split('/');
      const fname = (parts[parts.length - 1] || '').trim();
      const attInfo = isPres ? findAttachment(attLookup, fname) : undefined;
      const mime = attInfo?.mimetype || '';
      const isPhoto = Boolean(attInfo && (isImageFilename(fname) || mime.includes('image')));
      const isVideo = Boolean(attInfo && (isVideoFilename(fname) || mime.includes('video')));

      const [rawLabel, qLabel] = resolveLabel(colName, labelMap);

      if (isPres) {
        rItems.push({
          Raw_Label: rawLabel,
          Submitted_Answer: valStr,
          Attachment_Link: attInfo?.fullUrl || '',
        });
      }

      if (onlyPhotos && !(isPhoto || isVideo)) {
        continue;
      }

      if (loadPhotos && attInfo && (isPhoto || isVideo)) {
        mediaList.push({
          Field_Key: colName,
          Question_Title: qLabel,
          Filename: fname,
          Type: isPhoto ? 'image' : 'video',
          PreviewUrl: resolveMediaUrl(attInfo.previewUrl, 15),
          FullUrl: resolveMediaUrl(attInfo.fullUrl, isVideo ? 40 : 15),
        });
      }

      qaList.push({
        farm_id: auditRec.farm_id,
        Field_Key: colName,
        Question_Title: isPhoto ? `[Photo] ${qLabel}` : isVideo ? `[Video] ${qLabel}` : qLabel,
        Submitted_Answer: valStr,
        Item_Decision: 'OK',
        Suggested_Correction_or_Comment: '',
        Attach_to_Summary: false,
      });
    }

    return { baseQaItems: qaList, mediaQuestions: mediaList, rulesItems: rItems };
  }, [auditRec, flatSubmission, attLookup, labelMap, showEmpty, onlyPhotos, loadPhotos]);

  // Hydrate per-farm saved decisions & master fields whenever farm or AI check updates
  useEffect(() => {
    if (!auditRec) return;
    const savedList = savedFieldDecisions[auditRec.farm_id] || [];
    const savedMap = new Map<string, FieldDecisionRow>();
    for (const s of savedList) {
      savedMap.set(s.Field_Key, s);
    }

    const nextEdits: Record<string, EditableFieldState> = {};
    for (const item of baseQaItems) {
      const sv = savedMap.get(item.Field_Key);
      if (sv) {
        nextEdits[item.Field_Key] = {
          Item_Decision: sv.Item_Decision,
          Suggested_Correction_or_Comment: sv.Suggested_Correction_or_Comment,
          Attach_to_Summary: sv.Attach_to_Summary,
        };
      } else {
        nextEdits[item.Field_Key] = {
          Item_Decision: item.Item_Decision,
          Suggested_Correction_or_Comment: item.Suggested_Correction_or_Comment,
          Attach_to_Summary: item.Attach_to_Summary,
        };
      }
    }
    setEditsByKey(nextEdits);
    setCustomMergedComment(null);
    setGeneralNote(auditRec.General_Auditor_Note || '');
    setFinalStatus(
      FORM_STATUS_CHOICES.includes(auditRec.Form_Status) ? auditRec.Form_Status : STATUS_PENDING
    );
  }, [auditRec?.farm_id, auditRec?.verdict, auditRec?.area_check, savedFieldDecisions]);

  // Run PGS Business Rules Audit (independent of UI display toggles, matching app.py line 919)
  const allPresentRulesItems = useMemo(() => {
    if (!auditRec) return [] as PgsRuleTableRow[];
    const items: PgsRuleTableRow[] = [];
    for (const [colName, val] of Object.entries(flatSubmission)) {
      if (isMetaColumn(colName) || !present(val)) continue;
      const valStr = String(val).trim();
      const parts = valStr.split('/');
      const fname = (parts[parts.length - 1] || '').trim();
      const attInfo = findAttachment(attLookup, fname);
      const [rawLabel] = resolveLabel(colName, labelMap);
      items.push({
        Raw_Label: rawLabel,
        Submitted_Answer: valStr,
        Attachment_Link: attInfo?.fullUrl || '',
      });
    }
    return items;
  }, [auditRec, flatSubmission, attLookup, labelMap]);

  const pgsIssues = useMemo(() => {
    if (!auditRec) return [];
    const polyLatLon: PointLatLon[] | null = auditRec._poly
      ? auditRec._poly.map(([lon, lat]) => [lat, lon] as PointLatLon)
      : null;
    return checkRules(
      allPresentRulesItems.length > 0 ? allPresentRulesItems : rulesItems,
      auditRec.area_gps,
      polyLatLon
    );
  }, [auditRec, allPresentRulesItems, rulesItems]);

  // Merge baseQaItems with live editsByKey
  const mergedFieldRows = useMemo(() => {
    return baseQaItems.map((row) => {
      const ed = editsByKey[row.Field_Key];
      if (!ed) return row;
      return {
        ...row,
        Item_Decision: ed.Item_Decision,
        Suggested_Correction_or_Comment: ed.Suggested_Correction_or_Comment,
        Attach_to_Summary: ed.Attach_to_Summary,
      };
    });
  }, [baseQaItems, editsByKey]);

  const mediaFieldKeySet = useMemo(
    () => new Set(mediaQuestions.map((m) => m.Field_Key)),
    [mediaQuestions]
  );

  // Table excludes rows that already have a photo/video card (matching app.py line 1060)
  const tableRows = useMemo(() => {
    const nonMedia = mergedFieldRows.filter((r) => !mediaFieldKeySet.has(r.Field_Key));
    if (!fieldSearch.trim()) return nonMedia;
    const q = fieldSearch.trim().toLowerCase();
    return nonMedia.filter(
      (r) =>
        r.Question_Title.toLowerCase().includes(q) ||
        r.Submitted_Answer.toLowerCase().includes(q)
    );
  }, [mergedFieldRows, mediaFieldKeySet, fieldSearch]);

  // Consolidated summary lines
  const summaryData = useMemo(() => {
    let rejectedCount = 0;
    let modifiedCount = 0;
    const lines: string[] = [];

    for (const r of mergedFieldRows) {
      if (r.Item_Decision === 'Reject') rejectedCount++;
      if (r.Item_Decision === 'Modify') modifiedCount++;

      const flagged =
        r.Item_Decision === 'Reject' ||
        r.Item_Decision === 'Modify' ||
        r.Item_Decision === 'Clarify';
      const pick =
        r.Attach_to_Summary || (flagged && r.Suggested_Correction_or_Comment.trim() !== '');
      if (!pick) continue;
      if (!r.Attach_to_Summary && r.Item_Decision === 'OK') continue;

      const qShort = r.Question_Title.split('(')[0].trim();
      const ans =
        r.Submitted_Answer.length > 60
          ? `${r.Submitted_Answer.slice(0, 60)}...`
          : r.Submitted_Answer;
      const comm = r.Suggested_Correction_or_Comment.trim() || 'Flagged by auditor';
      lines.push(`• [${r.Item_Decision}] ${qShort} (Ans: '${ans}') -> ${comm}`);
    }

    const autoText =
      lines.length > 0 ? lines.join('\n') : auditRec?.Consolidated_Comments || '';
    return { lines, rejectedCount, modifiedCount, autoText };
  }, [mergedFieldRows, auditRec?.Consolidated_Comments]);

  const effectiveMergedText =
    customMergedComment !== null ? customMergedComment : summaryData.autoText;

  const updateFieldEdit = (fieldKey: string, patch: Partial<EditableFieldState>) => {
    setEditsByKey((prev) => {
      const current = prev[fieldKey] || {
        Item_Decision: 'OK' as ItemDecision,
        Suggested_Correction_or_Comment: '',
        Attach_to_Summary: false,
      };
      const updated = { ...current, ...patch };
      if (patch.Item_Decision === 'Reject' && patch.Attach_to_Summary === undefined) {
        updated.Attach_to_Summary = true;
      }
      return { ...prev, [fieldKey]: updated };
    });
    setCustomMergedComment(null);
  };

  const handleSaveOrReject = async (forceReject: boolean) => {
    if (!auditRec) return;
    setSaving(true);
    try {
      const chosenStatus: FormStatus = forceReject ? STATUS_REJECTED : finalStatus;
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const nowTs = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(
        now.getHours()
      )}:${pad(now.getMinutes())}`;

      const master: MasterReviewRecord = {
        farm_id: auditRec.farm_id,
        submission_id: auditRec._submissionId,
        Form_Status: chosenStatus,
        Rejected_Items_Count: summaryData.rejectedCount,
        Consolidated_Comments: withGeneralNote(effectiveMergedText, generalNote),
        General_Auditor_Note: generalNote,
        Auditor_Name: auditorName,
        Reviewed_At: nowTs,
      };

      setFinalStatus(chosenStatus);
      await onSaveDecision({
        master,
        fields: mergedFieldRows,
        syncKobo,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleAirtableClick = async () => {
    if (!auditRec) return;
    setSyncingAirtable(true);
    try {
      const now = new Date();
      const pad = (n: number) => String(n).padStart(2, '0');
      const nowTs = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(
        now.getHours()
      )}:${pad(now.getMinutes())}`;

      await onSyncAirtable({
        farm_id: auditRec.farm_id,
        submission_id: auditRec._submissionId,
        Form_Status: finalStatus,
        Rejected_Items_Count: summaryData.rejectedCount,
        Consolidated_Comments: withGeneralNote(effectiveMergedText, generalNote),
        General_Auditor_Note: generalNote,
        Auditor_Name: auditorName,
        Reviewed_At: nowTs,
      });
    } finally {
      setSyncingAirtable(false);
    }
  };

  if (!auditRec) return null;

  const satImg = satelliteImages[auditRec.farm_id];
  const isRunningSat = runningSatelliteFarmId === auditRec.farm_id;
  const hasValidGps = auditRec._poly !== null;
  const reviewedList = records.filter((r) => r.Form_Status !== STATUS_PENDING);

  return (
    <div className="space-y-6">
      {/* Navigation & Export Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={onBackToList}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-white hover:bg-slate-50 border border-slate-300 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>رجوع للقائمة</span>
          </button>

          <button
            type="button"
            onClick={onGoNext}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-white hover:bg-slate-50 border border-slate-300 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
          >
            <span>المزرعة اللي بعدها</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="inline-flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={onlyReviewedExcel}
              onChange={(e) => setOnlyReviewedExcel(e.target.checked)}
              className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
            />
            <span>Excel: reviewed forms only</span>
          </label>
          <button
            type="button"
            onClick={() => downloadReviewsExcel(records, onlyReviewedExcel)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Excel (.xlsx)</span>
          </button>
        </div>
      </div>

      {/* 5-Column KPI Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-5 bg-white border border-slate-200 rounded-xl divide-y sm:divide-y-0 sm:divide-x divide-slate-200">
        <div className="p-4">
          <div className="text-xs text-slate-500">Loaded</div>
          <div className="text-xl font-bold font-mono-num text-slate-900 mt-0.5">{kpis.total}</div>
        </div>
        <div className="p-4">
          <div className="text-xs text-slate-500">Pending</div>
          <div className="text-xl font-bold font-mono-num text-slate-700 mt-0.5">{kpis.pending}</div>
        </div>
        <div className="p-4">
          <div className="text-xs text-slate-500">Approved</div>
          <div className="text-xl font-bold font-mono-num text-emerald-700 mt-0.5">
            {kpis.approved}
          </div>
        </div>
        <div className="p-4">
          <div className="text-xs text-slate-500">Conditional</div>
          <div className="text-xl font-bold font-mono-num text-amber-700 mt-0.5">
            {kpis.conditional}
          </div>
        </div>
        <div className="p-4 col-span-2 sm:col-span-1">
          <div className="text-xs text-slate-500">Rejected</div>
          <div className="text-xl font-bold font-mono-num text-red-700 mt-0.5">{kpis.rejected}</div>
        </div>
      </div>

      {/* Farm Selector & Segmented Tabs */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex-1 max-w-xl">
          <label className="block text-xs font-medium text-slate-600 mb-1">
            Select Farm
          </label>
          <select
            value={auditRec.farm_id}
            onChange={(e) => onSelectFarmId(e.target.value)}
            className="w-full px-3 py-2 text-xs font-medium bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
          >
            {records.map((r) => (
              <option key={r.farm_id} value={r.farm_id}>
                Farm {r.farm_id} | {r.farm_name} | {r.engineer} | {r.Form_Status}
              </option>
            ))}
          </select>
        </div>

        {/* Segmented View Tabs */}
        <div className="inline-flex items-center gap-1 p-1 bg-slate-200/70 rounded-lg self-start md:self-end">
          <button
            type="button"
            onClick={() => setActiveTab('form')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'form'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Form &amp; Auditor
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('reviewed')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'reviewed'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Reviewed Forms ({reviewedList.length})
          </button>
        </div>
      </div>

      {activeTab === 'reviewed' ? (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-200 text-xs text-slate-600">
            <span className="font-mono-num font-semibold text-slate-900">
              {reviewedList.length}
            </span>{' '}
            reviewed of{' '}
            <span className="font-mono-num font-semibold text-slate-900">{records.length}</span>{' '}
            loaded — استخدم زرار Download Excel لتحميل التقرير الكامل.
          </div>

          {reviewedList.length === 0 ? (
            <div className="p-10 text-center text-sm text-slate-500">
              لسه مفيش استمارات اتراجعت في الجلسة الحالية.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-600">
                    <th className="py-2.5 px-4 whitespace-nowrap">Farm ID</th>
                    <th className="py-2.5 px-4 whitespace-nowrap">Engineer</th>
                    <th className="py-2.5 px-4 whitespace-nowrap">Form Status</th>
                    <th className="py-2.5 px-4 whitespace-nowrap text-right">Rejected Items</th>
                    <th className="py-2.5 px-4">Consolidated Comments</th>
                    <th className="py-2.5 px-4 whitespace-nowrap">Auditor</th>
                    <th className="py-2.5 px-4 whitespace-nowrap text-right">Reviewed At</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {reviewedList.map((r) => (
                    <tr
                      key={r.farm_id}
                      onClick={() => {
                        onSelectFarmId(r.farm_id);
                        setActiveTab('form');
                      }}
                      className="hover:bg-slate-50 cursor-pointer"
                    >
                      <td className="py-2.5 px-4 font-mono-num font-semibold text-slate-900 whitespace-nowrap">
                        {r.farm_id}
                      </td>
                      <td className="py-2.5 px-4 text-slate-700 whitespace-nowrap">{r.engineer}</td>
                      <td className="py-2.5 px-4 font-medium text-slate-800 whitespace-nowrap">
                        {r.Form_Status}
                      </td>
                      <td className="py-2.5 px-4 font-mono-num text-right text-slate-700">
                        {r.Rejected_Items_Count}
                      </td>
                      <td className="py-2.5 px-4 text-slate-600 max-w-md truncate">
                        {r.Consolidated_Comments || '—'}
                      </td>
                      <td className="py-2.5 px-4 text-slate-700 whitespace-nowrap">
                        {r.Auditor_Name}
                      </td>
                      <td className="py-2.5 px-4 font-mono-num text-slate-500 text-right whitespace-nowrap">
                        {r.Reviewed_At}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {/* Top Split: Satellite Boundary Verdict + Annotated Satellite Imagery */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between space-y-4">
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
                  <div>
                    <h2 className="text-base font-bold text-slate-900">
                      {auditRec.farm_name}{' '}
                      <span className="font-mono-num text-slate-500 font-normal">
                        ({auditRec.farm_id})
                      </span>
                    </h2>
                    <div className="text-xs text-slate-500 mt-0.5">
                      <span>{auditRec.region}</span>
                      <span className="mx-1.5">·</span>
                      <span>{auditRec.engineer}</span>
                      <span className="mx-1.5">·</span>
                      <span>Confidence: {auditRec.confidence}</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    disabled={isRunningSat || !hasValidGps}
                    onClick={() => onRunSatelliteAi(auditRec.farm_id)}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isRunningSat ? 'animate-spin' : ''}`} />
                    <span>
                      {isRunningSat
                        ? 'Analyzing Satellite...'
                        : satImg
                          ? 'Re-run Satellite AI'
                          : 'Run Satellite AI Check'}
                    </span>
                  </button>
                </div>

                {/* Boundary Verdict Banner */}
                <div
                  className={`p-3.5 rounded-lg border text-xs leading-relaxed ${
                    isVerdictNegative(auditRec.verdict)
                      ? 'bg-red-50/70 border-red-200 text-red-950'
                      : auditRec.verdict.includes('Excellent Boundary Match')
                        ? 'bg-emerald-50/70 border-emerald-200 text-emerald-950'
                        : 'bg-slate-50 border-slate-200 text-slate-800'
                  }`}
                >
                  <div className="font-semibold mb-1">
                    Boundary Verdict ({auditRec.confidence} Confidence)
                  </div>
                  <div>{auditRec.verdict}</div>
                </div>

                {/* Structured Geospatial Telemetry Rows */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 space-y-1">
                    <div className="text-slate-500">محيط المزرعة والمساحة (Perimeter &amp; Area)</div>
                    <div className="font-mono-num text-slate-900 font-semibold">
                      Perimeter: {fmtVal(auditRec.perimeter_m, 1)} m
                    </div>
                    <div className="text-slate-700">
                      Entered:{' '}
                      <strong className="font-mono-num">{fmtVal(auditRec.area_entered)}</strong> vs
                      GPS:{' '}
                      <strong className="font-mono-num">{fmtVal(auditRec.area_gps)}</strong> Feddan (
                      <span
                        className={
                          isAreaCheckNegative(auditRec.area_check)
                            ? 'text-red-700 font-semibold'
                            : 'text-emerald-700 font-semibold'
                        }
                      >
                        {auditRec.area_check}
                      </span>
                      )
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 space-y-1">
                    <div className="text-slate-500">أشجار الحدود (Boundary Trees Audit)</div>
                    <div className="font-mono-num text-slate-900 font-semibold">
                      Eng: {fmtVal(auditRec.engineer_count)} · AI: {fmtVal(auditRec.ai_count)} ·
                      Forecast: {fmtVal(auditRec.forecast)}
                    </div>
                    <div className="text-slate-600 truncate" title={auditRec.tree_details}>
                      {auditRec.tree_details} (at{' '}
                      <span className="font-mono-num">{auditRec.boundary_spacing_m}m</span> spacing)
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 space-y-1 sm:col-span-2 flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <span className="text-slate-500">Boundary Vegetation Cover: </span>
                      <strong className="font-mono-num text-slate-900">
                        {auditRec.veg_pct !== null ? `${auditRec.veg_pct}%` : '—'}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Recent NDVI: </span>
                      <strong className="font-mono-num text-slate-900">
                        {fmtVal(auditRec.ndvi_recent)}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Crown Median: </span>
                      <strong className="font-mono-num text-slate-900">
                        {auditRec.crown_m !== null ? `${auditRec.crown_m}m` : '—'}
                      </strong>
                    </div>
                    <div>
                      <span className="text-slate-500">Imagery Date: </span>
                      <strong className="font-mono-num text-slate-900">
                        {fmtVal(auditRec.imagery_date)}
                      </strong>
                    </div>
                  </div>
                </div>

                {/* Geometry & Cross-Farm Overlap Alerts */}
                {auditRec.gps_warnings && (
                  <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-950 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Geometry Quality: </span>
                      <span>{auditRec.gps_warnings}</span>
                    </div>
                  </div>
                )}

                {auditRec.flags.length > 0 && (
                  <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-900 flex items-start gap-2">
                    <XCircle className="w-4 h-4 text-red-700 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Cross-Farm Overlap Flags: </span>
                      <span>{auditRec.flags.join('; ')}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Satellite Mosaic Viewport */}
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
              {satImg ? (
                <div className="space-y-2.5 h-full flex flex-col">
                  <div className="relative flex-1 min-h-[270px] bg-slate-900 rounded-lg overflow-hidden group">
                    <img
                      src={satImg}
                      alt={`Annotated satellite boundary and detected perimeter trees for farm ${auditRec.farm_id}`}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-contain cursor-zoom-in"
                      onClick={() =>
                        setLightbox({
                          url: satImg,
                          title: `Farm ${auditRec.farm_id} — Boundary & Detected Perimeter Trees (${fmtVal(
                            auditRec.ai_count
                          )})`,
                        })
                      }
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          url: satImg,
                          title: `Farm ${auditRec.farm_id} — Boundary & Detected Perimeter Trees`,
                        })
                      }
                      className="absolute bottom-2.5 right-2.5 inline-flex items-center gap-1 px-2.5 py-1 bg-slate-900/80 hover:bg-slate-900 text-white text-[11px] rounded-md transition-opacity cursor-pointer"
                    >
                      <Maximize2 className="w-3 h-3" />
                      <span>تكبير الصورة</span>
                    </button>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Farm <span className="font-mono-num">{auditRec.farm_id}</span> · Boundary
                    Corridor (Yellow 8m Belt) &amp; Detected Boundary Trees (Red Boxes:{' '}
                    <span className="font-mono-num font-semibold text-slate-700">
                      {fmtVal(auditRec.ai_count)}
                    </span>
                    )
                  </p>
                </div>
              ) : !hasValidGps ? (
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center p-6 bg-amber-50/50 rounded-lg border border-amber-200">
                  <AlertTriangle className="w-6 h-6 text-amber-600 mb-2" />
                  <p className="text-xs font-medium text-amber-950">
                    No valid GPS polygon recorded in Kobo for this farm to fetch satellite imagery.
                  </p>
                </div>
              ) : (
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center p-6 bg-slate-50 rounded-lg border border-slate-200">
                  <RefreshCw
                    className={`w-6 h-6 text-slate-400 mb-2 ${isRunningSat ? 'animate-spin' : ''}`}
                  />
                  <p className="text-xs text-slate-600 max-w-xs">
                    {isRunningSat
                      ? `Fetching satellite imagery, drawing farm boundary & detecting perimeter trees for Farm ${auditRec.farm_id}...`
                      : "Click 'Run Satellite AI Check' to load satellite boundary & tree detections."}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* PGS Business Rules Audit */}
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h3 className="text-sm font-bold text-slate-900 mb-3">
              PGS Business Rules Audit (فحص القواعد الإدارية والزراعية)
            </h3>

            {pgsIssues.length > 0 ? (
              <div className="p-4 rounded-lg bg-red-50/70 border border-red-200 text-red-950 space-y-2">
                <div className="flex items-center gap-2 text-xs font-semibold">
                  <AlertTriangle className="w-4 h-4 text-red-700 shrink-0" />
                  <span>تم اكتشاف تناقضات إدارية في الاستمارة ({pgsIssues.length}):</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-xs pr-2">
                  {pgsIssues.map((iss, idx) => (
                    <li key={idx}>{iss}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-emerald-50/70 border border-emerald-200 text-emerald-950 flex items-center gap-2 text-xs font-medium">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>
                  فحص القواعد: جميع المرفقات والأرقام الأساسية وعينات التربة سليمة ومطابقة للمساحة.
                </span>
              </div>
            )}
          </div>

          {/* Field Filter Checkboxes */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center gap-6">
            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={showEmpty}
                onChange={(e) => setShowEmpty(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>Show empty fields</span>
            </label>

            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={onlyPhotos}
                onChange={(e) => setOnlyPhotos(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>Only photo/video questions</span>
            </label>

            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={loadPhotos}
                onChange={(e) => setLoadPhotos(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>Load photos</span>
            </label>

            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={loadVideos}
                onChange={(e) => setLoadVideos(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>Load videos</span>
            </label>
          </div>

          {/* Interactive Photo & Video Audit Cards */}
          {mediaQuestions.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900">
                  Photos &amp; Videos Audit Cards ({mediaQuestions.length}) — مراجعة المرفقات المصورة
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {mediaQuestions.map((mq) => {
                  const ed = editsByKey[mq.Field_Key] || {
                    Item_Decision: 'OK' as ItemDecision,
                    Suggested_Correction_or_Comment: '',
                    Attach_to_Summary: false,
                  };
                  const isBroken = brokenUrls.has(mq.PreviewUrl);

                  return (
                    <div
                      key={mq.Field_Key}
                      className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col justify-between space-y-3"
                    >
                      <div className="space-y-2.5">
                        <div className="text-xs font-semibold text-slate-900 leading-snug">
                          {mq.Question_Title}
                        </div>

                        {mq.Type === 'image' ? (
                          isBroken ? (
                            <div className="aspect-[4/3] rounded-lg bg-slate-100 border border-slate-200 flex flex-col items-center justify-center p-4 text-center">
                              <ImageIcon className="w-6 h-6 text-slate-400 mb-1.5" />
                              <span className="text-xs text-slate-600 font-medium">
                                {mq.Filename}
                              </span>
                            </div>
                          ) : (
                            <div className="relative aspect-[4/3] rounded-lg bg-slate-100 overflow-hidden border border-slate-200">
                              <img
                                src={mq.PreviewUrl}
                                alt={`${mq.Question_Title} (${mq.Filename})`}
                                referrerPolicy="no-referrer"
                                onError={() =>
                                  setBrokenUrls((prev) => new Set(prev).add(mq.PreviewUrl))
                                }
                                onClick={() =>
                                  setLightbox({
                                    url: mq.FullUrl,
                                    title: `${mq.Question_Title} — ${mq.Filename}`,
                                  })
                                }
                                className="w-full h-full object-cover cursor-zoom-in"
                              />
                            </div>
                          )
                        ) : loadVideos ? (
                          <video
                            src={mq.FullUrl}
                            controls
                            className="w-full aspect-[4/3] rounded-lg bg-slate-900"
                          />
                        ) : (
                          <div className="aspect-[4/3] rounded-lg bg-slate-100 border border-slate-200 flex flex-col items-center justify-center p-4 text-center">
                            <Video className="w-6 h-6 text-slate-400 mb-1.5" />
                            <span className="text-xs text-slate-600">
                              {mq.Filename} — فعّل Load videos لعرض الفيديو
                            </span>
                          </div>
                        )}

                        <div className="text-[11px] font-mono-num text-slate-500 truncate">
                          {mq.Filename}
                        </div>
                      </div>

                      {/* Segmented Decision Control + Comment */}
                      <div className="space-y-2.5 pt-2 border-t border-slate-100">
                        <div className="grid grid-cols-4 gap-1 p-1 bg-slate-100 rounded-lg">
                          {ITEM_DECISIONS.map((dec) => {
                            const active = ed.Item_Decision === dec;
                            const activeStyle =
                              dec === 'Reject'
                                ? 'bg-red-600 text-white'
                                : dec === 'Modify'
                                  ? 'bg-amber-600 text-white'
                                  : dec === 'Clarify'
                                    ? 'bg-slate-800 text-white'
                                    : 'bg-white text-emerald-800 shadow-xs';
                            return (
                              <button
                                key={dec}
                                type="button"
                                onClick={() => updateFieldEdit(mq.Field_Key, { Item_Decision: dec })}
                                className={`py-1 px-1.5 text-[11px] font-medium rounded-md transition-colors whitespace-nowrap cursor-pointer ${
                                  active ? activeStyle : 'text-slate-600 hover:text-slate-900'
                                }`}
                              >
                                {dec}
                              </button>
                            );
                          })}
                        </div>

                        <input
                          type="text"
                          value={ed.Suggested_Correction_or_Comment}
                          onChange={(e) =>
                            updateFieldEdit(mq.Field_Key, {
                              Suggested_Correction_or_Comment: e.target.value,
                            })
                          }
                          placeholder="سبب الرفض / تعليق المراجع..."
                          className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                        />

                        <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={ed.Attach_to_Summary}
                            onChange={(e) =>
                              updateFieldEdit(mq.Field_Key, {
                                Attach_to_Summary: e.target.checked,
                              })
                            }
                            className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
                          />
                          <span>إرفاق بالتقرير النهائي</span>
                        </label>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Field-Level QA Data Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h3 className="text-sm font-bold text-slate-900">
                Field-Level Form QA ({tableRows.length} rows)
              </h3>
              <div className="relative w-full sm:w-80">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="text"
                  value={fieldSearch}
                  onChange={(e) => setFieldSearch(e.target.value)}
                  placeholder="Search (Arabic / English)..."
                  className="w-full pl-9 pr-3 py-1.5 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
                />
              </div>
            </div>

            <div className="overflow-x-auto max-h-[520px] overflow-y-auto">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 z-10 bg-slate-50 border-b border-slate-200">
                  <tr className="text-xs font-semibold text-slate-600">
                    <th className="py-2.5 px-4 w-[32%]">Question</th>
                    <th className="py-2.5 px-4 w-[24%]">Submitted Answer</th>
                    <th className="py-2.5 px-4 w-[13%] whitespace-nowrap">Verdict</th>
                    <th className="py-2.5 px-4 w-[23%]">Comment / Correction</th>
                    <th className="py-2.5 px-4 w-[8%] text-center whitespace-nowrap">Attach?</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {tableRows.map((row) => (
                    <tr
                      key={row.Field_Key}
                      className={
                        row.Item_Decision === 'Reject'
                          ? 'bg-red-50/40'
                          : row.Item_Decision === 'Modify'
                            ? 'bg-amber-50/40'
                            : 'hover:bg-slate-50/80'
                      }
                    >
                      <td className="py-2 px-4 font-medium text-slate-800 align-top">
                        {row.Question_Title}
                      </td>
                      <td className="py-2 px-4 text-slate-700 font-mono-num break-words max-w-xs align-top">
                        {row.Submitted_Answer}
                      </td>
                      <td className="py-2 px-4 align-top">
                        <select
                          value={row.Item_Decision}
                          onChange={(e) =>
                            updateFieldEdit(row.Field_Key, {
                              Item_Decision: e.target.value as ItemDecision,
                            })
                          }
                          className="w-full px-2 py-1 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                        >
                          {ITEM_DECISIONS.map((d) => (
                            <option key={d} value={d}>
                              {d}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-2 px-4 align-top">
                        <input
                          type="text"
                          value={row.Suggested_Correction_or_Comment}
                          onChange={(e) =>
                            updateFieldEdit(row.Field_Key, {
                              Suggested_Correction_or_Comment: e.target.value,
                            })
                          }
                          placeholder="أضف تعليقاً أو تصحيحاً..."
                          className="w-full px-2.5 py-1 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                        />
                      </td>
                      <td className="py-2 px-4 text-center align-top">
                        <input
                          type="checkbox"
                          checked={row.Attach_to_Summary}
                          onChange={(e) =>
                            updateFieldEdit(row.Field_Key, {
                              Attach_to_Summary: e.target.checked,
                            })
                          }
                          className="mt-1 rounded border-slate-300 text-emerald-700 focus:ring-emerald-600 cursor-pointer"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Final Decision & Single Consolidated Comment */}
          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <h3 className="text-base font-bold text-slate-900 mb-4">
              Final Decision &amp; Single Consolidated Comment
            </h3>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-7 space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-600">
                  <span>
                    Merged comment:{' '}
                    <strong className="font-mono-num text-slate-900">
                      {summaryData.lines.length}
                    </strong>{' '}
                    items ·{' '}
                    <strong className="font-mono-num text-red-700">
                      {summaryData.rejectedCount}
                    </strong>{' '}
                    rejected ·{' '}
                    <strong className="font-mono-num text-amber-700">
                      {summaryData.modifiedCount}
                    </strong>{' '}
                    modified
                  </span>
                  {customMergedComment !== null && (
                    <button
                      type="button"
                      onClick={() => setCustomMergedComment(null)}
                      className="text-emerald-700 hover:underline cursor-pointer"
                    >
                      Reset to auto-generated
                    </button>
                  )}
                </div>

                <textarea
                  rows={6}
                  value={effectiveMergedText}
                  onChange={(e) => setCustomMergedComment(e.target.value)}
                  placeholder="ملخص الملاحظات المرفقة والبنود المرفوضة..."
                  className="w-full p-3 text-xs leading-relaxed bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
                />

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    General note / rejection reason
                  </label>
                  <input
                    type="text"
                    value={generalNote}
                    onChange={(e) => setGeneralNote(e.target.value)}
                    placeholder="ملاحظة عامة للمراجع أو سبب الرفض الرئيسي..."
                    className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
                  />
                </div>
              </div>

              <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
                <div className="space-y-2.5">
                  <div className="text-xs font-semibold text-slate-700">Final Status</div>
                  <div className="space-y-1.5">
                    {FORM_STATUS_CHOICES.map((choice) => (
                      <label
                        key={choice}
                        className={`flex items-center gap-2.5 px-3 py-2 rounded-lg border text-xs font-medium cursor-pointer transition-colors ${
                          finalStatus === choice
                            ? 'border-emerald-600 bg-emerald-50/60 text-slate-900'
                            : 'border-slate-200 hover:bg-slate-50 text-slate-700'
                        }`}
                      >
                        <input
                          type="radio"
                          name={`final_status_${auditRec.farm_id}`}
                          checked={finalStatus === choice}
                          onChange={() => setFinalStatus(choice)}
                          className="text-emerald-700 focus:ring-emerald-600"
                        />
                        <span>{choice}</span>
                      </label>
                    ))}
                  </div>

                  <label className="flex items-center gap-2 pt-1 text-xs text-slate-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={syncKobo}
                      onChange={(e) => setSyncKobo(e.target.checked)}
                      className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
                    />
                    <span>Sync status to KoboToolbox (يرفعها من قائمة الانتظار)</span>
                  </label>
                </div>

                <div className="space-y-2.5">
                  <div className="grid grid-cols-2 gap-2.5">
                    <button
                      type="button"
                      disabled={saving}
                      onClick={() => handleSaveOrReject(false)}
                      className="inline-flex items-center justify-center gap-1.5 py-2.5 px-4 bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                    >
                      <Save className="w-3.5 h-3.5" />
                      <span>{saving ? 'Saving...' : 'Save Decision'}</span>
                    </button>

                    <button
                      type="button"
                      disabled={saving}
                      onClick={() => handleSaveOrReject(true)}
                      className="inline-flex items-center justify-center gap-1.5 py-2.5 px-4 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                      <span>Reject Now</span>
                    </button>
                  </div>

                  <button
                    type="button"
                    disabled={syncingAirtable}
                    onClick={handleAirtableClick}
                    className="w-full inline-flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                  >
                    <CloudUpload className="w-3.5 h-3.5" />
                    <span>{syncingAirtable ? 'Sending to Airtable...' : 'Send to Airtable'}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Full-Resolution Image Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 bg-slate-950/80 flex items-center justify-center p-4"
          onClick={() => setLightbox(null)}
        >
          <div
            className="relative max-w-5xl w-full bg-white rounded-xl overflow-hidden border border-slate-700"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 bg-slate-900 text-white">
              <span className="text-xs font-medium truncate">{lightbox.title}</span>
              <button
                type="button"
                onClick={() => setLightbox(null)}
                className="p-1 text-slate-300 hover:text-white rounded-md cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="bg-slate-950 flex items-center justify-center max-h-[82vh] overflow-auto p-2">
              <img
                src={lightbox.url}
                alt={lightbox.title}
                referrerPolicy="no-referrer"
                className="max-h-[78vh] w-auto object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
