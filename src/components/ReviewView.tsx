import React, { useEffect, useMemo, useRef, useState } from 'react';
import JSZip from 'jszip';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CloudUpload,
  Download,
  Edit3,
  Image as ImageIcon,
  Maximize2,
  PackageOpen,
  RefreshCw,
  Save,
  Search,
  Sparkles,
  Trash2,
  Video,
  X,
  XCircle,
} from 'lucide-react';
import { fmtVal, present } from '../lib/arabicNorm';
import { downloadReviewsExcel } from '../lib/excelExport';
import { pointInRingPx } from '../lib/geometry';
import { worldPxToLonLat } from '../lib/satelliteAnalyzer';
import { getDetectionAuthHeaders } from '../services/satellite/config';
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
  type BasemapViewMode,
  type MasterReviewRecord,
  type MediaQuestionItem,
  type PointLatLon,
  type RawKoboSubmission,
  type SatelliteAnalysisResult,
  type SatelliteCalibrationOptions,
  type SatelliteProgress,
  type TreeClassName,
  type TreeDetectionBox,
  type TreeHealthStatus,
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
  satelliteResults: Record<string, SatelliteAnalysisResult>;
  runningSatelliteFarmId: string | null;
  satProgress?: SatelliteProgress | null;
  manualDelta?: { added: number; deleted: number; reclassified: number };
  onRunSatelliteAi: (
    farmId: string,
    calibration?: Partial<SatelliteCalibrationOptions>
  ) => Promise<void>;
  onCancelSatelliteAi?: () => void;
  onUpdateSatelliteBoxes?: (farmId: string, boxes: TreeDetectionBox[]) => void;
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
  satelliteResults,
  runningSatelliteFarmId,
  satProgress,
  manualDelta,
  onRunSatelliteAi,
  onCancelSatelliteAi,
  onUpdateSatelliteBoxes,
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

  // Image lightbox & interactive satellite viewport state
  const [lightbox, setLightbox] = useState<{ url: string; title: string } | null>(null);
  const [brokenUrls, setBrokenUrls] = useState<Set<string>>(new Set());

  const [providerMode, setProviderMode] = useState<BasemapViewMode>('google');
  const [offsetX_m, setOffsetX_m] = useState<number>(0);
  const [offsetY_m, setOffsetY_m] = useState<number>(0);
  const [inwardBufferM, setInwardBufferM] = useState<number>(-1.5);
  const [showCompactIds, setShowCompactIds] = useState<boolean>(false);
  const [highAccuracyMode, setHighAccuracyMode] = useState<boolean>(false);
  const [editTreesMode, setEditTreesMode] = useState<boolean>(false);
  const [exportingYolo, setExportingYolo] = useState<boolean>(false);
  const [queuingFeedback, setQueuingFeedback] = useState<boolean>(false);
  const [queuedTotal, setQueuedTotal] = useState<number | null>(null);
  const [feedbackNotice, setFeedbackNotice] = useState<string | null>(null);
  const [hoveredTree, setHoveredTree] = useState<TreeDetectionBox | null>(null);
  const [pinnedTree, setPinnedTree] = useState<TreeDetectionBox | null>(null);
  const svgOverlayRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    fetch('/api/dataset/feedback/stats')
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { ok?: boolean; queuedTiles?: number } | null) => {
        if (d?.ok && typeof d.queuedTiles === 'number') {
          setQueuedTotal(d.queuedTiles);
        }
      })
      .catch(() => undefined);
  }, []);

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

  // Sync calibration controls & clear active tooltip when switching farms
  useEffect(() => {
    if (!auditRec) return;
    const res = satelliteResults[auditRec.farm_id];
    setOffsetX_m(res?.offsetX_m ?? 0);
    setOffsetY_m(res?.offsetY_m ?? 0);
    setInwardBufferM(res?.inwardBufferM ?? -1.5);
    setEditTreesMode(false);
    setHoveredTree(null);
    setPinnedTree(null);
  }, [auditRec?.farm_id]);

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
  const satRes = satelliteResults[auditRec.farm_id];
  const isRunningSat = runningSatelliteFarmId === auditRec.farm_id;
  const hasValidGps = auditRec._poly !== null;
  const reviewedList = records.filter((r) => r.Form_Status !== STATUS_PENDING);
  const activeTree = hoveredTree || pinnedTree;

  const classLabelAr: Record<TreeClassName, string> = {
    orchard_tree: 'شجرة بستان (Orchard)',
    palm_tree: 'نخيل (Palm)',
    windbreak_tree: 'مصدات رياح (Windbreak)',
  };
  const classStroke: Record<TreeClassName, string> = {
    orchard_tree: '#10b981',
    palm_tree: '#f59e0b',
    windbreak_tree: '#06b6d4',
  };

  const classCounts = (satRes?.boxes || []).reduce(
    (acc, b) => {
      acc[b.className] = (acc[b.className] || 0) + 1;
      return acc;
    },
    { orchard_tree: 0, palm_tree: 0, windbreak_tree: 0 } as Record<TreeClassName, number>
  );

  const healthCounts = (satRes?.boxes || []).reduce(
    (acc, b) => {
      const h = b.health || 'healthy';
      acc[h] = (acc[h] || 0) + 1;
      return acc;
    },
    { healthy: 0, stressed: 0, dead: 0 } as Record<TreeHealthStatus, number>
  );
  const totalBoxesCount = satRes?.boxes.length || 0;
  const healthyPct =
    totalBoxesCount > 0 ? ((healthCounts.healthy / totalBoxesCount) * 100).toFixed(1) : '0.0';
  const stressedPct =
    totalBoxesCount > 0 ? ((healthCounts.stressed / totalBoxesCount) * 100).toFixed(1) : '0.0';
  const deadPct =
    totalBoxesCount > 0 ? ((healthCounts.dead / totalBoxesCount) * 100).toFixed(1) : '0.0';

  const calibrationDirty =
    satRes !== undefined &&
    (Math.abs(offsetX_m - satRes.offsetX_m) >= 0.1 ||
      Math.abs(offsetY_m - satRes.offsetY_m) >= 0.1 ||
      Math.abs(inwardBufferM - satRes.inwardBufferM) >= 0.1);

  const previewDxPx = satRes ? (offsetX_m - satRes.offsetX_m) / Math.max(satRes.mpp, 0.1) : 0;
  const previewDyPx = satRes ? -(offsetY_m - satRes.offsetY_m) / Math.max(satRes.mpp, 0.1) : 0;

  const triggerSatelliteWithCalib = (
    nextMode: BasemapViewMode = providerMode,
    nextHighAcc: boolean = highAccuracyMode
  ) => {
    setProviderMode(nextMode);
    setHoveredTree(null);
    setPinnedTree(null);
    void onRunSatelliteAi(auditRec.farm_id, {
      providerMode: nextMode,
      offsetX_m,
      offsetY_m,
      inwardBufferM,
      highAccuracy: nextHighAcc,
    });
  };

  const handleUpdateSingleTree = (
    treeId: number,
    patch: Partial<Pick<TreeDetectionBox, 'className' | 'classId' | 'health'>>
  ) => {
    if (!satRes || !onUpdateSatelliteBoxes) return;
    const nextBoxes = satRes.boxes.map((b) => {
      if (b.id !== treeId) return b;
      const nextCls = patch.className ?? b.className;
      const nextClsId: 0 | 1 | 2 =
        nextCls === 'palm_tree' ? 1 : nextCls === 'windbreak_tree' ? 2 : 0;
      return {
        ...b,
        ...patch,
        className: nextCls,
        classId: nextClsId,
        source: 'manual' as const,
      };
    });
    onUpdateSatelliteBoxes(auditRec.farm_id, nextBoxes);
    const updatedPinned = nextBoxes.find((b) => b.id === treeId) || null;
    setPinnedTree(updatedPinned);
  };

  const handleDeleteSingleTree = (treeId: number) => {
    if (!satRes || !onUpdateSatelliteBoxes) return;
    const nextBoxes = satRes.boxes.filter((b) => b.id !== treeId);
    onUpdateSatelliteBoxes(auditRec.farm_id, nextBoxes);
    if (pinnedTree?.id === treeId) setPinnedTree(null);
    if (hoveredTree?.id === treeId) setHoveredTree(null);
  };

  const handleCanvasClickEdit = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!satRes) return;
    if (!editTreesMode) {
      setPinnedTree(null);
      return;
    }
    if (!onUpdateSatelliteBoxes || !svgOverlayRef.current) return;
    const rect = svgOverlayRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const cx = Math.max(2, Math.min(satRes.canvasWidth - 2, Math.round(((e.clientX - rect.left) / rect.width) * satRes.canvasWidth)));
    const cy = Math.max(2, Math.min(satRes.canvasHeight - 2, Math.round(((e.clientY - rect.top) / rect.height) * satRes.canvasHeight)));
    const radiusPx = Math.max(5, Math.round(1.6 / Math.max(satRes.mpp, 0.12)));
    const maxId = satRes.boxes.reduce((mx, b) => Math.max(mx, b.id), 0);

    let lat: number | undefined;
    let lon: number | undefined;
    if (satRes.worldOriginX !== undefined && satRes.worldOriginY !== undefined) {
      const [lo, la] = worldPxToLonLat(
        satRes.worldOriginX + cx,
        satRes.worldOriginY + cy,
        satRes.zoomUsed
      );
      lon = lo;
      lat = la;
    }

    const newTree: TreeDetectionBox = {
      id: maxId + 1,
      xmin: Math.max(0, cx - radiusPx),
      ymin: Math.max(0, cy - radiusPx),
      xmax: Math.min(satRes.canvasWidth, cx + radiusPx),
      ymax: Math.min(satRes.canvasHeight, cy + radiusPx),
      cx,
      cy,
      radiusPx,
      score: 1.0,
      classId: 0,
      className: 'orchard_tree',
      health: 'healthy',
      onBoundary: true,
      shadowDrop: 0,
      laplacianVar: 0,
      crownDiameterM: Number((radiusPx * 2 * satRes.mpp).toFixed(2)),
      lat,
      lon,
      source: 'manual',
    };

    const nextBoxes = [...satRes.boxes, newTree];
    onUpdateSatelliteBoxes(auditRec.farm_id, nextBoxes);
    setPinnedTree(newTree);
  };

  interface BuiltYoloTile {
    baseName: string;
    pngBlob: Blob;
    dataUrl: string;
    yoloLines: string[];
    annotations: Array<{ classId: number; cx: number; cy: number; w: number; h: number }>;
  }

  const buildRawYoloTiles = async (): Promise<BuiltYoloTile[]> => {
    if (!satRes) return [];
    const srcUrl = satRes.rawImageUrl || satRes.cleanImageUrl;
    if (!srcUrl) return [];

    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Failed to load raw satellite mosaic'));
      el.src = srcUrl;
    });

    const w = img.naturalWidth || satRes.canvasWidth;
    const h = img.naturalHeight || satRes.canvasHeight;
    const mosaicCanvas = document.createElement('canvas');
    mosaicCanvas.width = w;
    mosaicCanvas.height = h;
    const mCtx = mosaicCanvas.getContext('2d')!;
    mCtx.drawImage(img, 0, 0, w, h);

    // Compute farm polygon mask for tile selection & consistent label clipping (raw tiles remain unmasked)
    const activeRing =
      satRes.clippedRingPx && satRes.clippedRingPx.length >= 3 ? satRes.clippedRingPx : satRes.ringPx;
    const clippedBoxes =
      activeRing && activeRing.length >= 3
        ? satRes.boxes.filter((b) => pointInRingPx(b.cx, b.cy, activeRing))
        : satRes.boxes;

    const tileSize = 512;
    const overlap = 0.2;
    const stride = Math.round(tileSize * (1 - overlap));
    const sW = Math.min(tileSize, w);
    const sH = Math.min(tileSize, h);

    const xs: number[] = [];
    for (let x = 0; x < w; x += stride) {
      const c = Math.max(0, Math.min(x, w - sW));
      if (xs.length === 0 || xs[xs.length - 1] !== c) xs.push(c);
      if (c + sW >= w) break;
    }
    const ys: number[] = [];
    for (let y = 0; y < h; y += stride) {
      const c = Math.max(0, Math.min(y, h - sH));
      if (ys.length === 0 || ys[ys.length - 1] !== c) ys.push(c);
      if (c + sH >= h) break;
    }

    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = tileSize;
    tileCanvas.height = tileSize;
    const tCtx = tileCanvas.getContext('2d')!;

    const out: BuiltYoloTile[] = [];
    let tileCounter = 0;
    for (const ty of ys) {
      for (const tx of xs) {
        // Check polygon overlap inside this tile (skip tiles <5% inside farm boundary)
        if (activeRing && activeRing.length >= 3) {
          let insideHits = 0;
          let sampled = 0;
          for (let sy = 8; sy < sH; sy += 16) {
            for (let sx = 8; sx < sW; sx += 16) {
              sampled++;
              if (pointInRingPx(tx + sx, ty + sy, activeRing)) insideHits++;
            }
          }
          if (sampled > 0 && insideHits / sampled < 0.05) continue;
        }

        tCtx.fillStyle = '#000000';
        tCtx.fillRect(0, 0, tileSize, tileSize);
        tCtx.drawImage(mosaicCanvas, tx, ty, sW, sH, 0, 0, sW, sH);

        tileCounter++;
        const baseName = `farm_${auditRec.farm_id}_${satRes.providerId}_tile_${String(tileCounter).padStart(3, '0')}`;
        const pngBlob = await new Promise<Blob | null>((res) =>
          tileCanvas.toBlob((b) => res(b), 'image/png')
        );
        if (!pngBlob) continue;
        const dataUrl = tileCanvas.toDataURL('image/png');

        const yoloLines: string[] = [];
        const annotations: Array<{ classId: number; cx: number; cy: number; w: number; h: number }> = [];
        for (const box of clippedBoxes) {
          if (box.cx >= tx && box.cx < tx + sW && box.cy >= ty && box.cy < ty + sH) {
            const bx1 = Math.max(0, box.xmin - tx);
            const by1 = Math.max(0, box.ymin - ty);
            const bx2 = Math.min(tileSize, box.xmax - tx);
            const by2 = Math.min(tileSize, box.ymax - ty);
            const bw = Math.max(2, bx2 - bx1);
            const bh = Math.max(2, by2 - by1);
            const ncx = Math.max(0, Math.min(1, (bx1 + bw / 2) / tileSize));
            const ncy = Math.max(0, Math.min(1, (by1 + bh / 2) / tileSize));
            const nw = Math.max(0.004, Math.min(1, bw / tileSize));
            const nh = Math.max(0.004, Math.min(1, bh / tileSize));
            yoloLines.push(
              `${box.classId} ${ncx.toFixed(6)} ${ncy.toFixed(6)} ${nw.toFixed(6)} ${nh.toFixed(6)}`
            );
            annotations.push({ classId: box.classId, cx: ncx, cy: ncy, w: nw, h: nh });
          }
        }
        out.push({ baseName, pngBlob, dataUrl, yoloLines, annotations });
      }
    }
    return out;
  };

  const handleExportYoloZip = async () => {
    if (!satRes || exportingYolo) return;
    setExportingYolo(true);
    try {
      const tiles = await buildRawYoloTiles();
      const zip = new JSZip();
      const imagesFolder = zip.folder('images')!;
      const labelsFolder = zip.folder('labels')!;

      for (const t of tiles) {
        imagesFolder.file(`${t.baseName}.png`, t.pngBlob);
        labelsFolder.file(`${t.baseName}.txt`, t.yoloLines.join('\n'));
      }

      zip.file(
        'data.yaml',
        [
          'path: .',
          'train: images/train',
          'val: images/val',
          'nc: 3',
          "names: ['orchard_tree', 'palm_tree', 'windbreak_tree']",
          '',
        ].join('\n')
      );

      const content = await zip.generateAsync({ type: 'blob' });
      const dlUrl = URL.createObjectURL(content);
      const a = document.createElement('a');
      a.href = dlUrl;
      a.download = `yolo_dataset_${auditRec.farm_id}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(dlUrl);
    } finally {
      setExportingYolo(false);
    }
  };

  const handleQueueActiveLearning = async () => {
    if (!satRes || queuingFeedback) return;
    setQueuingFeedback(true);
    setFeedbackNotice(null);
    try {
      const tiles = await buildRawYoloTiles();
      if (tiles.length === 0) {
        setFeedbackNotice('No overlapping tiles to queue.');
        return;
      }
      const batch = tiles.slice(0, 48).map((t) => ({
        imageBase64: t.dataUrl,
        annotations: t.annotations,
      }));
      const resp = await fetch('/api/dataset/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getDetectionAuthHeaders(),
        },
        body: JSON.stringify({
          farmId: auditRec.farm_id,
          provider: satRes.providerId,
          auditor: auditorName,
          manualDelta: manualDelta ?? { added: 0, deleted: 0, reclassified: 0 },
          tiles: batch,
        }),
      });
      const data = (await resp.json().catch(() => ({}))) as {
        ok?: boolean;
        savedTiles?: number;
        queuedTotal?: number;
        error?: string;
      };
      if (!resp.ok || !data.ok) {
        setFeedbackNotice(data.error || `Failed (${resp.status})`);
        return;
      }
      if (typeof data.queuedTotal === 'number') setQueuedTotal(data.queuedTotal);
      setFeedbackNotice(
        `Queued ${data.savedTiles ?? batch.length} raw tiles + YOLO labels to feedback_queue/`
      );
    } finally {
      setQueuingFeedback(false);
    }
  };

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
            <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between space-y-4">
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

                  <div className="flex items-center gap-2">
                    {isRunningSat && onCancelSatelliteAi && (
                      <button
                        type="button"
                        onClick={onCancelSatelliteAi}
                        className="inline-flex items-center gap-1 px-2.5 py-2 bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                        <span>Cancel</span>
                      </button>
                    )}
                    <button
                      type="button"
                      disabled={isRunningSat || !hasValidGps}
                      onClick={() => triggerSatelliteWithCalib(providerMode)}
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
                      <span className="text-slate-500">Pseudo-NDVI (RGB): </span>
                      <strong className="font-mono-num text-slate-900">
                        {fmtVal(auditRec.pseudo_ndvi)}
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

            {/* Satellite Mosaic Viewport & Geospatial Calibration */}
            <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-4 flex flex-col justify-between space-y-3">
              {/* Basemap Switcher, Mode Badge & High-Accuracy Toggle */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-100">
                <div className="inline-flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg text-[11px] font-medium">
                  {(
                    [
                      { id: 'google', label: 'Google (Z19)' },
                      { id: 'esri', label: 'Esri (Z18-19)' },
                      { id: 'compare', label: 'Side-by-Side Compare' },
                    ] as Array<{ id: BasemapViewMode; label: string }>
                  ).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      disabled={isRunningSat || !hasValidGps}
                      onClick={() => triggerSatelliteWithCalib(opt.id)}
                      className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                        providerMode === opt.id
                          ? 'bg-white text-slate-900 shadow-xs font-semibold'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <div className="flex flex-wrap items-center gap-3 text-[11px]">
                  <label className="inline-flex items-center gap-1.5 text-slate-700 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={highAccuracyMode}
                      onChange={(e) => setHighAccuracyMode(e.target.checked)}
                      className="rounded border-slate-300 text-indigo-700 focus:ring-indigo-600"
                    />
                    <span className="inline-flex items-center gap-1 font-medium text-indigo-900">
                      <Sparkles className="w-3 h-3 text-indigo-600" />
                      High-Accuracy (Pro)
                    </span>
                  </label>

                  <label className="inline-flex items-center gap-1.5 text-slate-700 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={showCompactIds}
                      onChange={(e) => setShowCompactIds(e.target.checked)}
                      className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
                    />
                    <span>Compact IDs (أرقام الأشجار)</span>
                  </label>
                </div>
              </div>

              {satRes ? (
                <div className="space-y-2.5 flex-1 flex flex-col">
                  {/* Detector Mode Status Strip + Edit Trees & YOLO Dataset Export */}
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                        satRes.detectorMode === 'gemini'
                          ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          : satRes.detectorMode === 'yolo'
                            ? 'bg-sky-50 border-sky-200 text-sky-900'
                            : 'bg-amber-50 border-amber-200 text-amber-900'
                      }`}
                    >
                      <span
                        className={`w-2 h-2 rounded-full ${
                          satRes.detectorMode === 'gemini'
                            ? 'bg-emerald-500'
                            : satRes.detectorMode === 'yolo'
                              ? 'bg-sky-500'
                              : 'bg-amber-500'
                        }`}
                      />
                      <span>
                        {satRes.detectorMode === 'gemini'
                          ? 'AI Vision (Gemini Spatial)'
                          : satRes.detectorMode === 'yolo'
                            ? 'YOLO Model'
                            : 'Offline Morphological Fallback'}
                      </span>
                    </span>

                    <div className="flex flex-wrap items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          setEditTreesMode((v) => !v);
                          setHoveredTree(null);
                        }}
                        className={`inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold rounded-md border transition-colors cursor-pointer ${
                          editTreesMode
                            ? 'bg-indigo-700 border-indigo-700 text-white'
                            : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                        }`}
                      >
                        <Edit3 className="w-3 h-3" />
                        <span>{editTreesMode ? 'Editing Trees (Done)' : 'Edit Trees'}</span>
                      </button>

                      <button
                        type="button"
                        disabled={exportingYolo}
                        onClick={handleExportYoloZip}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold rounded-md border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-50 text-slate-800 transition-colors cursor-pointer"
                      >
                        <PackageOpen className="w-3 h-3 text-emerald-700" />
                        <span>
                          {exportingYolo ? 'Exporting Zip...' : 'Export YOLO Zip'}
                        </span>
                      </button>

                      <button
                        type="button"
                        disabled={queuingFeedback}
                        onClick={handleQueueActiveLearning}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold rounded-md border border-sky-300 bg-sky-50 hover:bg-sky-100 disabled:opacity-50 text-sky-900 transition-colors cursor-pointer"
                        title="Send corrected raw tiles & YOLO annotations to dataset/feedback_queue/"
                      >
                        <CloudUpload className="w-3 h-3 text-sky-700" />
                        <span>
                          {queuingFeedback
                            ? 'Queuing...'
                            : `Queue Active Learning${queuedTotal !== null ? ` (${queuedTotal})` : ''}`}
                        </span>
                      </button>
                    </div>
                  </div>

                  {feedbackNotice && (
                    <div className="px-2.5 py-1.5 rounded-md bg-sky-50 border border-sky-200 text-[11px] text-sky-900">
                      {feedbackNotice}
                    </div>
                  )}

                  {satRes.aiVisionNotice && (
                    <div className="px-2.5 py-1.5 rounded-md bg-amber-50 border border-amber-200 text-[11px] text-amber-900">
                      {satRes.aiVisionNotice}
                    </div>
                  )}

                  {editTreesMode && (
                    <div className="px-2.5 py-1.5 rounded-md bg-indigo-50 border border-indigo-200 text-[11px] text-indigo-950">
                      وضع التعديل اليدوي: اضغط كليك شمال في أي مكان لإضافة شجرة، أو اضغط على علامة لتعديل نوعها/حالتها، أو كليك يمين للحذف السريع.
                    </div>
                  )}

                  {/* Crisp RTL HTML/CSS Arabic & Numeric Telemetry HUD (zero canvas glyph boxes) */}
                  <div
                    dir="rtl"
                    className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-lg bg-slate-900 text-slate-100 text-[11px] border border-slate-800"
                  >
                    <div className="flex flex-wrap items-center gap-2.5">
                      <span className="font-semibold text-emerald-300">
                        إجمالي الأشجار:{' '}
                        <strong className="font-mono-num text-white">{totalBoxesCount}</strong>
                      </span>
                      <span className="text-slate-600">|</span>
                      <span className="text-slate-300">
                        سليمة:{' '}
                        <strong className="font-mono-num text-emerald-400">
                          {healthCounts.healthy}
                        </strong>{' '}
                        <span className="font-mono-num text-slate-400">({healthyPct}%)</span>
                      </span>
                      <span className="text-slate-300">
                        مجهدة:{' '}
                        <strong className="font-mono-num text-amber-400">
                          {healthCounts.stressed}
                        </strong>{' '}
                        <span className="font-mono-num text-slate-400">({stressedPct}%)</span>
                      </span>
                      <span className="text-slate-300">
                        تالفة:{' '}
                        <strong className="font-mono-num text-red-400">
                          {healthCounts.dead}
                        </strong>{' '}
                        <span className="font-mono-num text-slate-400">({deadPct}%)</span>
                      </span>
                    </div>

                    <div dir="ltr" className="flex items-center gap-2 font-mono-num text-[10px] text-sky-300">
                      <span>
                        {satRes.provider} · Z{satRes.zoomUsed} ({satRes.mpp} m/px)
                      </span>
                    </div>
                  </div>

                  <div
                    className={`grid gap-2.5 flex-1 ${
                      providerMode === 'compare' && satRes.compareImageUrl
                        ? 'grid-cols-1 sm:grid-cols-2'
                        : 'grid-cols-1'
                    }`}
                  >
                    {/* Primary Interactive Viewport (Clean Image + Interactive SVG Canopy Circles) */}
                    <div className="relative min-h-[260px] bg-slate-950 rounded-lg overflow-hidden border border-slate-800">
                      <div
                        className="relative w-full h-full flex items-center justify-center"
                        style={{ aspectRatio: `${satRes.canvasWidth} / ${satRes.canvasHeight}` }}
                      >
                        <img
                          src={satRes.cleanImageUrl}
                          alt={`Satellite canopy detection for farm ${auditRec.farm_id}`}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-contain select-none"
                        />
                        <svg
                          ref={svgOverlayRef}
                          viewBox={`0 0 ${satRes.canvasWidth} ${satRes.canvasHeight}`}
                          onClick={handleCanvasClickEdit}
                          className={`absolute inset-0 w-full h-full ${
                            editTreesMode ? 'cursor-crosshair' : ''
                          }`}
                        >
                          {/* Live preview outline when dragging X/Y calibration sliders */}
                          {calibrationDirty && satRes.ringPx.length >= 3 && (
                            <polygon
                              points={satRes.ringPx
                                .map(([x, y]) => `${x + previewDxPx},${y + previewDyPx}`)
                                .join(' ')}
                              fill="none"
                              stroke="#f97316"
                              strokeWidth={2}
                              strokeDasharray="6 4"
                            />
                          )}

                          {/* Clean Canopy Circles with Interactive Hover/Click/Right-Click Delete */}
                          {satRes.boxes.map((tree) => {
                            const stroke =
                              tree.health === 'dead'
                                ? '#ef4444'
                                : tree.health === 'stressed'
                                  ? '#eab308'
                                  : classStroke[tree.className] || '#10b981';
                            const isSelected = activeTree?.id === tree.id;
                            const r = Math.max(4, tree.radiusPx);
                            return (
                              <g
                                key={tree.id}
                                className="cursor-pointer"
                                onMouseEnter={() => {
                                  if (!editTreesMode) setHoveredTree(tree);
                                }}
                                onMouseLeave={() => {
                                  if (!editTreesMode) setHoveredTree(null);
                                }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setPinnedTree((cur) => (cur?.id === tree.id ? null : tree));
                                }}
                                onContextMenu={(e) => {
                                  if (editTreesMode) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    handleDeleteSingleTree(tree.id);
                                  }
                                }}
                              >
                                <circle cx={tree.cx} cy={tree.cy} r={Math.max(10, r + 4)} fill="transparent" />
                                <circle
                                  cx={tree.cx}
                                  cy={tree.cy}
                                  r={isSelected ? r + 2 : r}
                                  fill={isSelected ? `${stroke}44` : `${stroke}22`}
                                  stroke={isSelected ? '#ffffff' : stroke}
                                  strokeWidth={isSelected ? 2.4 : 1.7}
                                  strokeDasharray={tree.source === 'manual' ? '3 2' : undefined}
                                />
                                <circle cx={tree.cx} cy={tree.cy} r={2} fill={stroke} />
                                {showCompactIds && (
                                  <text
                                    x={tree.cx}
                                    y={tree.cy - r - 2}
                                    textAnchor="middle"
                                    fill="#ffffff"
                                    stroke="#0f172a"
                                    strokeWidth={2.5}
                                    paintOrder="stroke"
                                    fontSize="9"
                                    fontWeight="700"
                                  >
                                    {tree.id}
                                  </text>
                                )}
                              </g>
                            );
                          })}
                        </svg>

                        {/* Interactive Hover / Click Inspector & Edit Popover */}
                        {activeTree && (
                          <div
                            onClick={(e) => e.stopPropagation()}
                            className={`absolute z-20 px-2.5 py-2 rounded-lg bg-slate-900/95 border border-slate-700 text-white text-[11px] shadow-lg w-[240px] ${
                              pinnedTree ? 'pointer-events-auto' : 'pointer-events-none'
                            }`}
                            style={{
                              left: `${Math.min(62, Math.max(4, (activeTree.cx / satRes.canvasWidth) * 100))}%`,
                              top: `${Math.min(58, Math.max(4, (activeTree.cy / satRes.canvasHeight) * 100))}%`,
                            }}
                          >
                            <div className="flex items-center justify-between gap-2 font-semibold">
                              <span className="font-mono-num text-emerald-300">
                                Tree #{activeTree.id} {activeTree.source === 'manual' ? '(Manual)' : ''}
                              </span>
                              <span className="font-mono-num text-sky-300">
                                {Math.round(activeTree.score * 100)}% conf
                              </span>
                            </div>
                            <div className="text-slate-200 mt-0.5">
                              {classLabelAr[activeTree.className]}
                            </div>
                            <div className="text-[10px] font-mono-num text-slate-300 mt-0.5">
                              Shadow Drop: {Math.round(activeTree.shadowDrop * 100)}% · Health:{' '}
                              {activeTree.health || 'healthy'}
                            </div>
                            {activeTree.lat !== undefined && activeTree.lon !== undefined && (
                              <div className="text-[10px] font-mono-num text-slate-400 mt-0.5">
                                {activeTree.lat.toFixed(6)}, {activeTree.lon.toFixed(6)}
                              </div>
                            )}

                            {pinnedTree && pinnedTree.id === activeTree.id && onUpdateSatelliteBoxes && (
                              <div className="mt-2 pt-2 border-t border-slate-700 space-y-1.5">
                                <div className="grid grid-cols-3 gap-1 text-[10px]">
                                  {(
                                    [
                                      { id: 'orchard_tree', lbl: 'بستان' },
                                      { id: 'palm_tree', lbl: 'نخيل' },
                                      { id: 'windbreak_tree', lbl: 'مصدات' },
                                    ] as Array<{ id: TreeClassName; lbl: string }>
                                  ).map((cOpt) => (
                                    <button
                                      key={cOpt.id}
                                      type="button"
                                      onClick={() =>
                                        handleUpdateSingleTree(activeTree.id, { className: cOpt.id })
                                      }
                                      className={`px-1 py-0.5 rounded border cursor-pointer ${
                                        activeTree.className === cOpt.id
                                          ? 'bg-emerald-600 border-emerald-500 text-white'
                                          : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white'
                                      }`}
                                    >
                                      {cOpt.lbl}
                                    </button>
                                  ))}
                                </div>
                                <div className="grid grid-cols-3 gap-1 text-[10px]">
                                  {(
                                    [
                                      { id: 'healthy', lbl: 'سليمة' },
                                      { id: 'stressed', lbl: 'مجهدة' },
                                      { id: 'dead', lbl: 'تالفة' },
                                    ] as Array<{ id: TreeHealthStatus; lbl: string }>
                                  ).map((hOpt) => (
                                    <button
                                      key={hOpt.id}
                                      type="button"
                                      onClick={() =>
                                        handleUpdateSingleTree(activeTree.id, { health: hOpt.id })
                                      }
                                      className={`px-1 py-0.5 rounded border cursor-pointer ${
                                        (activeTree.health || 'healthy') === hOpt.id
                                          ? 'bg-indigo-600 border-indigo-500 text-white'
                                          : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white'
                                      }`}
                                    >
                                      {hOpt.lbl}
                                    </button>
                                  ))}
                                </div>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteSingleTree(activeTree.id)}
                                  className="w-full inline-flex items-center justify-center gap-1 py-1 rounded bg-red-600/90 hover:bg-red-600 text-white text-[10px] font-semibold cursor-pointer"
                                >
                                  <Trash2 className="w-3 h-3" />
                                  <span>حذف الشجرة (Delete)</span>
                                </button>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setLightbox({
                            url: satRes.annotatedImageUrl,
                            title: `Farm ${auditRec.farm_id} — ${satRes.provider} Z${satRes.zoomUsed} (${satRes.boxes.length} trees)`,
                          });
                        }}
                        className="absolute bottom-2 right-2 inline-flex items-center gap-1 px-2 py-1 bg-slate-900/85 hover:bg-slate-900 text-white text-[11px] rounded-md cursor-pointer"
                      >
                        <Maximize2 className="w-3 h-3" />
                        <span>تكبير</span>
                      </button>
                    </div>

                    {/* Side-by-Side Comparison Panel (Esri / Alternate Basemap) */}
                    {providerMode === 'compare' && satRes.compareImageUrl && (
                      <div className="relative min-h-[260px] bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
                        <img
                          src={satRes.compareImageUrl}
                          alt={`Comparison imagery (${satRes.compareProvider})`}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-contain cursor-zoom-in"
                          onClick={() =>
                            setLightbox({
                              url: satRes.compareImageUrl!,
                              title: `Farm ${auditRec.farm_id} — ${satRes.compareProvider} Z${satRes.compareZoomUsed}`,
                            })
                          }
                        />
                        <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-900/85 text-[10px] font-mono-num text-amber-300">
                          {satRes.compareProvider} · Z{satRes.compareZoomUsed}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* X/Y Meter Offset Calibration (-10m..+10m) & Inward Boundary Buffer (-1.5m default) */}
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2 text-[11px]">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                      <label className="block">
                        <div className="flex justify-between text-slate-600 mb-0.5">
                          <span>Offset X (E/W)</span>
                          <span className="font-mono-num font-semibold text-slate-900">
                            {offsetX_m > 0 ? `+${offsetX_m}` : offsetX_m}m
                          </span>
                        </div>
                        <input
                          type="range"
                          min={-10}
                          max={10}
                          step={0.5}
                          value={offsetX_m}
                          onChange={(e) => setOffsetX_m(Number(e.target.value))}
                          className="w-full accent-emerald-700"
                        />
                      </label>

                      <label className="block">
                        <div className="flex justify-between text-slate-600 mb-0.5">
                          <span>Offset Y (N/S)</span>
                          <span className="font-mono-num font-semibold text-slate-900">
                            {offsetY_m > 0 ? `+${offsetY_m}` : offsetY_m}m
                          </span>
                        </div>
                        <input
                          type="range"
                          min={-10}
                          max={10}
                          step={0.5}
                          value={offsetY_m}
                          onChange={(e) => setOffsetY_m(Number(e.target.value))}
                          className="w-full accent-emerald-700"
                        />
                      </label>

                      <label className="block">
                        <div className="flex justify-between text-slate-600 mb-0.5">
                          <span>Boundary Buffer</span>
                          <span className="font-mono-num font-semibold text-slate-900">
                            {inwardBufferM > 0 ? `+${inwardBufferM}` : inwardBufferM}m
                          </span>
                        </div>
                        <input
                          type="range"
                          min={-5}
                          max={3}
                          step={0.5}
                          value={inwardBufferM}
                          onChange={(e) => setInwardBufferM(Number(e.target.value))}
                          className="w-full accent-amber-600"
                        />
                      </label>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                      <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-600">
                        <span className="inline-flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-emerald-500" />
                          Orchard: <strong className="font-mono-num">{classCounts.orchard_tree}</strong>
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-amber-500" />
                          Palm: <strong className="font-mono-num">{classCounts.palm_tree}</strong>
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-cyan-500" />
                          Windbreak:{' '}
                          <strong className="font-mono-num">{classCounts.windbreak_tree}</strong>
                        </span>
                      </div>

                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          disabled={isRunningSat}
                          onClick={() => {
                            setOffsetX_m(0);
                            setOffsetY_m(0);
                            setInwardBufferM(-1.5);
                            void onRunSatelliteAi(auditRec.farm_id, {
                              providerMode,
                              offsetX_m: 0,
                              offsetY_m: 0,
                              inwardBufferM: -1.5,
                            });
                          }}
                          className="px-2 py-1 text-[11px] text-slate-600 hover:text-slate-900 border border-slate-300 bg-white rounded cursor-pointer"
                        >
                          Reset
                        </button>
                        <button
                          type="button"
                          disabled={isRunningSat}
                          onClick={() => triggerSatelliteWithCalib(providerMode)}
                          className="px-2.5 py-1 text-[11px] font-semibold text-white bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 rounded cursor-pointer"
                        >
                          {calibrationDirty ? 'Apply Calibration' : 'Re-Detect'}
                        </button>
                      </div>
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-500 flex flex-wrap items-center justify-between gap-2">
                    <span>
                      Engine: <strong className="text-slate-700">{satRes.detectorEngine}</strong>
                    </span>
                    <span className="font-mono-num">
                      Verified: <strong>{satRes.boxes.length}</strong> (Counted:{' '}
                      <strong>{fmtVal(auditRec.ai_count)}</strong>)
                    </span>
                  </p>
                </div>
              ) : !hasValidGps ? (
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center p-6 bg-amber-50/50 rounded-lg border border-amber-200">
                  <AlertTriangle className="w-6 h-6 text-amber-600 mb-2" />
                  <p className="text-xs font-medium text-amber-950">
                    No valid GPS polygon recorded in Kobo for this farm to fetch satellite imagery.
                  </p>
                </div>
              ) : auditRec.ai_error && !isRunningSat ? (
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center p-6 bg-red-50/70 rounded-lg border border-red-200 space-y-3">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                  <div className="space-y-1">
                    <p className="text-xs font-semibold text-red-950">تعذّر جلب صورة القمر الصناعي</p>
                    <p className="text-[11px] text-red-800 max-w-xs">{auditRec.ai_error}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => triggerSatelliteWithCalib(providerMode)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-red-700 hover:bg-red-800 text-white text-xs font-medium rounded-md transition-colors cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>إعادة المحاولة</span>
                  </button>
                </div>
              ) : (
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center p-6 bg-slate-50 rounded-lg border border-slate-200 space-y-3">
                  <RefreshCw
                    className={`w-6 h-6 text-slate-400 ${isRunningSat ? 'animate-spin' : ''}`}
                  />
                  <p className="text-xs text-slate-600 max-w-xs">
                    {isRunningSat
                      ? satProgress?.message ||
                        `Fetching ${providerMode === 'esri' ? 'Esri Z18-19' : 'Google Z19-20'} imagery & running SAHI detection for Farm ${auditRec.farm_id}...`
                      : "Click 'Run Satellite AI Check' to load satellite boundary & tree detections."}
                  </p>
                  {isRunningSat && satProgress && satProgress.total > 0 && (
                    <div className="w-full max-w-xs space-y-1.5">
                      <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-600 transition-all duration-200"
                          style={{
                            width: `${Math.min(
                              100,
                              Math.round((satProgress.loaded / Math.max(1, satProgress.total)) * 100)
                            )}%`,
                          }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[10px] font-mono-num text-slate-500">
                        <span className="uppercase">{satProgress.stage}</span>
                        <span>
                          {satProgress.loaded}/{satProgress.total}
                        </span>
                      </div>
                    </div>
                  )}
                  {isRunningSat && onCancelSatelliteAi && (
                    <button
                      type="button"
                      onClick={onCancelSatelliteAi}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-[11px] font-medium rounded-md cursor-pointer"
                    >
                      <X className="w-3 h-3" />
                      <span>Cancel Analysis</span>
                    </button>
                  )}
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
