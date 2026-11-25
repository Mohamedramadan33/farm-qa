import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Menu, AlertCircle } from 'lucide-react';
import { LoginScreen } from './components/LoginScreen';
import { ReviewView } from './components/ReviewView';
import { Sidebar, type ScopeMode } from './components/Sidebar';
import { WorklistView } from './components/WorklistView';
import {
  FARMER_KW,
  REGION_KW,
  applyNdvi,
  auditRow,
  buildFlattenedRows,
  computeConfidence,
  computeVerdict,
  findColsByLabel,
  firstValue,
  mapAllColumns,
  runCrossChecks,
} from './lib/farmQaCore';
import { analyzeFarmSatellite } from './lib/satelliteAnalyzer';
import { DETECTION_CONFIG, setDetectionAuthToken } from './services/satellite/config';
import { displayName } from './lib/workDivider';
import {
  STATUS_PENDING,
  TEAM,
  type AssignmentRecord,
  type FarmAuditRecord,
  type FieldDecisionRow,
  type FlattenedRow,
  type LabelMap,
  type LibraryEngineId,
  type MasterReviewRecord,
  type RawKoboSubmission,
  type SatelliteAnalysisResult,
  type SatelliteCalibrationOptions,
  type SatelliteProgress,
  type TeamMemberKey,
  type TreeDetectionBox,
  type UserSession,
  type WorklistItem,
} from './types/qa';

const LS_PASSWORDS_KEY = 'ebda_farm_qa_passwords_v1';
const LS_SESSION_KEY = 'ebda_farm_qa_session_v1';
const LS_MASTER_REVIEWS_KEY = 'ebda_farm_qa_master_reviews_v1';
const LS_FIELD_REVIEWS_KEY = 'ebda_farm_qa_field_reviews_v1';

interface FlashToast {
  id: number;
  type: 'success' | 'warn' | 'error';
  text: string;
}

export function App() {
  // Team authentication & password state
  const [passwords, setPasswords] = useState<Record<string, string>>(() => {
    try {
      const raw = localStorage.getItem(LS_PASSWORDS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  const [user, setUser] = useState<UserSession | null>(() => {
    try {
      const raw = localStorage.getItem(LS_SESSION_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  // Submissions, labels, assignments, and persisted reviews
  const [rawRows, setRawRows] = useState<RawKoboSubmission[]>([]);
  const [flatRows, setFlatRows] = useState<FlattenedRow[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [labelMap, setLabelMap] = useState<LabelMap>({});
  const [assignments, setAssignments] = useState<Record<string, AssignmentRecord>>({});
  const [doneSubs, setDoneSubs] = useState<Set<string>>(new Set());

  const [masterReviews, setMasterReviews] = useState<Record<string, MasterReviewRecord>>(() => {
    try {
      const raw = localStorage.getItem(LS_MASTER_REVIEWS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  const [fieldReviews, setFieldReviews] = useState<Record<string, FieldDecisionRow[]>>(() => {
    try {
      const raw = localStorage.getItem(LS_FIELD_REVIEWS_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  // UI navigation & loaded audit records
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [refreshingKobo, setRefreshingKobo] = useState(false);
  const [view, setView] = useState<'list' | 'review'>('list');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const [loadedRecords, setLoadedRecords] = useState<FarmAuditRecord[]>([]);
  const [selectedFarmId, setSelectedFarmId] = useState<string>('');
  const [worklistOrder, setWorklistOrder] = useState<Array<[number, string]>>([]);
  const [currentDfIdx, setCurrentDfIdx] = useState<number | null>(null);

  // Satellite options & cached results
  const [autoSat, setAutoSat] = useState(true);
  const [useNdvi, setUseNdvi] = useState(false);
  const [satAuthReady, setSatAuthReady] = useState(false);
  const [satelliteImages, setSatelliteImages] = useState<Record<string, string>>({});
  const [satelliteResults, setSatelliteResults] = useState<Record<string, SatelliteAnalysisResult>>({});
  const [runningSatelliteFarmId, setRunningSatelliteFarmId] = useState<string | null>(null);
  const [satProgress, setSatProgress] = useState<SatelliteProgress | null>(null);
  const satAbortRef = useRef<AbortController | null>(null);
  const [manualEditsDeltaByFarm, setManualEditsDeltaByFarm] = useState<
    Record<string, { added: number; deleted: number; reclassified: number }>
  >({});

  // Toast feedback & Google Maps quota defense (Case A)
  const [toast, setToast] = useState<FlashToast | null>(null);
  const [mapsQuotaExceeded, setMapsQuotaExceeded] = useState(false);

  useEffect(() => {
    const onQuotaExceeded = () => setMapsQuotaExceeded(true);
    window.addEventListener('gmp-quota-exceeded', onQuotaExceeded);
    return () => window.removeEventListener('gmp-quota-exceeded', onQuotaExceeded);
  }, []);

  const pushToast = useCallback((type: FlashToast['type'], text: string) => {
    const id = Date.now();
    setToast({ id, type, text });
    window.setTimeout(() => {
      setToast((cur) => (cur?.id === id ? null : cur));
    }, 5000);
  }, []);

  const fetchAllWorkspaceData = useCallback(
    async (forceRefresh = false) => {
      try {
        const [subResp, asgResp, revResp] = await Promise.all([
          fetch(`/api/kobo/submissions${forceRefresh ? '?refresh=1' : ''}`),
          fetch('/api/team/assignments'),
          fetch('/api/reviews'),
        ]);

        if (!subResp.ok) {
          const errBody = await subResp.json().catch(() => ({}));
          throw new Error(errBody.error || `HTTP ${subResp.status}`);
        }

        const subData = (await subResp.json()) as {
          rows: RawKoboSubmission[];
          labels: LabelMap;
        };

        const built = buildFlattenedRows(subData.rows);
        setRawRows(subData.rows);
        setFlatRows(built.rows);
        setColumns(built.columns);
        setLabelMap(subData.labels || {});

        if (asgResp.ok) {
          const asgData = (await asgResp.json()) as {
            assignments: Record<string, AssignmentRecord>;
          };
          setAssignments(asgData.assignments || {});
        }

        if (revResp.ok) {
          const revData = (await revResp.json()) as {
            master: Record<string, MasterReviewRecord>;
            fields: Record<string, FieldDecisionRow[]>;
          };
          if (revData.master && Object.keys(revData.master).length > 0) {
            setMasterReviews((prev) => {
              const merged = { ...prev, ...revData.master };
              localStorage.setItem(LS_MASTER_REVIEWS_KEY, JSON.stringify(merged));
              return merged;
            });
          }
          if (revData.fields && Object.keys(revData.fields).length > 0) {
            setFieldReviews((prev) => {
              const merged = { ...prev, ...revData.fields };
              localStorage.setItem(LS_FIELD_REVIEWS_KEY, JSON.stringify(merged));
              return merged;
            });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        pushToast('error', `تعذّر تحميل بيانات Kobo: ${msg}`);
      } finally {
        setLoadingInitial(false);
        setRefreshingKobo(false);
      }
    },
    [pushToast]
  );

  useEffect(() => {
    let active = true;
    if (user && !user.mustChangePassword) {
      fetchAllWorkspaceData(false);
      fetch('/api/auth/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userKey: user.key }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data: { ok?: boolean; token?: string } | null) => {
          if (!active) return;
          if (data?.ok && data.token) {
            setDetectionAuthToken(data.token);
          }
          setSatAuthReady(true);
        })
        .catch(() => {
          if (active) setSatAuthReady(true);
        });
    } else {
      setDetectionAuthToken(null);
      setSatAuthReady(false);
    }
    return () => {
      active = false;
    };
  }, [user, fetchAllWorkspaceData]);

  // Derived column mappings
  const colsMap = useMemo(() => mapAllColumns(flatRows, columns), [flatRows, columns]);
  const regionCols = useMemo(
    () => findColsByLabel(columns, labelMap, REGION_KW),
    [columns, labelMap]
  );
  const farmerCols = useMemo(() => {
    const found = findColsByLabel(columns, labelMap, FARMER_KW);
    return found.length > 0 ? found : colsMap.plot_owner || [];
  }, [columns, labelMap, colsMap]);

  // Build Worklist (all submissions without Kobo validation uid and not marked done)
  const allWorklist = useMemo<WorklistItem[]>(() => {
    const list: WorklistItem[] = [];
    rawRows.forEach((raw, idx) => {
      const valStatus = raw._validation_status;
      const alreadyValidated = Boolean(valStatus && typeof valStatus === 'object' && valStatus.uid);
      if (alreadyValidated) return;

      const row = flatRows[idx] || {};
      const subId = String(raw._id ?? '');
      if (doneSubs.has(subId)) return;

      const asgKey = (assignments[subId]?.to || '') as TeamMemberKey | '';
      list.push({
        idx,
        subId,
        farmId: firstValue(row, colsMap.farm_id || []) || '—',
        farmName: firstValue(row, colsMap.farm_name || []) || '—',
        farmer: firstValue(row, farmerCols) || '—',
        region: firstValue(row, regionCols) || '—',
        engineer: firstValue(row, colsMap.engineer || []) || '—',
        submitted: (firstValue(row, colsMap.time || []) || '—').slice(0, 10),
        assignedKey: asgKey,
        assignedName: displayName(asgKey),
      });
    });

    list.sort((a, b) => a.submitted.localeCompare(b.submitted));
    return list;
  }, [rawRows, flatRows, colsMap, farmerCols, regionCols, assignments, doneSubs]);

  // Helper to build audit records for a subset of indices (and compute cross-farm overlaps against all rows)
  const loadIndicesIntoAuditor = useCallback(
    (indices: number[]) => {
      if (indices.length === 0) {
        pushToast('warn', 'لم يتم العثور على استمارات مطابقة لنطاق البحث.');
        return false;
      }

      // Compute audit rows across all submissions so crossChecks catches overlapping polygons even when opening 1 farm!
      const allAudited = flatRows.map((row, idx) => {
        const fid = firstValue(row, colsMap.farm_id || []) || `ROW-${idx + 1}`;
        return auditRow(row, idx, colsMap, regionCols, farmerCols, masterReviews[fid]);
      });
      runCrossChecks(allAudited);

      const indexSet = new Set(indices);
      const picked = allAudited.filter((r) => indexSet.has(r._dfIndex));
      if (picked.length === 0) return false;

      setLoadedRecords(picked);
      setSelectedFarmId(picked[0].farm_id);
      setCurrentDfIdx(picked[0]._dfIndex);
      setView('review');
      return true;
    },
    [flatRows, colsMap, regionCols, farmerCols, masterReviews, pushToast]
  );

  const handleCancelSatelliteAi = useCallback(() => {
    if (satAbortRef.current) {
      satAbortRef.current.abort();
      satAbortRef.current = null;
    }
    setRunningSatelliteFarmId(null);
    setSatProgress(null);
  }, []);

  // Satellite AI execution
  const handleRunSatelliteAi = useCallback(
    async (farmId: string, calibration?: Partial<SatelliteCalibrationOptions>) => {
      const target = loadedRecords.find((r) => r.farm_id === farmId);
      if (!target || !target._poly) return;

      if (satAbortRef.current) {
        satAbortRef.current.abort();
      }
      const controller = new AbortController();
      satAbortRef.current = controller;

      setLoadedRecords((prev) =>
        prev.map((rec) => (rec.farm_id === farmId ? { ...rec, ai_error: '' } : rec))
      );
      setRunningSatelliteFarmId(farmId);
      setSatProgress({ stage: 'tiles', loaded: 0, total: 1, message: 'Starting satellite tile download...' });
      try {
        const prevRes = satelliteResults[farmId];
        const res = await analyzeFarmSatellite({
          farmId: target.farm_id,
          poly: target._poly,
          species: target.species,
          cropType: target.crop_type,
          engineerCount: target.engineer_count,
          forecast: target.forecast,
          submissionTime: target.submission_time,
          useNdvi,
          providerMode: calibration?.providerMode ?? prevRes?.providerId ?? 'google',
          offsetX_m: calibration?.offsetX_m ?? prevRes?.offsetX_m ?? 0,
          offsetY_m: calibration?.offsetY_m ?? prevRes?.offsetY_m ?? 0,
          inwardBufferM: calibration?.inwardBufferM ?? prevRes?.inwardBufferM ?? -1.5,
          highRes: calibration?.highRes ?? calibration?.highAccuracy ?? prevRes?.highResMode ?? false,
          highAccuracy: calibration?.highRes ?? calibration?.highAccuracy ?? prevRes?.highResMode ?? false,
          confThresh:
            calibration?.confThresh ?? prevRes?.confThreshUsed ?? DETECTION_CONFIG.aiModelConfThresh,
          signal: controller.signal,
          onProgress: (p) => setSatProgress(p),
        });

        setSatelliteImages((prev) => ({ ...prev, [farmId]: res.annotatedImageUrl }));
        setSatelliteResults((prev) => ({ ...prev, [farmId]: res }));
        setManualEditsDeltaByFarm((prev) => ({
          ...prev,
          [farmId]: { added: 0, deleted: 0, reclassified: 0 },
        }));

        setLoadedRecords((prev) =>
          prev.map((rec) => {
            if (rec.farm_id !== farmId) return rec;
            const band: [number, number] = [res.aiLow, res.aiHigh];
            let vText = computeVerdict({
              E: rec.engineer_count,
              A: res.aiCount,
              F: rec.forecast,
              band,
              vegPct: res.vegPct,
              imgAge: res.imgAge,
              crownM: res.crownM,
            });
            vText = applyNdvi(vText, res.pseudoNdvi, res.vegPct, res.imgAge);

            const updated: FarmAuditRecord = {
              ...rec,
              ai_count: res.aiCount,
              ai_low: res.aiLow,
              ai_high: res.aiHigh,
              ai_interior_trees: res.aiInteriorTrees,
              ai_raw: res.aiRaw,
              ai_factor: res.aiFactor,
              veg_pct: res.vegPct,
              crown_m: res.crownM,
              imagery_date: res.imageryDate,
              img_age: res.imgAge,
              pseudo_ndvi: res.pseudoNdvi,
              zoom_used: res.zoomUsed,
              ai_error: '',
              verdict: vText,
            };
            updated.confidence = computeConfidence(updated);
            return updated;
          })
        );
      } catch (err) {
        if (controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
          pushToast('warn', 'تم إلغاء تحليل القمر الصناعي.');
          return;
        }
        const msg = err instanceof Error ? err.message : String(err);
        setLoadedRecords((prev) =>
          prev.map((rec) => (rec.farm_id === farmId ? { ...rec, ai_error: msg } : rec))
        );
        pushToast('warn', `Satellite AI: ${msg}`);
      } finally {
        if (satAbortRef.current === controller) {
          satAbortRef.current = null;
          setRunningSatelliteFarmId(null);
          setSatProgress(null);
        }
      }
    },
    [loadedRecords, satelliteResults, useNdvi, pushToast]
  );

  // Live Human-in-the-Loop tree edit handler (add / delete / reclassify / health update)
  const handleUpdateSatelliteBoxes = useCallback(
    (farmId: string, updatedBoxes: TreeDetectionBox[]) => {
      const curSat = satelliteResults[farmId];
      if (!curSat) return;

      const prevById = new Map(curSat.boxes.map((b) => [b.id, b]));
      const nextById = new Map(updatedBoxes.map((b) => [b.id, b]));
      let addedDelta = 0;
      let deletedDelta = 0;
      let reclassDelta = 0;
      for (const b of updatedBoxes) {
        const old = prevById.get(b.id);
        if (!old) addedDelta++;
        else if (old.classId !== b.classId || old.health !== b.health) reclassDelta++;
      }
      for (const b of curSat.boxes) {
        if (!nextById.has(b.id)) deletedDelta++;
      }
      if (addedDelta > 0 || deletedDelta > 0 || reclassDelta > 0) {
        setManualEditsDeltaByFarm((prev) => {
          const cur = prev[farmId] || { added: 0, deleted: 0, reclassified: 0 };
          return {
            ...prev,
            [farmId]: {
              added: cur.added + addedDelta,
              deleted: cur.deleted + deletedDelta,
              reclassified: cur.reclassified + reclassDelta,
            },
          };
        });
      }

      const boundaryBoxes = updatedBoxes.filter((b) => b.onBoundary);
      const interiorBoxes = updatedBoxes.filter((b) => !b.onBoundary);
      const primaryBoxes = curSat.inwardBufferM < -0.5 ? updatedBoxes : boundaryBoxes;
      const factor = curSat.aiFactor || 1.0;
      const aiCount = Math.round(primaryBoxes.length * factor);
      const aiLow = aiCount;
      const aiHigh = aiCount;

      let crownM = curSat.crownM;
      if (primaryBoxes.length > 0) {
        const diams = primaryBoxes
          .map((b) => b.crownDiameterM ?? Math.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin)) * curSat.mpp)
          .sort((a, b) => a - b);
        crownM = Number(diams[Math.floor(diams.length / 2)].toFixed(2));
      } else {
        crownM = null;
      }

      setSatelliteResults((prev) => ({
        ...prev,
        [farmId]: {
          ...curSat,
          boxes: updatedBoxes,
          aiCount,
          aiLow,
          aiHigh,
          aiInteriorTrees: interiorBoxes.length,
          aiRaw: primaryBoxes.length,
          crownM,
        },
      }));

      setLoadedRecords((prev) =>
        prev.map((rec) => {
          if (rec.farm_id !== farmId) return rec;
          let vText = computeVerdict({
            E: rec.engineer_count,
            A: aiCount,
            F: rec.forecast,
            band: [aiLow, aiHigh],
            vegPct: curSat.vegPct,
            imgAge: curSat.imgAge,
            crownM,
          });
          vText = applyNdvi(vText, curSat.pseudoNdvi, curSat.vegPct, curSat.imgAge);
          const updated: FarmAuditRecord = {
            ...rec,
            ai_count: aiCount,
            ai_low: aiLow,
            ai_high: aiHigh,
            ai_interior_trees: interiorBoxes.length,
            ai_raw: primaryBoxes.length,
            crown_m: crownM,
            verdict: vText,
          };
          updated.confidence = computeConfidence(updated);
          return updated;
        })
      );
    },
    [satelliteResults]
  );

  // Adopt a specific detection library's boxes & count from the Multi-Library Comparison table
  const handleAdoptLibraryEngine = useCallback(
    (farmId: string, engineId: LibraryEngineId) => {
      const curSat = satelliteResults[farmId];
      const lib = curSat?.libraryComparison?.libraries.find(
        (l) => l.id === engineId && l.status === 'ran'
      );
      if (!curSat || !lib || !curSat.libraryComparison) return;

      const adoptedBoxes = lib.boxes;
      const boundaryBoxes = adoptedBoxes.filter((b) => b.onBoundary);
      const interiorBoxes = adoptedBoxes.filter((b) => !b.onBoundary);
      const primaryBoxes = curSat.inwardBufferM < -0.5 ? adoptedBoxes : boundaryBoxes;
      const aiCount = primaryBoxes.length;
      const crownM = lib.medianCrownM;

      setSatelliteResults((prev) => ({
        ...prev,
        [farmId]: {
          ...curSat,
          detectorEngine: `${lib.name} (${lib.libraryBadge})`,
          boxes: adoptedBoxes,
          aiCount,
          aiLow: Math.max(0, Math.round(aiCount * 0.9)),
          aiHigh: Math.round(aiCount * 1.1),
          aiInteriorTrees: interiorBoxes.length,
          aiRaw: aiCount,
          crownM,
          libraryComparison: {
            ...curSat.libraryComparison!,
            activeEngineId: engineId,
          },
        },
      }));

      setLoadedRecords((prev) =>
        prev.map((rec) => {
          if (rec.farm_id !== farmId) return rec;
          let vText = computeVerdict({
            E: rec.engineer_count,
            A: aiCount,
            F: rec.forecast,
            band: [Math.max(0, Math.round(aiCount * 0.9)), Math.round(aiCount * 1.1)],
            vegPct: curSat.vegPct,
            imgAge: curSat.imgAge,
            crownM,
          });
          vText = applyNdvi(vText, curSat.pseudoNdvi, curSat.vegPct, curSat.imgAge);
          const updated: FarmAuditRecord = {
            ...rec,
            ai_count: aiCount,
            ai_low: Math.max(0, Math.round(aiCount * 0.9)),
            ai_high: Math.round(aiCount * 1.1),
            ai_interior_trees: interiorBoxes.length,
            ai_raw: aiCount,
            crown_m: crownM,
            verdict: vText,
          };
          updated.confidence = computeConfidence(updated);
          return updated;
        })
      );
      pushToast('success', `تم اعتماد نتيجة "${lib.nameAr}" (${aiCount} شجرة) للمزرعة ${farmId}.`);
    },
    [satelliteResults, pushToast]
  );

  // Auto-run satellite AI when opening a farm if enabled and not yet analyzed/errored
  useEffect(() => {
    if (view !== 'review' || !autoSat || !satAuthReady || !selectedFarmId || runningSatelliteFarmId) {
      return;
    }
    const rec = loadedRecords.find((r) => r.farm_id === selectedFarmId);
    if (!rec || !rec._poly) return;
    const alreadyAttempted =
      Boolean(satelliteImages[selectedFarmId]) || rec.ai_count !== null || Boolean(rec.ai_error);
    if (!alreadyAttempted) {
      handleRunSatelliteAi(selectedFarmId);
    }
  }, [
    view,
    autoSat,
    satAuthReady,
    selectedFarmId,
    loadedRecords,
    satelliteImages,
    runningSatelliteFarmId,
    handleRunSatelliteAi,
  ]);

  // Auth handlers
  const handleLoginSuccess = (session: UserSession) => {
    setUser(session);
    localStorage.setItem(LS_SESSION_KEY, JSON.stringify(session));
  };

  const handlePasswordChange = (key: TeamMemberKey, newPassword: string) => {
    setPasswords((prev) => {
      const next = { ...prev, [key]: newPassword };
      localStorage.setItem(LS_PASSWORDS_KEY, JSON.stringify(next));
      return next;
    });
    if (user && user.key === key) {
      const updatedUser: UserSession = { ...user, mustChangePassword: false };
      setUser(updatedUser);
      localStorage.setItem(LS_SESSION_KEY, JSON.stringify(updatedUser));
    }
  };

  const handleLogout = () => {
    handleCancelSatelliteAi();
    setDetectionAuthToken(null);
    setSatAuthReady(false);
    setUser(null);
    localStorage.removeItem(LS_SESSION_KEY);
    setLoadedRecords([]);
    setView('list');
  };

  // Worklist & Sidebar handlers
  const handleOpenFarmFromWorklist = (dfIndex: number, order: Array<[number, string]>) => {
    setWorklistOrder(order);
    loadIndicesIntoAuditor([dfIndex]);
  };

  const handleGoNextFarm = () => {
    const ids = worklistOrder.map(([i]) => i);
    const curPos = currentDfIdx !== null ? ids.indexOf(currentDfIdx) : -1;
    const start = curPos >= 0 ? curPos + 1 : 0;
    for (let k = start; k < worklistOrder.length; k++) {
      const [idx, subId] = worklistOrder[k];
      if (!doneSubs.has(subId)) {
        loadIndicesIntoAuditor([idx]);
        return;
      }
    }
    pushToast('warn', 'مفيش مزارع تانية في القائمة بعد المزرعة الحالية.');
  };

  const handleLoadScopeForms = (mode: ScopeMode, farmIdInput: string, dateInput: string) => {
    if (mode === 'Specific Farm (Farm ID)') {
      const q = farmIdInput.trim().toLowerCase();
      if (!q) {
        pushToast('warn', 'اكتب كود المزرعة (Farm ID) الأول.');
        return;
      }
      const matched: number[] = [];
      flatRows.forEach((row, idx) => {
        const fid = (firstValue(row, colsMap.farm_id || []) || '').toLowerCase();
        if (fid === q || fid.includes(q)) matched.push(idx);
      });
      loadIndicesIntoAuditor(matched);
    } else if (mode === 'By Submission Date') {
      const d = dateInput.trim();
      const matched: number[] = [];
      flatRows.forEach((row, idx) => {
        const t = firstValue(row, colsMap.time || []) || '';
        if (t.startsWith(d)) matched.push(idx);
      });
      loadIndicesIntoAuditor(matched);
    } else {
      loadIndicesIntoAuditor(flatRows.map((_, idx) => idx));
    }
  };

  const handleReassignSelected = async (
    subIds: string[],
    target: TeamMemberKey,
    farmIds: Record<string, string>
  ) => {
    const resp = await fetch('/api/team/reassign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subIds, target, by: user?.key, farmIds }),
    });
    if (resp.ok) {
      const data = await resp.json();
      setAssignments(data.assignments || {});
      pushToast('success', `تم نقل ${subIds.length} مزرعة إلى ${TEAM[target].name}`);
    }
  };

  const handleRedistributeEqual = async (subIds: string[], farmIds: Record<string, string>) => {
    const resp = await fetch('/api/team/redistribute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subIds, by: user?.key, farmIds }),
    });
    if (resp.ok) {
      const data = (await resp.json()) as {
        counts: Record<TeamMemberKey, number>;
        assignments: Record<string, AssignmentRecord>;
      };
      setAssignments(data.assignments || {});
      const summary = Object.entries(data.counts)
        .map(([k, v]) => `${displayName(k)}: ${v}`)
        .join(' · ');
      pushToast('success', `تم التقسيم بالتساوي: ${summary}`);
    }
  };

  const handleRunDivideWork = async (): Promise<string[]> => {
    const items = allWorklist.map((w) => ({
      subId: w.subId,
      farmCode: w.farmId,
      governorate: w.region,
    }));
    const resp = await fetch('/api/team/divide-work', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items, by: user?.key }),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as {
      logs: string[];
      assignments: Record<string, AssignmentRecord>;
    };
    setAssignments(data.assignments || {});
    pushToast('success', 'تم توزيع المزارع المتبقية بعدالة حسب المحافظات بنجاح.');
    return data.logs || [];
  };

  const handleSaveDecision = async ({
    master,
    fields,
    syncKobo,
  }: {
    master: MasterReviewRecord;
    fields: FieldDecisionRow[];
    syncKobo: boolean;
  }) => {
    // Persist locally & on backend
    setMasterReviews((prev) => {
      const next = { ...prev, [master.farm_id]: master };
      localStorage.setItem(LS_MASTER_REVIEWS_KEY, JSON.stringify(next));
      return next;
    });

    const significantFields = fields.filter(
      (f) =>
        f.Field_Key.startsWith('_AI_') ||
        f.Item_Decision !== 'OK' ||
        f.Suggested_Correction_or_Comment.trim() !== '' ||
        f.Attach_to_Summary
    );
    setFieldReviews((prev) => {
      const next = { ...prev, [master.farm_id]: significantFields };
      localStorage.setItem(LS_FIELD_REVIEWS_KEY, JSON.stringify(next));
      return next;
    });

    setLoadedRecords((prev) =>
      prev.map((r) =>
        r.farm_id === master.farm_id
          ? {
              ...r,
              Form_Status: master.Form_Status,
              Rejected_Items_Count: master.Rejected_Items_Count,
              Consolidated_Comments: master.Consolidated_Comments,
              General_Auditor_Note: master.General_Auditor_Note,
              Auditor_Name: master.Auditor_Name,
              Reviewed_At: master.Reviewed_At,
            }
          : r
      )
    );

    await fetch('/api/reviews/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ master, fields: significantFields }),
    });

    let msg = `Saved! Farm ${master.farm_id} -> ${master.Form_Status}`;

    if (syncKobo && master.submission_id && master.Form_Status !== STATUS_PENDING) {
      const code = master.Form_Status.includes('Approved')
        ? 'validation_status_approved'
        : master.Form_Status.includes('REJECTED')
          ? 'validation_status_not_approved'
          : 'validation_status_on_hold';

      const kResp = await fetch('/api/kobo/validation-status', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ submissionId: master.submission_id, statusCode: code }),
      });
      const kJson = await kResp.json().catch(() => ({ ok: false }));
      if (kJson.ok) {
        msg += ' — synced to Kobo';
        setDoneSubs((prev) => new Set(prev).add(String(master.submission_id)));
      } else {
        pushToast('warn', 'تم الحفظ محلياً، ولكن تعذّرت مزامنة الحالة مع KoboToolbox.');
        return;
      }
    }

    pushToast('success', msg);
  };

  const handleSyncAirtable = async (master: MasterReviewRecord) => {
    if (master.Form_Status === STATUS_PENDING) {
      pushToast('warn', 'اختار Final Status غير Pending قبل الإرسال إلى Airtable.');
      return;
    }
    const resp = await fetch('/api/airtable/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(master),
    });
    const data = (await resp.json().catch(() => ({ ok: false, message: 'فشل الاتصال' }))) as {
      ok: boolean;
      message: string;
    };
    pushToast(data.ok ? 'success' : 'error', data.message);
  };

  if (!user || user.mustChangePassword) {
    return (
      <LoginScreen
        passwords={passwords}
        onLoginSuccess={handleLoginSuccess}
        onPasswordChanged={handlePasswordChange}
        pendingForcedUser={user?.mustChangePassword ? user : null}
      />
    );
  }

  return (
    <div className="min-h-screen h-screen flex bg-slate-50 overflow-hidden">
      <Sidebar
        user={user}
        passwords={passwords}
        onChangePassword={handlePasswordChange}
        onLogout={handleLogout}
        activeView={view}
        hasLoadedForms={loadedRecords.length > 0}
        onNavigateView={setView}
        onRunDivideWork={handleRunDivideWork}
        autoSat={autoSat}
        setAutoSat={setAutoSat}
        useNdvi={useNdvi}
        setUseNdvi={setUseNdvi}
        onRefreshKobo={async () => {
          setRefreshingKobo(true);
          await fetchAllWorkspaceData(true);
          pushToast('success', 'تم تحديث بيانات الاستمارات من Kobo بنجاح.');
        }}
        refreshingKobo={refreshingKobo}
        onLoadScopeForms={handleLoadScopeForms}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* SaaS Contextual Top Bar Contract: Breadcrumb Left — Primary Action Right */}
        <header className="h-14 shrink-0 bg-white border-b border-slate-200 px-6 flex items-center justify-between gap-8">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileSidebarOpen(true)}
              className="lg:hidden p-1.5 -ml-1.5 text-slate-600 hover:text-slate-900 rounded-md cursor-pointer"
              aria-label="Open menu"
            >
              <Menu className="w-5 h-5" />
            </button>
            <nav className="flex items-center gap-2 text-xs font-medium text-slate-500 truncate">
              <span className="text-slate-800 font-semibold">EBDA &amp; Kobo</span>
              <span aria-hidden="true">/</span>
              {view === 'list' ? (
                <span className="text-slate-900">Worklist (قائمة الشغل)</span>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => setView('list')}
                    className="hover:text-slate-900 transition-colors cursor-pointer"
                  >
                    Worklist
                  </button>
                  <span aria-hidden="true">/</span>
                  <span className="font-mono-num text-slate-900 font-semibold">
                    {selectedFarmId || 'Audit'}
                  </span>
                </>
              )}
            </nav>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <span className="hidden sm:inline text-xs text-slate-600 whitespace-nowrap">
              Auditor: <strong className="text-slate-900">{user.name}</strong>
            </span>
          </div>
        </header>

        {/* Google Maps Platform Case A Quota Exceeded Sticky Banner */}
        {mapsQuotaExceeded && (
          <div className="bg-amber-100 border-b border-amber-300 text-amber-950 px-6 py-2.5 text-xs flex items-center justify-between gap-4 shrink-0">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-700 shrink-0" />
              <span>
                Google Maps Platform demo quota was reached or blocked. Live Google Maps tiles are temporarily paused —{' '}
                <a
                  href="https://developers.google.com/maps/ai/ai-studio?utm_campaign=gmp_mcp_codeassist_v1_aistudio#quota_exceeded_errors"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline font-semibold text-amber-900 hover:text-amber-700"
                >
                  Quota &amp; Billing details
                </a>
                .
              </span>
            </div>
            <button
              type="button"
              onClick={() => setMapsQuotaExceeded(false)}
              className="text-amber-900 hover:underline font-medium shrink-0 cursor-pointer"
            >
              إغلاق
            </button>
          </div>
        )}

        {/* Flash Toast Notification */}
        {toast && (
          <div className="px-6 pt-4 shrink-0">
            <div
              className={`px-4 py-2.5 rounded-lg border text-xs font-medium flex items-center justify-between gap-3 ${
                toast.type === 'success'
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-950'
                  : toast.type === 'warn'
                    ? 'bg-amber-50 border-amber-200 text-amber-950'
                    : 'bg-red-50 border-red-200 text-red-950'
              }`}
            >
              <div className="flex items-center gap-2">
                {toast.type === 'success' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-700 shrink-0" />
                )}
                <span>{toast.text}</span>
              </div>
              <button
                type="button"
                onClick={() => setToast(null)}
                className="text-slate-500 hover:text-slate-900 text-xs cursor-pointer"
              >
                إغلاق
              </button>
            </div>
          </div>
        )}

        {/* Main Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-[1360px] mx-auto">
            {loadingInitial ? (
              <div className="space-y-4">
                <div className="h-8 w-72 bg-slate-200/80 rounded-md animate-pulse" />
                <div className="h-16 bg-white border border-slate-200 rounded-xl animate-pulse" />
                <div className="h-80 bg-white border border-slate-200 rounded-xl animate-pulse" />
              </div>
            ) : view === 'list' || loadedRecords.length === 0 ? (
              <WorklistView
                currentUserKey={user.key}
                allWorklist={allWorklist}
                masterReviews={masterReviews}
                regionFound={regionCols.length > 0}
                onOpenFarmByIndex={handleOpenFarmFromWorklist}
                onReassignSelected={handleReassignSelected}
                onRedistributeEqual={handleRedistributeEqual}
              />
            ) : (
              <ReviewView
                records={loadedRecords}
                selectedFarmId={selectedFarmId}
                onSelectFarmId={(fid) => {
                  setSelectedFarmId(fid);
                  const found = loadedRecords.find((r) => r.farm_id === fid);
                  if (found) setCurrentDfIdx(found._dfIndex);
                }}
                rawRows={rawRows}
                flatRows={flatRows}
                labelMap={labelMap}
                auditorName={user.name}
                satelliteImages={satelliteImages}
                satelliteResults={satelliteResults}
                runningSatelliteFarmId={runningSatelliteFarmId}
                satProgress={satProgress}
                manualDelta={manualEditsDeltaByFarm[selectedFarmId]}
                onRunSatelliteAi={handleRunSatelliteAi}
                onCancelSatelliteAi={handleCancelSatelliteAi}
                onUpdateSatelliteBoxes={handleUpdateSatelliteBoxes}
                onAdoptLibraryEngine={handleAdoptLibraryEngine}
                savedFieldDecisions={fieldReviews}
                onSaveDecision={handleSaveDecision}
                onSyncAirtable={handleSyncAirtable}
                onBackToList={() => setView('list')}
                onGoNext={handleGoNextFarm}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
