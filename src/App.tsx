import React, { useCallback, useEffect, useMemo, useState } from 'react';
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
import { displayName } from './lib/workDivider';
import {
  STATUS_PENDING,
  TEAM,
  type AssignmentRecord,
  type FarmAuditRecord,
  type FieldDecisionRow,
  type FlattenedRow,
  type LabelMap,
  type MasterReviewRecord,
  type RawKoboSubmission,
  type TeamMemberKey,
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
  const [satelliteImages, setSatelliteImages] = useState<Record<string, string>>({});
  const [runningSatelliteFarmId, setRunningSatelliteFarmId] = useState<string | null>(null);

  // Toast feedback
  const [toast, setToast] = useState<FlashToast | null>(null);

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
    if (user && !user.mustChangePassword) {
      fetchAllWorkspaceData(false);
    }
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

  // Satellite AI execution
  const handleRunSatelliteAi = useCallback(
    async (farmId: string) => {
      const target = loadedRecords.find((r) => r.farm_id === farmId);
      if (!target || !target._poly) return;

      setRunningSatelliteFarmId(farmId);
      try {
        const res = await analyzeFarmSatellite({
          poly: target._poly,
          species: target.species,
          submissionTime: target.submission_time,
          useNdvi,
        });

        setSatelliteImages((prev) => ({ ...prev, [farmId]: res.annotatedImageUrl }));

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
            vText = applyNdvi(vText, res.ndviRecent, res.vegPct, res.imgAge);

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
              ndvi_recent: res.ndviRecent,
              zoom_used: res.zoomUsed,
              ai_error: '',
              verdict: vText,
            };
            updated.confidence = computeConfidence(updated);
            return updated;
          })
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        pushToast('warn', `Satellite AI warning: ${msg}`);
      } finally {
        setRunningSatelliteFarmId(null);
      }
    },
    [loadedRecords, useNdvi, pushToast]
  );

  // Auto-run satellite AI when opening a farm if enabled and not yet analyzed
  useEffect(() => {
    if (view !== 'review' || !autoSat || !selectedFarmId || runningSatelliteFarmId) return;
    const rec = loadedRecords.find((r) => r.farm_id === selectedFarmId);
    if (!rec || !rec._poly) return;
    const alreadyRan = Boolean(satelliteImages[selectedFarmId]) || rec.ai_count !== null;
    if (!alreadyRan) {
      handleRunSatelliteAi(selectedFarmId);
    }
  }, [
    view,
    autoSat,
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
                runningSatelliteFarmId={runningSatelliteFarmId}
                onRunSatelliteAi={handleRunSatelliteAi}
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
