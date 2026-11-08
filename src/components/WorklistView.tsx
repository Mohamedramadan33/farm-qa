import React, { useMemo, useState } from 'react';
import {
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Play,
  Scale,
  Search,
} from 'lucide-react';
import { displayName } from '../lib/workDivider';
import {
  STATUS_PENDING,
  TEAM,
  TEAM_KEYS,
  type MasterReviewRecord,
  type TeamMemberKey,
  type WorklistItem,
} from '../types/qa';

interface WorklistViewProps {
  currentUserKey: TeamMemberKey;
  allWorklist: WorklistItem[];
  masterReviews: Record<string, MasterReviewRecord>;
  regionFound: boolean;
  onOpenFarmByIndex: (dfIndex: number, currentFilteredOrder: Array<[number, string]>) => void;
  onReassignSelected: (
    subIds: string[],
    target: TeamMemberKey,
    farmIds: Record<string, string>
  ) => Promise<void>;
  onRedistributeEqual: (subIds: string[], farmIds: Record<string, string>) => Promise<void>;
}

export const WorklistView: React.FC<WorklistViewProps> = ({
  currentUserKey,
  allWorklist,
  masterReviews,
  regionFound,
  onOpenFarmByIndex,
  onReassignSelected,
  onRedistributeEqual,
}) => {
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState<'me' | 'all' | TeamMemberKey>('all');
  const [selectedRegion, setSelectedRegion] = useState('الكل');
  const [selectedEngineer, setSelectedEngineer] = useState('الكل');
  const [hideReviewed, setHideReviewed] = useState(true);
  const [selectedSubIds, setSelectedSubIds] = useState<Set<string>>(new Set());
  const [reassignOpen, setReassignOpen] = useState(false);
  const [moveTarget, setMoveTarget] = useState<TeamMemberKey>('mohamed');
  const [busyAction, setBusyAction] = useState(false);

  // Per-member counts prior to filters
  const memberCounts = useMemo(() => {
    const counts: Record<TeamMemberKey, number> = {
      mohamed: 0,
      mansy: 0,
      haidy: 0,
      islam: 0,
    };
    for (const item of allWorklist) {
      if (item.assignedKey && item.assignedKey in counts) {
        counts[item.assignedKey]++;
      }
    }
    return counts;
  }, [allWorklist]);

  const regions = useMemo(() => {
    const set = new Set<string>();
    for (const item of allWorklist) {
      if (item.region && item.region !== '—') set.add(item.region);
    }
    return Array.from(set).sort();
  }, [allWorklist]);

  const engineers = useMemo(() => {
    const set = new Set<string>();
    for (const item of allWorklist) {
      if (item.engineer && item.engineer !== '—') set.add(item.engineer);
    }
    return Array.from(set).sort();
  }, [allWorklist]);

  const reviewedLocalSubs = useMemo(() => {
    const set = new Set<string>();
    for (const r of Object.values(masterReviews)) {
      if (r.Form_Status && r.Form_Status !== STATUS_PENDING && r.submission_id) {
        set.add(String(r.submission_id));
      }
    }
    return set;
  }, [masterReviews]);

  const filteredList = useMemo(() => {
    return allWorklist.filter((item) => {
      if (hideReviewed && reviewedLocalSubs.has(item.subId)) {
        return false;
      }
      if (scope === 'me' && item.assignedKey !== currentUserKey) {
        return false;
      }
      if (scope !== 'all' && scope !== 'me' && item.assignedKey !== scope) {
        return false;
      }
      if (selectedRegion !== 'الكل' && item.region !== selectedRegion) {
        return false;
      }
      if (selectedEngineer !== 'الكل' && item.engineer !== selectedEngineer) {
        return false;
      }
      if (query.trim()) {
        const q = query.trim().toLowerCase();
        const blob = `${item.farmId} ${item.farmName} ${item.farmer} ${item.region}`.toLowerCase();
        if (!blob.includes(q)) return false;
      }
      return true;
    });
  }, [
    allWorklist,
    hideReviewed,
    reviewedLocalSubs,
    scope,
    currentUserKey,
    selectedRegion,
    selectedEngineer,
    query,
  ]);

  const currentOrder = useMemo(
    () => filteredList.map((w) => [w.idx, w.subId] as [number, string]),
    [filteredList]
  );

  const visibleSelected = useMemo(
    () => filteredList.filter((w) => selectedSubIds.has(w.subId)),
    [filteredList, selectedSubIds]
  );

  const allVisibleChecked =
    filteredList.length > 0 && visibleSelected.length === filteredList.length;

  const toggleAllVisible = () => {
    if (allVisibleChecked) {
      setSelectedSubIds(new Set());
    } else {
      setSelectedSubIds(new Set(filteredList.map((w) => w.subId)));
    }
  };

  const toggleRow = (subId: string) => {
    setSelectedSubIds((prev) => {
      const next = new Set(prev);
      if (next.has(subId)) next.delete(subId);
      else next.add(subId);
      return next;
    });
  };

  const handleMoveSelected = async () => {
    if (!visibleSelected.length) return;
    setBusyAction(true);
    try {
      const farmIds: Record<string, string> = {};
      for (const w of visibleSelected) {
        farmIds[w.subId] = w.farmId;
      }
      await onReassignSelected(
        visibleSelected.map((w) => w.subId),
        moveTarget,
        farmIds
      );
      setSelectedSubIds(new Set());
    } finally {
      setBusyAction(false);
    }
  };

  const handleRedistributeDisplayed = async () => {
    if (!filteredList.length) return;
    setBusyAction(true);
    try {
      const farmIds: Record<string, string> = {};
      for (const w of filteredList) {
        farmIds[w.subId] = w.farmId;
      }
      await onRedistributeEqual(
        filteredList.map((w) => w.subId),
        farmIds
      );
      setSelectedSubIds(new Set());
    } finally {
      setBusyAction(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Team Workload Metadata */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900">
            قائمة الشغل — استمارات لسه ماتعملهاش Validation
          </h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-600">
            {TEAM_KEYS.map((k, idx) => (
              <React.Fragment key={k}>
                {idx > 0 && <span aria-hidden="true">·</span>}
                <span>
                  {TEAM[k].name}:{' '}
                  <strong className="font-mono-num text-slate-900">{memberCounts[k]}</strong>
                </span>
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-slate-500">المزارع المتبقية</div>
          <div className="text-2xl font-bold font-mono-num text-slate-900">
            {filteredList.length}
          </div>
        </div>
      </div>

      {!regionFound && (
        <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900">
          لم يتم تحديد عمود المنطقة تلقائياً في الاستمارة — تحقق من مسمى حقل المحافظة في Kobo.
        </div>
      )}

      {/* Filter Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-12 gap-3 items-end">
          <div className="lg:col-span-4">
            <label className="block text-xs font-medium text-slate-600 mb-1">
              بحث (مزرعة / مزارع / منطقة)
            </label>
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="ابحث بكود المزرعة أو اسم المزارع أو المحافظة..."
                className="w-full pl-9 pr-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
              />
            </div>
          </div>

          <div className="lg:col-span-2">
            <label className="block text-xs font-medium text-slate-600 mb-1">
              المسؤول
            </label>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value as 'me' | 'all' | TeamMemberKey)}
              className="w-full px-2.5 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
            >
              <option value="me">مزارعي ({displayName(currentUserKey)})</option>
              <option value="all">الكل</option>
              {TEAM_KEYS.map((k) => (
                <option key={k} value={k}>
                  {TEAM[k].name}
                </option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label className="block text-xs font-medium text-slate-600 mb-1">
              المنطقة
            </label>
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="w-full px-2.5 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
            >
              <option value="الكل">الكل</option>
              {regions.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2">
            <label className="block text-xs font-medium text-slate-600 mb-1">
              المهندس
            </label>
            <select
              value={selectedEngineer}
              onChange={(e) => setSelectedEngineer(e.target.value)}
              className="w-full px-2.5 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
            >
              <option value="الكل">الكل</option>
              {engineers.map((eng) => (
                <option key={eng} value={eng}>
                  {eng}
                </option>
              ))}
            </select>
          </div>

          <div className="lg:col-span-2 pb-1.5">
            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={hideReviewed}
                onChange={(e) => setHideReviewed(e.target.checked)}
                className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
              />
              <span>إخفاء اللي اتراجع</span>
            </label>
          </div>
        </div>
      </div>

      {/* Worklist Table or Empty State */}
      {filteredList.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-10 text-center">
          <p className="text-sm font-semibold text-slate-900">
            مفيش استمارات متبقية بالفلاتر دي.
          </p>
          <p className="text-xs text-slate-500 mt-1">
            جرب تغيير فلتر المسؤول إلى &ldquo;الكل&rdquo; أو إلغاء تحديد &ldquo;إخفاء اللي اتراجع&rdquo;.
          </p>
          {scope !== 'all' && (
            <button
              type="button"
              onClick={() => setScope('all')}
              className="mt-4 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium rounded-lg transition-colors cursor-pointer"
            >
              عرض كل المزارع
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80 text-xs font-semibold text-slate-600">
                    <th className="py-2.5 px-3.5 w-10">
                      <input
                        type="checkbox"
                        checked={allVisibleChecked}
                        onChange={toggleAllVisible}
                        aria-label="Select all rows"
                        className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600 cursor-pointer"
                      />
                    </th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Farm ID</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Farm Name</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Farmer</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Region</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Engineer</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap">Assigned</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap text-right">Submitted</th>
                    <th className="py-2.5 px-3.5 whitespace-nowrap text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {filteredList.map((row) => {
                    const checked = selectedSubIds.has(row.subId);
                    const localRev = Object.values(masterReviews).find(
                      (m) => String(m.submission_id) === row.subId || m.farm_id === row.farmId
                    );
                    const isReviewed = localRev && localRev.Form_Status !== STATUS_PENDING;
                    return (
                      <tr
                        key={row.subId}
                        onClick={() => toggleRow(row.subId)}
                        className={`h-10 transition-colors cursor-pointer ${
                          checked ? 'bg-emerald-50/60' : 'hover:bg-slate-50'
                        }`}
                      >
                        <td
                          className="py-2 px-3.5"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleRow(row.subId)}
                            className="rounded border-slate-300 text-emerald-700 focus:ring-emerald-600 cursor-pointer"
                          />
                        </td>
                        <td className="py-2 px-3.5 font-mono-num font-semibold text-slate-900 whitespace-nowrap">
                          {row.farmId}
                          {isReviewed && (
                            <span className="ml-2 text-[11px] font-normal text-emerald-700">
                              · Reviewed
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-3.5 font-medium text-slate-800">{row.farmName}</td>
                        <td className="py-2 px-3.5 text-slate-600">{row.farmer}</td>
                        <td className="py-2 px-3.5 text-slate-600 whitespace-nowrap">{row.region}</td>
                        <td className="py-2 px-3.5 text-slate-600 whitespace-nowrap">{row.engineer}</td>
                        <td className="py-2 px-3.5 text-slate-700 font-medium whitespace-nowrap">
                          {row.assignedName}
                        </td>
                        <td className="py-2 px-3.5 font-mono-num text-slate-500 text-right whitespace-nowrap">
                          {row.submitted}
                        </td>
                        <td
                          className="py-2 px-3.5 text-right whitespace-nowrap"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            onClick={() => onOpenFarmByIndex(row.idx, currentOrder)}
                            className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-emerald-800 hover:text-emerald-950 hover:bg-emerald-50 rounded-md transition-colors cursor-pointer"
                          >
                            <span>افتح</span>
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Primary Worklist Action Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                disabled={visibleSelected.length === 0}
                onClick={() => onOpenFarmByIndex(visibleSelected[0].idx, currentOrder)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-700 hover:bg-emerald-800 disabled:opacity-45 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap cursor-pointer"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                <span>افتح المزرعة المحددة ({visibleSelected.length})</span>
              </button>

              <button
                type="button"
                onClick={() => onOpenFarmByIndex(filteredList[0].idx, currentOrder)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 border border-slate-300 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>ابدأ من أول مزرعة</span>
              </button>
            </div>

            {/* Expandable Reassignment Controls */}
            <button
              type="button"
              onClick={() => setReassignOpen((o) => !o)}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 bg-white border border-slate-200 rounded-lg transition-colors whitespace-nowrap cursor-pointer"
            >
              <ArrowRightLeft className="w-3.5 h-3.5" />
              <span>إعادة التوزيع</span>
              {reassignOpen ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {reassignOpen && (
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">
                    انقل المحدد إلى
                  </label>
                  <select
                    value={moveTarget}
                    onChange={(e) => setMoveTarget(e.target.value as TeamMemberKey)}
                    className="w-full px-3 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
                  >
                    {TEAM_KEYS.map((k) => (
                      <option key={k} value={k}>
                        {TEAM[k].name} ({TEAM[k].email})
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  disabled={visibleSelected.length === 0 || busyAction}
                  onClick={handleMoveSelected}
                  className="py-2 px-4 bg-slate-900 hover:bg-slate-800 disabled:opacity-45 text-white text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                >
                  نقل المحدد ({visibleSelected.length})
                </button>

                <button
                  type="button"
                  disabled={filteredList.length === 0 || busyAction}
                  onClick={handleRedistributeDisplayed}
                  className="inline-flex items-center justify-center gap-2 py-2 px-4 bg-white hover:bg-slate-50 border border-slate-300 text-slate-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                >
                  <Scale className="w-3.5 h-3.5" />
                  <span>قسّم المعروض ({filteredList.length}) بالتساوي</span>
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
