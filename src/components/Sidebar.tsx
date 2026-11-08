import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  ClipboardList,
  FileSearch,
  KeyRound,
  LogOut,
  Play,
  RefreshCw,
  Shuffle,
  X,
} from 'lucide-react';
import { DEFAULT_PASSWORD } from '../lib/workDivider';
import type { TeamMemberKey, UserSession } from '../types/qa';

export type ScopeMode = 'Specific Farm (Farm ID)' | 'By Submission Date' | 'All Farms';

interface SidebarProps {
  user: UserSession;
  passwords: Record<string, string>;
  onChangePassword: (key: TeamMemberKey, newPassword: string) => void;
  onLogout: () => void;
  activeView: 'list' | 'review';
  hasLoadedForms: boolean;
  onNavigateView: (v: 'list' | 'review') => void;
  onRunDivideWork: () => Promise<string[]>;
  autoSat: boolean;
  setAutoSat: (v: boolean) => void;
  useNdvi: boolean;
  setUseNdvi: (v: boolean) => void;
  onRefreshKobo: () => Promise<void>;
  refreshingKobo: boolean;
  onLoadScopeForms: (mode: ScopeMode, farmIdInput: string, dateInput: string) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  user,
  passwords,
  onChangePassword,
  onLogout,
  activeView,
  hasLoadedForms,
  onNavigateView,
  onRunDivideWork,
  autoSat,
  setAutoSat,
  useNdvi,
  setUseNdvi,
  onRefreshKobo,
  refreshingKobo,
  onLoadScopeForms,
  mobileOpen,
  onCloseMobile,
}) => {
  const [pwOpen, setPwOpen] = useState(false);
  const [oldPw, setOldPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [newPw2, setNewPw2] = useState('');
  const [pwMsg, setPwMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const [dividing, setDividing] = useState(false);
  const [divideLogs, setDivideLogs] = useState<string[]>([]);

  const [scopeMode, setScopeMode] = useState<ScopeMode>('All Farms');
  const [farmIdInput, setFarmIdInput] = useState('');
  const [dateInput, setDateInput] = useState('2026-11-02');

  const handlePasswordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPwMsg(null);
    const savedPw = passwords[user.key] ?? DEFAULT_PASSWORD;
    if (oldPw !== savedPw) {
      setPwMsg({ ok: false, text: 'الباسورد الحالي غلط.' });
      return;
    }
    if (newPw.length < 6) {
      setPwMsg({ ok: false, text: 'الباسورد الجديد لازم يكون 6 خانات على الأقل.' });
      return;
    }
    if (newPw === DEFAULT_PASSWORD) {
      setPwMsg({ ok: false, text: 'اختار باسورد غير الافتراضي.' });
      return;
    }
    if (newPw !== newPw2) {
      setPwMsg({ ok: false, text: 'الباسوردين مش متطابقين.' });
      return;
    }
    onChangePassword(user.key, newPw);
    setOldPw('');
    setNewPw('');
    setNewPw2('');
    setPwMsg({ ok: true, text: 'تم تغيير الباسورد بنجاح.' });
  };

  const handleDivideWorkClick = async () => {
    setDividing(true);
    setDivideLogs(['Running Distribution Pipeline...']);
    try {
      const logs = await onRunDivideWork();
      setDivideLogs(logs);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setDivideLogs([`خطأ أثناء توزيع المهام: ${msg}`]);
    } finally {
      setDividing(false);
    }
  };

  const handleLoadFormsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onLoadScopeForms(scopeMode, farmIdInput, dateInput);
    onCloseMobile();
  };

  const content = (
    <aside className="w-[276px] shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-y-auto">
      {/* Brand & Mobile Close */}
      <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
        <span className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap">
          EBDA &amp; Kobo
        </span>
        <button
          type="button"
          onClick={onCloseMobile}
          className="lg:hidden p-1.5 text-slate-500 hover:text-slate-800 rounded-md cursor-pointer"
          aria-label="Close sidebar"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Auditor Profile & Password Expander */}
      <div className="p-4 border-b border-slate-200 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-900 truncate">{user.name}</div>
            <div className="text-xs text-slate-500 truncate">{user.email}</div>
          </div>
          <button
            type="button"
            onClick={onLogout}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200/70 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Logout</span>
          </button>
        </div>

        <div>
          <button
            type="button"
            onClick={() => {
              setPwOpen((o) => !o);
              setPwMsg(null);
            }}
            className="w-full flex items-center justify-between px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 rounded-md transition-colors cursor-pointer"
          >
            <span className="inline-flex items-center gap-1.5">
              <KeyRound className="w-3.5 h-3.5 text-slate-400" />
              <span>تغيير الباسورد</span>
            </span>
            {pwOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>

          {pwOpen && (
            <form onSubmit={handlePasswordSubmit} className="mt-2.5 space-y-2 px-1">
              <input
                type="password"
                value={oldPw}
                onChange={(e) => setOldPw(e.target.value)}
                placeholder="الباسورد الحالي"
                className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                required
              />
              <input
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="الباسورد الجديد"
                className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                required
              />
              <input
                type="password"
                value={newPw2}
                onChange={(e) => setNewPw2(e.target.value)}
                placeholder="تأكيد الباسورد الجديد"
                className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-md focus:outline-none focus:border-emerald-600"
                required
              />
              {pwMsg && (
                <p className={`text-[11px] ${pwMsg.ok ? 'text-emerald-700' : 'text-red-600'}`}>
                  {pwMsg.text}
                </p>
              )}
              <button
                type="submit"
                className="w-full py-1.5 px-3 bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium rounded-md transition-colors cursor-pointer"
              >
                غيّر الباسورد
              </button>
            </form>
          )}
        </div>
      </div>

      {/* Primary Navigation */}
      <div className="p-4 border-b border-slate-200 space-y-1">
        <button
          type="button"
          onClick={() => {
            onNavigateView('list');
            onCloseMobile();
          }}
          className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm font-medium rounded-lg transition-colors cursor-pointer ${
            activeView === 'list'
              ? 'bg-emerald-50 text-emerald-900 font-semibold'
              : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
          }`}
        >
          <ClipboardList className="w-4 h-4 shrink-0" />
          <span className="whitespace-nowrap">قائمة الشغل (Worklist)</span>
        </button>

        <button
          type="button"
          disabled={!hasLoadedForms}
          onClick={() => {
            onNavigateView('review');
            onCloseMobile();
          }}
          className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
            !hasLoadedForms
              ? 'text-slate-300 cursor-not-allowed'
              : activeView === 'review'
                ? 'bg-emerald-50 text-emerald-900 font-semibold cursor-pointer'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 cursor-pointer'
          }`}
        >
          <FileSearch className="w-4 h-4 shrink-0" />
          <span className="whitespace-nowrap">مراجعة الاستمارة (Auditor)</span>
        </button>
      </div>

      {/* Work Management Pipeline */}
      <div className="p-4 border-b border-slate-200 space-y-2.5">
        <h2 className="text-xs font-semibold text-slate-700">Work Management</h2>
        <button
          type="button"
          disabled={dividing}
          onClick={handleDivideWorkClick}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-slate-900 hover:bg-slate-800 disabled:opacity-60 text-white text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
        >
          <Shuffle className="w-3.5 h-3.5 shrink-0" />
          <span>{dividing ? 'Running Distribution...' : 'Divide Work Fairly'}</span>
        </button>

        {divideLogs.length > 0 && (
          <div className="p-2.5 rounded-lg bg-slate-900 text-slate-100 text-[11px] font-mono-num leading-relaxed overflow-x-auto">
            {divideLogs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        )}
      </div>

      {/* Scope Loader & Satellite AI Options */}
      <form onSubmit={handleLoadFormsSubmit} className="p-4 space-y-3.5 flex-1">
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Scope
          </label>
          <select
            value={scopeMode}
            onChange={(e) => setScopeMode(e.target.value as ScopeMode)}
            className="w-full px-2.5 py-2 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600"
          >
            <option value="Specific Farm (Farm ID)">Specific Farm (Farm ID)</option>
            <option value="By Submission Date">By Submission Date</option>
            <option value="All Farms">All Farms</option>
          </select>
        </div>

        {scopeMode === 'Specific Farm (Farm ID)' && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Farm ID
            </label>
            <input
              type="text"
              value={farmIdInput}
              onChange={(e) => setFarmIdInput(e.target.value)}
              placeholder="e.g. FAY-1042"
              className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600 font-mono-num"
            />
          </div>
        )}

        {scopeMode === 'By Submission Date' && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Submission Date
            </label>
            <input
              type="date"
              value={dateInput}
              onChange={(e) => setDateInput(e.target.value)}
              className="w-full px-2.5 py-1.5 text-xs bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-emerald-600 font-mono-num"
            />
          </div>
        )}

        <div className="space-y-2 pt-1">
          <label className="flex items-start gap-2 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={autoSat}
              onChange={(e) => setAutoSat(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
            />
            <span>Auto-run Satellite &amp; Tree AI on selected farm</span>
          </label>

          <label className="flex items-start gap-2 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={useNdvi}
              onChange={(e) => setUseNdvi(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
            />
            <span>Include NDVI in AI check (Sentinel-2)</span>
          </label>
        </div>

        <div className="pt-2 space-y-2">
          <button
            type="button"
            disabled={refreshingKobo}
            onClick={onRefreshKobo}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 text-xs font-medium rounded-lg transition-colors whitespace-nowrap cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 shrink-0 ${refreshingKobo ? 'animate-spin' : ''}`} />
            <span>{refreshingKobo ? 'Refreshing...' : 'Refresh Kobo Data'}</span>
          </button>

          <button
            type="submit"
            className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 shrink-0" />
            <span>Load Forms</span>
          </button>
        </div>
      </form>
    </aside>
  );

  return (
    <>
      <div className="hidden lg:block h-full">{content}</div>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/40"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <div className="relative z-10 h-full">{content}</div>
        </div>
      )}
    </>
  );
};
