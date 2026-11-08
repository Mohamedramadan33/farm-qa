import React, { useState } from 'react';
import { KeyRound, LogIn, ShieldCheck } from 'lucide-react';
import { DEFAULT_PASSWORD, resolveUser } from '../lib/workDivider';
import { TEAM, TEAM_KEYS, type TeamMemberKey, type UserSession } from '../types/qa';

interface LoginScreenProps {
  passwords: Record<string, string>;
  onLoginSuccess: (session: UserSession) => void;
  onPasswordChanged: (key: TeamMemberKey, newPassword: string) => void;
  pendingForcedUser: UserSession | null;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({
  passwords,
  onLoginSuccess,
  onPasswordChanged,
  pendingForcedUser,
}) => {
  const [ident, setIdent] = useState('mohamed@ebda.earth');
  const [password, setPassword] = useState(DEFAULT_PASSWORD);
  const [error, setError] = useState('');

  // Forced password change state
  const [oldPw, setOldPw] = useState(DEFAULT_PASSWORD);
  const [newPw, setNewPw] = useState('');
  const [newPw2, setNewPw2] = useState('');
  const [chgError, setChgError] = useState('');

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    const key = resolveUser(ident);
    if (!key) {
      setError('المستخدم مش موجود في الفريق (استخدم حساب أحد أعضاء فريق EBDA)');
      return;
    }
    const savedPw = passwords[key] ?? DEFAULT_PASSWORD;
    if (password !== savedPw) {
      setError('الباسورد غلط');
      return;
    }
    const member = TEAM[key];
    onLoginSuccess({
      key,
      name: member.name,
      email: member.email,
      mustChangePassword: savedPw === DEFAULT_PASSWORD,
    });
  };

  const handleForcedChangeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!pendingForcedUser) return;
    setChgError('');

    const savedPw = passwords[pendingForcedUser.key] ?? DEFAULT_PASSWORD;
    if (oldPw !== savedPw) {
      setChgError('الباسورد الحالي غلط.');
      return;
    }
    if (newPw.length < 6) {
      setChgError('الباسورد الجديد لازم يكون 6 حروف أو أرقام على الأقل.');
      return;
    }
    if (newPw === DEFAULT_PASSWORD) {
      setChgError('اختار باسورد غير الافتراضي (000000).');
      return;
    }
    if (newPw !== newPw2) {
      setChgError('الباسوردين مش متطابقين.');
      return;
    }

    onPasswordChanged(pendingForcedUser.key, newPw);
  };

  if (pendingForcedUser && pendingForcedUser.mustChangePassword) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col justify-center items-center px-4 py-12">
        <div className="w-full max-w-md bg-white border border-slate-200 rounded-xl p-8">
          <div className="flex items-center gap-3 mb-2">
            <KeyRound className="w-5 h-5 text-emerald-700 shrink-0" />
            <h1 className="text-xl font-bold text-slate-900">
              غيّر الباسورد الافتراضي أول مرة
            </h1>
          </div>
          <p className="text-sm text-slate-600 mb-6">
            مرحباً {pendingForcedUser.name} ({pendingForcedUser.email}) — لحماية حسابات المراجعة يجب تعيين كلمة مرور خاصة بك قبل المتابعة.
          </p>

          <form onSubmit={handleForcedChangeSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1.5">
                الباسورد الحالي
              </label>
              <input
                type="password"
                value={oldPw}
                onChange={(e) => setOldPw(e.target.value)}
                className="w-full px-3.5 py-2.5 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-emerald-600 font-mono-num"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1.5">
                الباسورد الجديد (6 خانات على الأقل)
              </label>
              <input
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="أدخل كلمة مرور جديدة"
                className="w-full px-3.5 py-2.5 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-emerald-600"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1.5">
                تأكيد الباسورد الجديد
              </label>
              <input
                type="password"
                value={newPw2}
                onChange={(e) => setNewPw2(e.target.value)}
                placeholder="أعد كتابة كلمة المرور الجديدة"
                className="w-full px-3.5 py-2.5 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-emerald-600"
                required
              />
            </div>

            {chgError && (
              <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
                {chgError}
              </div>
            )}

            <button
              type="submit"
              className="w-full py-2.5 px-4 bg-emerald-700 hover:bg-emerald-800 text-white text-sm font-semibold rounded-lg transition-colors cursor-pointer"
            >
              غيّر الباسورد والمتابعة
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center items-center px-4 py-12">
      <div className="w-full max-w-md bg-white border border-slate-200 rounded-xl p-8">
        <div className="mb-6 text-center">
          <div className="inline-flex items-center justify-center w-11 h-11 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-700 mb-3">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            EBDA &amp; Kobo — Login
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            منصة فحص استمارات PGS والتحقق الجغرافي بالأقمار الصناعية
          </p>
        </div>

        <form onSubmit={handleLoginSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Email / Username
            </label>
            <input
              type="text"
              value={ident}
              onChange={(e) => setIdent(e.target.value)}
              placeholder="mohamed@ebda.earth"
              className="w-full px-3.5 py-2.5 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-emerald-600"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="000000"
              className="w-full px-3.5 py-2.5 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-emerald-600 font-mono-num"
              required
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-emerald-700 hover:bg-emerald-800 text-white text-sm font-semibold rounded-lg transition-colors cursor-pointer"
          >
            <LogIn className="w-4 h-4" />
            <span>Login</span>
          </button>
        </form>

        <div className="mt-6 pt-5 border-t border-slate-200">
          <div className="text-xs font-medium text-slate-500 mb-2.5">
            حسابات فريق المراجعة (كلمة المرور الافتراضية:{' '}
            <span className="font-mono-num text-slate-700">000000</span>):
          </div>
          <div className="grid grid-cols-2 gap-2">
            {TEAM_KEYS.map((k) => {
              const m = TEAM[k];
              const active = resolveUser(ident) === k;
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => {
                    setIdent(m.email);
                    setPassword(passwords[k] ?? DEFAULT_PASSWORD);
                    setError('');
                  }}
                  className={`text-left px-3 py-2 rounded-lg border text-xs transition-colors cursor-pointer ${
                    active
                      ? 'border-emerald-600 bg-emerald-50/60 text-emerald-950'
                      : 'border-slate-200 hover:border-slate-300 text-slate-700'
                  }`}
                >
                  <div className="font-semibold">{m.name}</div>
                  <div className="text-[11px] text-slate-500 truncate">{m.email}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
