"""
team_store.py — مستخدمين الفريق + توزيع المزارع، متخزنين في Google Sheets
(نفس الـ Spreadsheet بتاع work_divider، في تابين جداد بيتعملوا تلقائياً):

  _users        → username | email | salt | pw_hash | changed | updated_at
  _assignments  → sub_id | farm_id | assigned_to | assigned_by | updated_at

الباسوردات بتتخزن Hash (PBKDF2) مش نص صريح.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets as pysecrets
from datetime import datetime

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

from work_divider import SPREADSHEET_KEY

# ═══════════════════════ TEAM ═══════════════════════
TEAM = {
    "mansy": {"email": "mansy.ayman@ebda.earth", "name": "Mansy"},
    "haidy": {"email": "haidy.abdelnasser@ebda.earth", "name": "Haidy"},
    "islam": {"email": "eslam.ashraf@ebda.earth", "name": "Islam"},
}
ALIASES = {"eslam": "islam", "islam.ashraf": "islam", "eslam.ashraf": "islam"}
DEFAULT_PASSWORD = "000000"

USERS_TAB = "_users"
ASSIGN_TAB = "_assignments"
USERS_HEADERS = ("username", "email", "salt", "pw_hash", "changed", "updated_at")
ASSIGN_HEADERS = ("sub_id", "farm_id", "assigned_to", "assigned_by", "updated_at")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _cell(row, i, default=""):
    return row[i] if row is not None and len(row) > i else default


def display_name(key) -> str:
    if not key:
        return "—"
    return TEAM.get(str(key), {}).get("name", str(key))


def resolve_user(ident: str):
    """يقبل: mansy / Mansy / mansy.ayman@ebda.earth / islam / eslam ... ويرجّع الـ key أو None."""
    i = str(ident or "").strip().lower()
    if not i:
        return None
    if i in TEAM:
        return i
    if i in ALIASES:
        return ALIASES[i]
    for k, v in TEAM.items():
        email = v["email"].lower()
        if i == email or i == email.split("@")[0]:
            return k
    return None


# ═══════════════════════ SHEETS ═══════════════════════
@st.cache_resource(show_spinner=False)
def _spreadsheet():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"])
    return gspread.authorize(creds).open_by_key(SPREADSHEET_KEY)


@st.cache_resource(show_spinner=False)
def _ws(title: str, headers: tuple):
    sh = _spreadsheet()
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=len(headers))
        ws.update(range_name="A1", values=[list(headers)], value_input_option="RAW")
        return ws


# ═══════════════════════ USERS / PASSWORDS ═══════════════════════
def _hash(pw: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt_hex), 120_000).hex()


def _new_hash(pw: str):
    salt = pysecrets.token_hex(16)
    return salt, _hash(pw, salt)


def _users_with_seed():
    """بيرجّع (ws, rows) وبيضيف أي مستخدم ناقص بالباسورد الافتراضي."""
    ws = _ws(USERS_TAB, USERS_HEADERS)
    vals = ws.get_all_values()
    have = {_cell(r, 0) for r in vals[1:]}
    new_rows = []
    for key, info in TEAM.items():
        if key not in have:
            salt, h = _new_hash(DEFAULT_PASSWORD)
            new_rows.append([key, info["email"], salt, h, "no", _now()])
    if new_rows:
        ws.append_rows(new_rows, value_input_option="RAW")
        vals = ws.get_all_values()
    return ws, vals


def login(key: str, password: str):
    """يرجّع (نجح؟, غيّر الباسورد قبل كده؟)."""
    _, vals = _users_with_seed()
    for r in vals[1:]:
        if _cell(r, 0) == key:
            ok = hmac.compare_digest(_hash(password, _cell(r, 2)), _cell(r, 3))
            return ok, _cell(r, 4) == "yes"
    return False, False


def change_password(key: str, old_pw: str, new_pw: str):
    """يرجّع (نجح؟, رسالة)."""
    if len(new_pw) < 6:
        return False, "الباسورد الجديد لازم يكون 6 حروف/أرقام على الأقل."
    if new_pw == DEFAULT_PASSWORD:
        return False, "اختار باسورد غير الافتراضي."
    ws, vals = _users_with_seed()
    for i, r in enumerate(vals[1:], start=2):
        if _cell(r, 0) == key:
            if not hmac.compare_digest(_hash(old_pw, _cell(r, 2)), _cell(r, 3)):
                return False, "الباسورد الحالي غلط."
            salt, h = _new_hash(new_pw)
            ws.update(range_name=f"C{i}:F{i}", values=[[salt, h, "yes", _now()]],
                      value_input_option="RAW")
            return True, "تم تغيير الباسورد ✅"
    return False, "المستخدم مش موجود."


# ═══════════════════════ ASSIGNMENTS ═══════════════════════
@st.cache_data(ttl=20, show_spinner=False)
def load_assignments() -> dict:
    """{sub_id: {"to": key, "by": who, "at": time}}"""
    vals = _ws(ASSIGN_TAB, ASSIGN_HEADERS).get_all_values()
    return {_cell(r, 0): {"to": _cell(r, 2), "by": _cell(r, 3), "at": _cell(r, 4)}
            for r in vals[1:] if _cell(r, 0)}


def sync_new(farms, pending_ids=None) -> int:
    """
    farms: list of (sub_id, farm_id).
    أي مزرعة لسه ملهاش مسؤول بتتوزّع على الأقل واحد فيهم عدداً، فالتقسيم بيطلع متساوي.
    المزارع اللي لسه ماتراجعتش بتتوزّع الأول عشان نصيبهم يطلع متساوي.
    """
    ws = _ws(ASSIGN_TAB, ASSIGN_HEADERS)
    vals = ws.get_all_values()
    existing = {_cell(r, 0): _cell(r, 2) for r in vals[1:] if _cell(r, 0)}
    ids = {str(s) for s, _ in farms}
    counts = {k: 0 for k in TEAM}
    for sid, to in existing.items():
        if sid in ids and to in counts:
            counts[to] += 1

    pend = {str(x) for x in (pending_ids or set())}
    new = [(str(s), str(f)) for s, f in farms if str(s) not in existing]
    new.sort(key=lambda t: (t[0] not in pend, int(t[0]) if t[0].isdigit() else 0, t[0]))

    rows, order = [], list(TEAM)
    for sid, fid in new:
        to = min(order, key=lambda k: (counts[k], order.index(k)))
        counts[to] += 1
        rows.append([sid, fid, to, "auto", _now()])
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
        load_assignments.clear()
    return len(rows)


def _apply(mapping: dict, by: str, farm_ids: dict | None = None):
    ws = _ws(ASSIGN_TAB, ASSIGN_HEADERS)
    vals = ws.get_all_values()
    rowno = {_cell(r, 0): i for i, r in enumerate(vals[1:], start=2) if _cell(r, 0)}
    now = _now()
    updates, appends = [], []
    for sid, to in mapping.items():
        sid = str(sid)
        if sid in rowno:
            n = rowno[sid]
            updates.append({"range": f"C{n}:E{n}", "values": [[to, by, now]]})
        else:
            appends.append([sid, (farm_ids or {}).get(sid, ""), to, by, now])
    if updates:
        ws.batch_update(updates, value_input_option="RAW")
    if appends:
        ws.append_rows(appends, value_input_option="RAW")
    load_assignments.clear()


def reassign(sub_ids, to_key: str, by: str, farm_ids: dict | None = None):
    if to_key not in TEAM:
        raise ValueError("مستخدم غير معروف")
    _apply({str(s): to_key for s in sub_ids}, by, farm_ids)


def redistribute(sub_ids, by: str, farm_ids: dict | None = None) -> dict:
    """يقسم القايمة دي بالتساوي (round-robin) على كل الفريق. بيرجّع عدد كل واحد."""
    keys = list(TEAM)
    ordered = sorted((str(s) for s in sub_ids), key=lambda s: (int(s) if s.isdigit() else 0, s))
    mapping = {sid: keys[i % len(keys)] for i, sid in enumerate(ordered)}
    _apply(mapping, by, farm_ids)
    out = {k: 0 for k in keys}
    for to in mapping.values():
        out[to] += 1
    return out
