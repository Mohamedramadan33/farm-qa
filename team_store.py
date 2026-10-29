"""
team_store.py — إدارة المستخدمين وتسجيل الدخول وتوجيه المزارع عبر Airtable
"""
from __future__ import annotations

import hashlib
import hmac
import secrets as pysecrets
from datetime import datetime
import streamlit as st
from pyairtable import Api

TEAM = {
    "mohamed": {"email": "mohamed@ebda.earth", "name": "Mohamed"},
    "mansy": {"email": "mansy.ayman@ebda.earth", "name": "Mansy"},
    "haidy": {"email": "haidy.abdelnasser@ebda.earth", "name": "Haidy"},
    "islam": {"email": "eslam.ashraf@ebda.earth", "name": "Islam"},
}
ALIASES = {"eslam": "islam", "islam.ashraf": "islam"}
DEFAULT_PASSWORD = "000000"

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def display_name(key) -> str:
    if not key:
        return "—"
    return TEAM.get(str(key), {}).get("name", str(key))

def resolve_user(ident: str):
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

def login(key: str, password: str):
    """التحقق من تسجيل الدخول (الباسورد الافتراضي 000000)"""
    if key in TEAM:
        # السماح بالدخول بالباسورد الافتراضي أو أي باسورد مسجل في الـ Session
        passwords = st.session_state.setdefault("_passwords", {})
        saved_pw = passwords.get(key, DEFAULT_PASSWORD)
        if hmac.compare_digest(saved_pw.encode(), password.encode()):
            changed = (saved_pw != DEFAULT_PASSWORD)
            return True, changed
    return False, False

def change_password(key: str, old_pw: str, new_pw: str):
    """تغيير الباسورد وحفظه مؤقتاً في الـ Session"""
    if len(new_pw) < 6:
        return False, "الباسورد الجديد لازم يكون 6 حروف/أرقام على الأقل."
    if new_pw == DEFAULT_PASSWORD:
        return False, "اختار باسورد غير الافتراضي."
    
    passwords = st.session_state.setdefault("_passwords", {})
    saved_pw = passwords.get(key, DEFAULT_PASSWORD)
    
    if not hmac.compare_digest(saved_pw.encode(), old_pw.encode()):
        return False, "الباسورد الحالي غلط."
    
    passwords[key] = new_pw
    return True, "تم تغيير الباسورد ✅"

def _get_airtable_table():
    cfg = st.secrets["airtable"]
    return Api(cfg["api_key"]).table(cfg["base_id"], cfg["table_name"])

def load_assignments() -> dict:
    try:
        table = _get_airtable_table()
        records = table.all()
        assignments = {}
        for r in records:
            fields = r.get("fields", {})
            sub_id = str(fields.get("submission_id") or fields.get("sub_id") or "")
            if sub_id:
                assignments[sub_id] = {
                    "to": fields.get("assigned_to", "mohamed"),
                    "by": fields.get("assigned_by", "auto"),
                    "at": fields.get("Reviewed_At", _now())
                }
        return assignments
    except Exception:
        return {}

def sync_new(farms, pending_ids=None) -> int:
    return 0

def reassign(sub_ids, to_key: str, by: str, farm_ids: dict | None = None):
    pass

def redistribute(sub_ids, by: str, farm_ids: dict | None = None) -> dict:
    return {k: 0 for k in TEAM}
