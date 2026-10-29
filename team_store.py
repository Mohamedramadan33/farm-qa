"""
team_store.py — مستخدمين الفريق + توزيع المزارع، متخزنين في Airtable
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

def _get_airtable_table():
    cfg = st.secrets["airtable"]
    return Api(cfg["api_key"]).table(cfg["base_id"], cfg["table_name"])

def load_assignments() -> dict:
    """بيرجّع توجيهات المزارع من Airtable"""
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
