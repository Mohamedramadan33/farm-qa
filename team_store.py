# -*- coding: utf-8 -*-
"""
team_store.py — Team authentication, review states, and remote sync (Airtable / KoboToolbox).
"""

import json
import os
import datetime
from typing import Dict, Any, Optional, List
import requests

from work_divider import TEAM, TEAM_KEYS, DEFAULT_PASSWORD, display_name

DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")
REVIEWS_FILE = os.path.join(DATA_DIR, "reviews.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save_json(path: str, data: Any):
    _ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_users_db() -> Dict[str, Dict[str, Any]]:
    default_users = {
        k: {
            "password": DEFAULT_PASSWORD,
            "must_change_password": True,
            "name": TEAM[k]["name"],
            "email": TEAM[k]["email"],
        }
        for k in TEAM_KEYS
    }
    saved = _load_json(USERS_FILE, {})
    for k, v in default_users.items():
        if k not in saved:
            saved[k] = v
    return saved


def verify_user(user_key: str, password: str) -> Optional[Dict[str, Any]]:
    users = get_users_db()
    u = users.get(user_key.lower().strip())
    if u and u.get("password") == password:
        return {
            "key": user_key.lower().strip(),
            "name": u["name"],
            "email": u["email"],
            "must_change_password": u.get("must_change_password", False),
        }
    return None


def update_password(user_key: str, new_password: str) -> bool:
    users = get_users_db()
    k = user_key.lower().strip()
    if k in users and len(new_password) >= 6:
        users[k]["password"] = new_password
        users[k]["must_change_password"] = False
        _save_json(USERS_FILE, users)
        return True
    return False


def get_all_reviews() -> Dict[str, Any]:
    return _load_json(REVIEWS_FILE, {"master": {}, "fields": {}})


def save_master_review(farm_id: str, submission_id: str, record: Dict[str, Any]):
    data = get_all_reviews()
    record["farm_id"] = str(farm_id)
    record["submission_id"] = str(submission_id)
    record["Reviewed_At"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    data["master"][str(farm_id)] = record
    _save_json(REVIEWS_FILE, data)


def save_field_decisions(farm_id: str, field_rows: List[Dict[str, Any]]):
    data = get_all_reviews()
    filtered = [
        f for f in field_rows
        if f.get("Field_Key", "").startswith("_AI_")
        or f.get("Item_Decision") != "OK"
        or f.get("Suggested_Correction_or_Comment", "").strip() != ""
        or f.get("Attach_to_Summary")
    ]
    data["fields"][str(farm_id)] = filtered
    _save_json(REVIEWS_FILE, data)


def get_assignments() -> Dict[str, Dict[str, Any]]:
    return _load_json(ASSIGNMENTS_FILE, {})


def save_assignments(assignments: Dict[str, Dict[str, Any]]):
    _save_json(ASSIGNMENTS_FILE, assignments)


def set_assignment(sub_id: str, to_member: str, by_user: str = "manual", farm_id: str = ""):
    assigns = get_assignments()
    assigns[str(sub_id)] = {
        "to": to_member,
        "by": by_user,
        "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "farm_id": str(farm_id),
    }
    save_assignments(assigns)


def sync_to_kobo_validation(
    server_url: str,
    asset_uid: str,
    token: str,
    submission_id: str,
    status_code: str
) -> Dict[str, Any]:
    """
    PATCH validation status in KoboToolbox API
    status_code: 'validation_status_approved', 'validation_status_not_approved', 'validation_status_on_hold'
    """
    if not token or not asset_uid:
        return {"ok": False, "error": "Kobo Token or Asset UID is missing"}

    clean_server = server_url.rstrip("/")
    url = f"{clean_server}/api/v2/assets/{asset_uid}/data/validation_statuses/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    try:
        sub_id_int = int(submission_id)
        sub_val = sub_id_int
    except Exception:
        sub_val = submission_id

    payload = {
        "payload": {
            "submission_ids": [sub_val],
            "validation_status.uid": status_code,
        }
    }
    try:
        resp = requests.patch(url, headers=headers, json=payload, timeout=20)
        return {"ok": resp.status_code in [200, 201, 204], "status": resp.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def sync_to_airtable(
    api_key: str,
    base_id: str,
    table_name: str,
    review_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Upsert review record into Airtable using farm_id as merge key
    """
    if not api_key or not base_id or not table_name:
        return {
            "ok": True,
            "message": "تم حفظ المراجعة محلياً بنجاح (لم يتم تفعيل مفاتيح Airtable في الإعدادات)."
        }

    url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    fields = {
        "farm_id": str(review_record.get("farm_id", "")),
        "submission_id": str(review_record.get("submission_id", "")),
        "Form_Status": str(review_record.get("Form_Status", "")),
        "Rejected_Items_Count": int(review_record.get("Rejected_Items_Count", 0)),
        "Consolidated_Comments": str(review_record.get("Consolidated_Comments", "")),
        "General_Auditor_Note": str(review_record.get("General_Auditor_Note", "")),
        "Auditor_Name": str(review_record.get("Auditor_Name", "")),
        "Reviewed_At": str(review_record.get("Reviewed_At", "")),
    }

    body = {
        "performUpsert": {"fieldsToMergeOn": ["farm_id"]},
        "typecast": True,
        "records": [{"fields": fields}],
    }
    try:
        resp = requests.patch(url, headers=headers, json=body, timeout=20)
        if resp.ok:
            return {"ok": True, "message": "تمت المزامنة بنجاح مع Airtable (Upsert)."}
        return {"ok": False, "message": f"رفض Airtable الطلب (كود {resp.status_code}): {resp.text[:250]}"}
    except Exception as e:
        return {"ok": False, "message": f"خطأ في الاتصال بـ Airtable: {e}"}
