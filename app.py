from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import re
import sys
import threading
import time
import types
from datetime import datetime

# ── ALBUMENTATIONS PATCH BEFORE ANY OTHER IMPORT ──
try:
    import albumentations
    if not hasattr(albumentations, "functional"):
        f_mod = getattr(getattr(albumentations, "augmentations", None), "functional", types.ModuleType("albumentations.functional"))
        albumentations.functional = f_mod
        sys.modules["albumentations.functional"] = f_mod
except Exception:
    pass

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── استدعاء مكتبة Airtable ──
from pyairtable import Api

st.set_page_config(page_title="EBDA & Kobo", layout="wide", page_icon="🛰️")

try:
    if "KOBO_TOKEN" in st.secrets:
        os.environ["KOBO_TOKEN"] = str(st.secrets["KOBO_TOKEN"])
except Exception:
    pass

import farm_qa_core as qa  # noqa: E402

if not qa.KOBO_TOKEN:
    st.error("❌ KOBO_TOKEN غير موجود. ضيفه في App settings → Secrets.")
    st.stop()


def stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


# ═══════════════════════ LOGIN (CENTERED & WITHOUT LOCK) ═══════════════════════
def load_users() -> dict:
    try:
        if "users" in st.secrets:
            return {str(k): str(v) for k, v in st.secrets["users"].items()}
    except Exception:
        pass
    return {}


if not st.session_state.get("user"):
    users = load_users()
    st.write("")
    st.write("")
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown("<h2 style='text-align: center;'>EBDA & Kobo — Login</h2>", unsafe_allow_html=True)
        if not users:
            st.error("مفيش مستخدمين متسجلين. افتح Manage app ← Settings ← Secrets وضيف السطرين دول (الباسورد بين علامتي تنصيص):")
            st.code('[users]\nmohamed = "000000"', language="toml")
            st.stop()
        with st.container(border=True):
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                ok = stretch(st.form_submit_button, "Login", type="primary")
            if ok:
                pw = users.get(u.strip())
                if pw is not None and hmac.compare_digest(pw.encode(), p.encode()):
                    st.session_state["user"] = u.strip()
                    st.rerun()
                else:
                    time.sleep(1)
                    st.error("Username أو Password غلط")
    st.stop()

USER = st.session_state["user"]

# ═══════════════════════ CONSTANTS ═══════════════════════
MASTER_LOG_FILE = "master_form_reviews.csv"
FIELD_LOG_FILE = "field_level_reviews.csv"
DEBUG_DIR = "debug_imgs"
MASTER_COLS = ["farm_id", "submission_id", "Form_Status", "Rejected_Items_Count",
               "Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At"]
FIELD_COLS = ["farm_id", "Field_Key", "Question_Title", "Submitted_Answer",
              "Item_Decision", "Suggested_Correction_or_Comment", "Attach_to_Summary"]
AI_KEYS = ["ai_count", "ai_low", "ai_high", "ai_interior_trees", "veg_pct", "crown_m", "img_age", "ndvi_recent",
           "ai_error", "imagery_date", "zoom_used", "ai_raw", "ai_factor", "verdict", "confidence", "perimeter_m", "boundary_spacing_m"]
STATUS_PENDING = "⏳ Pending Review"
META_PREFIXES = ("_", "meta/", "formhub/", "__version__", "deviceid", "simserial", "phonenumber", "subscriberid")


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and v != v) else v


def to_bool(x) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in ("true", "1", "yes")


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


# ═══════════════════════ SHARED RESOURCES ═══════════════════════
@st.cache_resource(show_spinner="Loading DeepForest AI Model...")
def get_tree_counter():
    c = qa.TreeCounter()
    c._load()
    return c


@st.cache_resource
def get_ai_semaphore():
    return threading.Semaphore(1)


@st.cache_resource
def get_lock():
    return threading.Lock()


@st.cache_resource(ttl=600, show_spinner="Fetching submissions from KoboToolbox...")
def get_kobo_raw_and_df():
    raw_rows = qa.fetch_kobo()
    return raw_rows, qa.build_dataframe(raw_rows)


@st.cache_resource(ttl=600, show_spinner=False)
def get_cols_map():
    return qa.map_all(get_kobo_raw_and_df()[1])


@st.cache_data(ttl=3600, show_spinner=False)
def get_kobo_labels() -> dict:
    labels = {}
    try:
        r = requests.get(f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}.json",
                         headers={"Authorization": f"Token {qa.KOBO_TOKEN}"}, timeout=30)
        if r.ok:
            for item in r.json().get("content", {}).get("survey", []):
                name = item.get("name") or item.get("$autoname")
                lbl = item.get("label")
                if name and lbl:
                    labels[name.lower()] = " / ".join(str(x) for x in lbl if x) if isinstance(lbl, list) else str(lbl)
    except Exception:
        pass
    return labels


@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def fetch_kobo_media_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers={"Authorization": f"Token {qa.KOBO_TOKEN}"}, timeout=45)
        return r.content if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def run_ai_for_farm(sub_id: str, use_ndvi: bool):
    _, flat_df = get_kobo_raw_and_df()
    m = flat_df["_id"].astype(str) == str(sub_id)
    if not m.any():
        raise RuntimeError("Submission not found (refresh Kobo data)")
    row = flat_df[m].iloc[0]
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with get_ai_semaphore():
        try:
            counter = get_tree_counter()
        except Exception as e:
            counter = None
            qa.log.warning("DeepForest load warning: %s", e)
        rec = qa.audit_row(row, get_cols_map(), counter=counter, use_ai=True,
                           debug_dir=DEBUG_DIR, use_ndvi=use_ndvi)
    img = None
    p = os.path.join(DEBUG_DIR, f"{str(rec['farm_id']).replace('/', '_')}.jpg")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            img = fh.read()
    rec.pop("_poly", None)
    return rec, img


def update_kobo_validation_status(submission_id: str, status_code: str) -> bool:
    try:
        r = requests.patch(
            f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}/data/validation_statuses/",
            headers={"Authorization": f"Token {qa.KOBO_TOKEN}", "Content-Type": "application/json"},
            json={"payload": {"submission_ids": [int(submission_id)], "validation_status.uid": status_code}},
            timeout=20)
        return r.ok
    except Exception:
        return False


# ═══════════════════════ HELPERS ═══════════════════════
def build_attachment_lookup(raw_row: dict) -> dict:
    lookup = {}
    atts = raw_row.get("_attachments", [])
    if not isinstance(atts, list):
        return lookup
    for a in atts:
        if not isinstance(a, dict):
            continue
        full = str(a.get("filename", ""))
        short = full.split("/")[-1].strip()
        mime = str(a.get("mimetype", "")).lower()
        full_url = a.get("download_url")
        preview_url = a.get("download_medium_url") or a.get("download_small_url") or full_url
        if short and full_url:
            info = {"preview_url": preview_url, "full_url": full_url, "mimetype": mime, "filename": short}
            lookup[short] = info
            lookup[full] = info
    return lookup


def is_image_filename(val) -> bool:
    return isinstance(val, str) and val.strip().lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def is_video_filename(val) -> bool:
    return isinstance(val, str) and val.strip().lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"))


def resolve_label(col_path: str, label_map: dict) -> str:
    leaf = re.sub(r"\[\d+\]", "", str(col_path)).split("/")[-1].lower()
    idx = re.findall(r"\[\d+\]", str(col_path))
    prefix = f"{idx[0]} " if idx else ""
    return f"{prefix}{label_map[leaf]} ({leaf})" if leaf in label_map else str(col_path)


def blank_record(farm_id, err) -> dict:
    return {"farm_id": farm_id, "engineer": "—", "verdict": f"❌ Processing Error: {err}",
            "confidence": "🔴 Low", "flags": [], "_poly": None}


def apply_ai(idx, rec: dict):
    df = st.session_state["results"]
    for k in AI_KEYS:
        if k in rec:
            if k not in df.columns:
                df[k] = None
            df.at[idx, k] = rec[k]


# ═══════════════════════ STORAGE & DATABASE ═══════════════════════

def _csv_read(path: str, headers: list) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype={"farm_id": str})
        except Exception:
            pass
    return pd.DataFrame(columns=headers)


def _csv_write(path: str, df: pd.DataFrame):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def append_to_airtable(rec: dict):
    """
    دالة إرسال البيانات التراكمية إلى Airtable
    تعتمد على وجود إعدادات [airtable] في ملف secrets.toml
    """
    try:
        api = Api(st.secrets["airtable"]["api_key"])
        table = api.table(st.secrets["airtable"]["base_id"], st.secrets["airtable"]["table_name"])
        
        row_data = {
            "farm_id": str(rec.get("farm_id", "")),
            "submission_id": str(rec.get("submission_id", "")),
            "Form_Status": str(rec.get("Form_Status", "")),
            "Rejected_Items_Count": str(rec.get("Rejected_Items_Count", "0")), # إرسال كـ String
            "Consolidated_Comments": str(rec.get("Consolidated_Comments", "")),
            "General_Auditor_Note": str(rec.get("General_Auditor_Note", "")),
            "Auditor_Name": str(rec.get("Auditor_Name", "")),
            "Reviewed_At": str(rec.get("Reviewed_At", ""))
        }
        
        table.create(row_data)
        return True, ""
    except Exception as e:
        # التقاط الخطأ وإرجاعه للواجهة لكي نعرف السبب الحقيقي
        return False, str(e)


def save_master_decision(rec: dict):
    fid = str(rec["farm_id"])
    
    # 1. تحديث الملف المحلي (CSV) للـ Backup
    with get_lock():
        df = _csv_read(MASTER_LOG_FILE, MASTER_COLS)
        if not df.empty:
            df = df[df["farm_id"].astype(str) != fid]
        _csv_write(MASTER_LOG_FILE, pd.concat([df, pd.DataFrame([rec])], ignore_index=True))
    
    # 2. إرسال القرار لـ Airtable وإرجاع حالة الإرسال للواجهة
    return append_to_airtable(rec)


def save_field_decisions(farm_id: str, field_df: pd.DataFrame):
    fid = str(farm_id)
    f = field_df.copy()
    f["Attach_to_Summary"] = f["Attach_to_Summary"].map(to_bool)
    f["Suggested_Correction_or_Comment"] = f["Suggested_Correction_or_Comment"].fillna("").astype(str)
    keep = (f["Field_Key"].astype(str).str.startswith("_AI_") | (f["Item_Decision"] != "✅ OK")
            | (f["Suggested_Correction_or_Comment"].str.strip() != "") | f["Attach_to_Summary"])
    f = f[keep].copy()
    f["farm_id"] = fid
    f = f[FIELD_COLS]
    with get_lock():
        old = _csv_read(FIELD_LOG_FILE, FIELD_COLS)
        if not old.empty:
            old = old[old["farm_id"].astype(str) != fid]
        _csv_write(FIELD_LOG_FILE, pd.concat([old, f], ignore_index=True))


def get_field_decisions(farm_id: str) -> pd.DataFrame:
    key = f"fd_{farm_id}"
    if key not in st.session_state:
        df = _csv_read(FIELD_LOG_FILE, FIELD_COLS)
        st.session_state[key] = df[df["farm_id"].astype(str) == str(farm_id)]
    return st.session_state[key]


# ═══════════════════════ EXCEL EXPORT ═══════════════════════
def build_reviews_xlsx(res: pd.DataFrame, only_reviewed: bool) -> bytes:
    df = res[res["Form_Status"] != STATUS_PENDING] if only_reviewed else res
    out = pd.DataFrame({
        "Farm ID": df["farm_id"],
        "Farm Name": df["farm_name"] if "farm_name" in df.columns else "",
        "Engineer": df["engineer"],
        "Status": df["Form_Status"],
        "Rejected Items": df["Rejected_Items_Count"],
        "Comment": df["Consolidated_Comments"],
        "Auditor": df["Auditor_Name"],
        "Reviewed At": df["Reviewed_At"],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="Reviews")
        ws = w.sheets["Reviews"]
        from openpyxl.styles import Alignment
        for col, width in zip("ABCDEFGH", (14, 22, 18, 30, 10, 90, 18, 18)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    return buf.getvalue()


# ═══════════════════════ SIDEBAR & LOADING ═══════════════════════
st.title("EBDA & Kobo")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
if "flash_warn" in st.session_state:
    st.warning(st.session_state.pop("flash_warn"))

with st.sidebar:
    st.success(f"👤 {USER}")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    mode = st.selectbox("Scope:", ["Specific Farm (Farm ID)", "By Submission Date", "All Farms"])
    farm_id_input, date_str = None, None
    if mode == "Specific Farm (Farm ID)":
        farm_id_input = st.text_input("Farm ID:")
    elif mode == "By Submission Date":
        date_str = str(st.date_input("Submission Date:"))

    auto_sat = st.checkbox("🛰️ Auto-run Satellite & Tree AI on selected farm", value=True)
    use_ndvi = st.checkbox("Include NDVI in AI check (slower)", value=False)
    if st.button("🔄 Refresh Kobo Data"):
        get_kobo_raw_and_df.clear()
        get_cols_map.clear()
        st.rerun()
    run_btn = stretch(st.button, "🚀 Load Forms", type="primary")

if run_btn:
    if mode == "Specific Farm (Farm ID)" and not farm_id_input:
        st.warning("اكتب Farm ID الأول.")
    else:
        mode_key = "farm" if farm_id_input else "date" if date_str and mode == "By Submission Date" else "all"
        try:
            raw_rows, raw_df = get_kobo_raw_and_df()
            filtered_df = qa.filter_df(raw_df, mode=mode_key, date=date_str, farm=farm_id_input)
            if filtered_df.empty:
                st.warning("⚠️ No forms matched.")
            else:
                cols = get_cols_map()
                recs, total = [], len(filtered_df)
                prog = st.progress(0, text="Loading forms...")
                for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
                    try:
                        rec = qa.audit_row(row, cols, counter=None, use_ai=False)
                    except Exception as e:
                        rec = blank_record(qa.first_value(row, cols["farm_id"]) or f"row{i}", e)
                    rec["_df_index"] = idx
                    rec["_submission_id"] = row.get("_id", "")
                    recs.append(rec
