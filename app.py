from __future__ import annotations

import base64  # noqa: F401  (kept for compatibility)
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

from pyairtable import Api
from pgs_rules import check_rules
from work_divider import run_divide_work_pipeline

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

qa.load_calibration()  # FIX: كان ملف المعايرة عمره ما بيتحمّل من التطبيق


def stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


# ═══════════════════════ LOGIN ═══════════════════════
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
EDIT_COLS = ["Item_Decision", "Suggested_Correction_or_Comment", "Attach_to_Summary"]
EDITOR_COLS = ["Field_Key", "Question_Title", "Submitted_Answer"] + EDIT_COLS
AI_KEYS = ["ai_count", "ai_low", "ai_high", "ai_interior_trees", "veg_pct", "crown_m", "img_age", "ndvi_recent",
           "ai_error", "imagery_date", "zoom_used", "ai_raw", "ai_factor", "verdict", "confidence", "perimeter_m", "boundary_spacing_m"]
STATUS_PENDING = "⏳ Pending Review"
DECISIONS = ["✅ OK", "❌ Reject", "✏️ Modify", "❓ Clarify"]
META_PREFIXES = ("_", "meta/", "formhub/", "__version__", "deviceid", "simserial", "phonenumber", "subscriberid")
MAX_IMG_MB = 15
MAX_VIDEO_MB = 40


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and v != v) else v


def to_bool(x) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in ("true", "1", "yes")


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def xl_safe(v):
    """يمنع Excel formula injection."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


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
    """name / xpath (lowercase) -> (first_label, joined_label)."""
    labels = {}
    try:
        r = requests.get(f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}.json",
                         headers={"Authorization": f"Token {qa.KOBO_TOKEN}"}, timeout=30)
        if r.ok:
            for item in r.json().get("content", {}).get("survey", []):
                name = item.get("name") or item.get("$autoname")
                lbl = item.get("label")
                if not (name and lbl):
                    continue
                if isinstance(lbl, list):
                    parts = [str(x) for x in lbl if x]
                    if not parts:
                        continue
                    entry = (parts[0], " / ".join(parts))
                else:
                    entry = (str(lbl), str(lbl))
                labels.setdefault(name.lower(), entry)
                xp = item.get("$xpath")
                if xp:
                    labels[str(xp).lower()] = entry
    except Exception:
        pass
    return labels


@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def fetch_kobo_media_bytes(url: str, max_mb: int = MAX_IMG_MB) -> bytes | None:
    try:
        with requests.get(url, headers={"Authorization": f"Token {qa.KOBO_TOKEN}"}, timeout=45, stream=True) as r:
            if not r.ok:
                return None
            size = int(r.headers.get("content-length") or 0)
            if size and size > max_mb * 1024 * 1024:
                return None
            return r.content
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


def find_attachment(lookup: dict, fname: str):
    # Kobo بيبدّل المسافات بـ _ في أسماء المرفقات أحياناً
    return lookup.get(fname) or lookup.get(fname.replace(" ", "_"))


def is_image_filename(val) -> bool:
    return isinstance(val, str) and val.strip().lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def is_video_filename(val) -> bool:
    return isinstance(val, str) and val.strip().lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"))


def resolve_label(col_path: str, label_map: dict) -> tuple[str, str]:
    """يرجع (raw_label, display_label).
    raw_label = نص السؤال الصافي (بيستخدمه pgs_rules في المقارنات).
    display_label = اللي بيتعرض للمراجع."""
    clean = re.sub(r"\[\d+\]", "", str(col_path)).lower()
    leaf = clean.split("/")[-1]
    idx = re.findall(r"\[\d+\]", str(col_path))
    prefix = f"{idx[0]} " if idx else ""
    entry = label_map.get(clean) or label_map.get(leaf)
    if entry:
        first, joined = entry
        return first, f"{prefix}{joined} ({leaf})"
    return leaf, str(col_path)


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


def with_general_note(text: str, note: str) -> str:
    return f"[General Note: {note.strip()}]\n{text}" if note.strip() else text


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


def append_to_airtable(rec: dict) -> tuple[bool, str]:
    """يرجع (True/False, رسالة واضحة). بيعمل upsert على farm_id عشان ميكررش الصفوف."""
    try:
        cfg = st.secrets["airtable"]
        api_key, base_id, table_name = cfg["api_key"], cfg["base_id"], cfg["table_name"]
    except Exception:
        return False, "إعدادات [airtable] ناقصة في Secrets (api_key / base_id / table_name)."

    row = {
        "farm_id": str(rec.get("farm_id", "")),
        "submission_id": str(rec.get("submission_id", "")),
        "Form_Status": str(rec.get("Form_Status", "")),
        "Rejected_Items_Count": int(rec.get("Rejected_Items_Count", 0) or 0),
        "Consolidated_Comments": str(rec.get("Consolidated_Comments", "")),
        "General_Auditor_Note": str(rec.get("General_Auditor_Note", "")),
        "Auditor_Name": str(rec.get("Auditor_Name", "")),
        "Reviewed_At": str(rec.get("Reviewed_At", "")),
    }
    try:
        table = Api(api_key, timeout=(10, 30)).table(base_id, table_name)
        result = table.batch_upsert([{"fields": row}], key_fields=["farm_id"], typecast=True)
        if result.get("createdRecords"):
            return True, "تم الإرسال — اتضافت المزرعة كسجل جديد."
        return True, "تم الإرسال — اتحدّث السجل الموجود للمزرعة."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        code = getattr(e.response, "status_code", "?")
        hint = {
            401: "الـ API key غلط أو منتهي.",
            403: "التوكن معندوش صلاحية data.records:write أو مش مدّي وصول للـ Base.",
            404: "الـ base_id أو table_name غلط.",
            422: "اسم عمود غلط أو نوع بيانات مش مطابق في الجدول.",
        }.get(code, "")
        return False, f"Airtable رفض الطلب ({code}). {hint} {body}"
    except requests.exceptions.Timeout:
        return False, "انتهت المهلة — Airtable مردّش. جرّب تاني."
    except requests.exceptions.ConnectionError:
        return False, "مشكلة اتصال بالإنترنت أو Airtable مش متاح."
    except Exception as e:
        return False, f"خطأ غير متوقع: {e}"


def save_master_decision(rec: dict):
    """حفظ محلي (CSV) فقط — الإرسال لـ Airtable بيتم من الزرار المخصص."""
    fid = str(rec["farm_id"])
    with get_lock():
        df = _csv_read(MASTER_LOG_FILE, MASTER_COLS)
        if not df.empty:
            df = df[df["farm_id"].astype(str) != fid]
        _csv_write(MASTER_LOG_FILE, pd.concat([df, pd.DataFrame([rec])], ignore_index=True))


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
        "Comment": df["Consolidated_Comments"].map(xl_safe),
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
    st.markdown("### ⚡ Work Management")
    if st.button("Divide Work (Google Sheets)", type="primary"):
        with st.status("Running Distribution Pipeline...", expanded=True) as status:
            log_container = st.empty()
            logs = []

            def ui_logger(msg):
                logs.append(str(msg))
                log_container.code("\n".join(logs))
            try:
                run_divide_work_pipeline(ui_logger)
                status.update(label="Distribution Complete!", state="complete", expanded=False)
                st.success("✅ تم تحديث المهام على Google Sheets بنجاح")
            except Exception as e:
                status.update(label="Error Occurred!", state="error")
                st.error(f"خطأ أثناء توزيع المهام: {e}")

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
            cols = get_cols_map()
            filtered_df = qa.filter_df(raw_df, mode=mode_key, date=date_str, farm=farm_id_input, cols=cols)
            if filtered_df.empty:
                st.warning("⚠️ No forms matched.")
            else:
                recs, total = [], len(filtered_df)
                prog = st.progress(0, text="Loading forms...")
                for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
                    try:
                        rec = qa.audit_row(row, cols, counter=None, use_ai=False)
                    except Exception as e:
                        rec = blank_record(qa.first_value(row, cols["farm_id"]) or f"row{i}", e)
                    rec["_df_index"] = idx
                    rec["_submission_id"] = row.get("_id", "")
                    recs.append(rec)
                    if i % 20 == 0 or i == total:
                        prog.progress(i / total, text=f"Loaded {i}/{total}")
                prog.empty()

                qa.cross_checks(recs)
                for r in recs:
                    r["flags"] = "; ".join(r.get("flags", []))

                res_df = pd.DataFrame(recs)
                res_df["farm_id"] = res_df["farm_id"].astype(str)

                keep = MASTER_COLS[:1] + MASTER_COLS[2:]
                ml = _csv_read(MASTER_LOG_FILE, MASTER_COLS)
                if not ml.empty:
                    ml = ml[keep].drop_duplicates("farm_id", keep="last")
                    ml["farm_id"] = ml["farm_id"].astype(str)
                    res_df = res_df.merge(ml, on="farm_id", how="left")
                else:
                    for c in keep[1:]:
                        res_df[c] = ""

                res_df["Form_Status"] = res_df["Form_Status"].fillna(STATUS_PENDING).replace("", STATUS_PENDING)
                res_df["Rejected_Items_Count"] = pd.to_numeric(res_df["Rejected_Items_Count"], errors="coerce").fillna(0).astype(int)
                for c in ("Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At"):
                    res_df[c] = res_df[c].fillna("").astype(str)

                for k in [k for k in st.session_state if k.startswith("fd_")]:
                    del st.session_state[k]
                st.session_state.update(results=res_df, raw_rows=raw_rows, flat_df=raw_df, ai_imgs={})
                st.session_state.pop("xlsx", None)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.exception(e)

# ═══════════════════════ MAIN ═══════════════════════
if "results" in st.session_state:
    res = st.session_state["results"]
    raw_rows = st.session_state["raw_rows"]
    flat_df = st.session_state["flat_df"]
    label_map = get_kobo_labels()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Loaded", len(res))
    k2.metric("Pending ⏳", int((res["Form_Status"] == STATUS_PENDING).sum()))
    k3.metric("Approved 👍", int((res["Form_Status"] == "✅ Approved Entire Form").sum()))
    k4.metric("Conditional ⚠️", int((res["Form_Status"] == "⚠️ Conditional (Modifications Required)").sum()))
    k5.metric("Rejected 🚫", int((res["Form_Status"] == "🚫 REJECTED ENTIRE FORM").sum()))

    st.divider()

    sel_info = {r.farm_id: f"Farm {r.farm_id}  |  {r.engineer}  |  {r.Form_Status}" for r in res.itertuples()}
    col_sel, col_dl = st.columns([2, 1])
    with col_sel:
        selected_farm = st.selectbox("🎯 Select Farm:", list(sel_info.keys()), format_func=lambda x: sel_info.get(x, x))
    with col_dl:
        only_rev = st.checkbox("Excel: reviewed forms only", value=True)
        if st.button("📊 Prepare Excel"):
            st.session_state["xlsx"] = build_reviews_xlsx(res, only_rev)
        if "xlsx" in st.session_state:
            stretch(st.download_button, "📥 Download Excel", st.session_state["xlsx"],
                    file_name=f"farm_reviews_{datetime.now():%Y%m%d_%H%M}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if selected_farm:
        audit_row_idx = res.index[res["farm_id"].astype(str) == selected_farm][0]
        audit_rec = res.loc[audit_row_idx]
        flat_submission = flat_df.loc[audit_rec["_df_index"]]
        att_lookup = build_attachment_lookup(raw_rows[audit_rec["_df_index"]])
        sub_id = str(audit_rec.get("_submission_id", ""))

        ai_img = st.session_state.get("ai_imgs", {}).get(selected_farm)
        ac = audit_rec.get("ai_count")
        ai_ran = (selected_farm in st.session_state.get("ai_imgs", {})) or bool(audit_rec.get("ai_error")) or not (ac is None or ac != ac)
        has_valid_gps = not str(audit_rec.get("area_check", "")).startswith("❌ Calculation Failed")

        if auto_sat and not ai_ran and has_valid_gps and sub_id:
            with st.spinner(f"🛰️ Fetching satellite imagery, drawing farm boundary & detecting trees for Farm {selected_farm}..."):
                try:
                    rec_ai, img_b = run_ai_for_farm(sub_id, use_ndvi)
                    apply_ai(audit_row_idx, rec_ai)
                    st.session_state.setdefault("ai_imgs", {})[selected_farm] = img_b or b""
                    audit_rec = st.session_state["results"].loc[audit_row_idx]
                    ai_img = img_b
                    ai_ran = True
                except Exception as e:
                    st.session_state.setdefault("ai_imgs", {})[selected_farm] = b""
                    st.warning(f"Satellite AI warning: {e}")

        tab_form, tab_reviews = st.tabs(["📋 Form & Auditor", "✅ Reviewed Forms"])

        with tab_form:
            top_left, top_right = st.columns([1.3, 1])

            with top_left:
                st.info(
                    f"**🤖 Boundary Verdict ({fmt(audit_rec.get('confidence', ''))}):** {audit_rec['verdict']}\n\n"
                    f"- **Farm Perimeter (محيط المزرعة):** `{fmt(audit_rec.get('perimeter_m'))} m` | **Area:** entered `{fmt(audit_rec.get('area_entered'))}` vs GPS `{fmt(audit_rec.get('area_gps'))}` Feddan ({fmt(audit_rec.get('area_check'))})\n"
                    f"- **Boundary Trees (أشجار الحدود):** engineer `{fmt(audit_rec.get('engineer_count'))}` ({fmt(audit_rec.get('tree_details'))}) | **Boundary AI:** `{fmt(audit_rec.get('ai_count'))}` | **Perimeter Forecast:** `{fmt(audit_rec.get('forecast'))}` trees (at `{fmt(audit_rec.get('boundary_spacing_m'))}m` spacing)\n"
                    f"- **NDVI:** `{fmt(audit_rec.get('ndvi_recent'))}` | **Boundary Vegetation Cover:** `{fmt(audit_rec.get('veg_pct'))}%`"
                )
                if audit_rec.get("gps_warnings"):
                    st.warning(f"**Geometry:** {audit_rec['gps_warnings']}")
                if audit_rec.get("flags"):
                    st.error(f"**Flags:** {audit_rec['flags']}")

                if st.button("🔁 Re-run satellite AI" if ai_ran else "🛰️ Run satellite AI check", key=f"ai_{selected_farm}"):
                    with st.spinner("Running DeepForest & drawing farm boundary..."):
                        try:
                            run_ai_for_farm.clear()
                            rec_ai, img_b = run_ai_for_farm(sub_id, use_ndvi)
                            apply_ai(audit_row_idx, rec_ai)
                            st.session_state.setdefault("ai_imgs", {})[selected_farm] = img_b or b""
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI check failed: {e}")

            with top_right:
                if ai_img:
                    try:
                        stretch(
                            st.image,
                            Image.open(io.BytesIO(ai_img)),
                            caption=f"🛰️ Farm {selected_farm} | Boundary (Yellow) & Detected Boundary Trees (Red Boxes: {fmt(audit_rec.get('ai_count'))})"
                        )
                    except Exception:
                        st.caption("Could not display satellite image.")
                elif not has_valid_gps:
                    st.warning("⚠️ No valid GPS polygon recorded in Kobo for this farm to fetch satellite imagery.")
                else:
                    st.caption("Click '🛰️ Run satellite AI check' to load satellite boundary & tree detections.")

            st.divider()

            saved = get_field_decisions(selected_farm)
            smap = saved.drop_duplicates("Field_Key", keep="last").set_index("Field_Key").to_dict(orient="index") if not saved.empty else {}

            area_bad = str(audit_rec.get("area_check", "")).startswith("❌")
            verdict_bad = "❌" in str(audit_rec.get("verdict", ""))
            qa_items = [
                {"Field_Key": "_AI_AREA_CHECK", "Question_Title": "🛰️ [Check] Farm Area & Perimeter (Entered vs GPS)",
                 "Submitted_Answer": f"Perimeter: {fmt(audit_rec.get('perimeter_m'))}m | Entered: {fmt(audit_rec.get('area_entered'))} | GPS: {fmt(audit_rec.get('area_gps'))} ({fmt(audit_rec.get('area_check'))})",
                 "Item_Decision": "❌ Reject" if area_bad else "✅ OK",
                 "Suggested_Correction_or_Comment": f"Area mismatch: GPS shows {audit_rec.get('area_gps')} Feddan (Perimeter: {audit_rec.get('perimeter_m')}m)" if area_bad else "",
                 "Attach_to_Summary": bool(area_bad)},
                {"Field_Key": "_AI_TREE_CHECK", "Question_Title": "🛰️ [Check] Boundary Tree Count (Engineer vs AI vs Perimeter Forecast)",
                 "Submitted_Answer": f"Eng: {fmt(audit_rec.get('engineer_count'))} | Boundary AI: {fmt(audit_rec.get('ai_count'))} | Perimeter Forecast: {fmt(audit_rec.get('forecast'))}",
                 "Item_Decision": "❌ Reject" if verdict_bad else "✅ OK",
                 "Suggested_Correction_or_Comment": str(audit_rec.get("verdict", "")) if verdict_bad else "",
                 "Attach_to_Summary": bool(verdict_bad)},
            ]
            media_questions = []
            rules_items = []  # FIX: جدول القواعد مستقل عن فلاتر العرض (only_photos / show_empty)

            f1, f2, f3, f4 = st.columns(4)
            show_empty = f1.checkbox("Show empty fields", value=False)
            only_photos = f2.checkbox("📸🎥 Only photo/video questions", value=False)
            load_photos = f3.checkbox("🖼️ Load photos", value=True)
            load_videos = f4.checkbox("🎥 Load videos (heavy)", value=False)

            for col_name, val in flat_submission.items():
                if str(col_name).startswith(META_PREFIXES):
                    continue
                present = qa.present(val)
                if not present and not show_empty:
                    continue
                val_str = str(val).strip() if present else "— (Empty)"
                fname = val_str.split("/")[-1].strip()
                att_info = find_attachment(att_lookup, fname) if present else None
                mime = att_info["mimetype"] if att_info else ""
                is_photo = att_info is not None and (is_image_filename(fname) or "image" in mime)
                is_video = att_info is not None and (is_video_filename(fname) or "video" in mime)

                raw_label, q_label = resolve_label(col_name, label_map)

                if present:
                    rules_items.append({"Raw_Label": raw_label, "Submitted_Answer": val_str,
                                        "Attachment_Link": att_info["full_url"] if att_info else ""})

                if only_photos and not (is_photo or is_video):
                    continue

                if load_photos and att_info and (is_photo or is_video):
                    b = None
                    if is_photo:
                        b = fetch_kobo_media_bytes(att_info["preview_url"], MAX_IMG_MB)
                    elif load_videos:
                        b = fetch_kobo_media_bytes(att_info["full_url"], MAX_VIDEO_MB)
                    media_questions.append({"Field_Key": str(col_name), "Question_Title": q_label, "Filename": fname,
                                           "Type": "image" if is_photo else "video", "Bytes": b})

                icon = "📸 " if is_photo else "🎥 " if is_video else ""
                qa_items.append({
                    "Field_Key": str(col_name),
                    "Question_Title": icon + q_label,
                    "Submitted_Answer": val_str,
                    "Item_Decision": "✅ OK",
                    "Suggested_Correction_or_Comment": "", "Attach_to_Summary": False})

            # ── PGS RULES ENGINE CHECK ──
            st.divider()
            st.markdown("### 📋 PGS Business Rules Audit")

            farm_poly = audit_rec.get("_poly")
            poly_latlon = None
            if farm_poly is not None and getattr(farm_poly, "geom_type", "") == "Polygon":
                # FIX: shapely (lon, lat) → pgs_rules بيتوقع (lat, lon)
                poly_latlon = [(y, x) for x, y in farm_poly.exterior.coords]

            rule_issues = check_rules(
                table=rules_items,
                calc_area=audit_rec.get("area_gps"),
                farm_polygon=poly_latlon,
            )

            if rule_issues:
                error_list = "\n".join([f"- {iss}" for iss in rule_issues])
                st.error(f"⚠️ **تم اكتشاف تناقضات إدارية في الاستمارة:**\n{error_list}")
            else:
                st.success("✅ **فحص القواعد:** جميع المرفقات والأرقام الأساسية سليمة ومطابقة للمساحة.")

            st.divider()

            # ── PREPARE DATAFRAME ──
            form_qa_df = pd.DataFrame(qa_items)

            if smap:
                for i, r in form_qa_df.iterrows():
                    s = smap.get(r["Field_Key"])
                    if s:
                        dec = s.get("Item_Decision", "✅ OK")
                        form_qa_df.at[i, "Item_Decision"] = dec if isinstance(dec, str) and dec else "✅ OK"
                        cm = s.get("Suggested_Correction_or_Comment", "")
                        form_qa_df.at[i, "Suggested_Correction_or_Comment"] = "" if pd.isna(cm) else str(cm)
                        form_qa_df.at[i, "Attach_to_Summary"] = to_bool(s.get("Attach_to_Summary", False))

            # ── INTERACTIVE PHOTO & VIDEO AUDIT CARDS ──
            media_overrides = {}
            if media_questions:
                with st.expander(f"📸🎥 Photos & Videos Audit Cards ({len(media_questions)}) — تقدر ترفض أو تقبل وتكتب سبب الرفض تحت كل صورة أو فيديو", expanded=True):
                    mc = st.columns(3)
                    for i, mq in enumerate(media_questions):
                        fk = mq["Field_Key"]
                        sv = smap.get(fk, {})
                        with mc[i % 3]:
                            with st.container(border=True):
                                st.markdown(f"**{'🎥' if mq['Type'] == 'video' else '📸'} {mq['Question_Title']}**")
                                if mq["Bytes"] is None:
                                    st.caption(f"⏭️ {mq['Filename']} — " + ("مش متحمّل (فعّل Load videos)" if mq["Type"] == "video" else "تعذّر تحميل الصورة"))
                                elif mq["Type"] == "image":
                                    try:
                                        stretch(st.image, Image.open(io.BytesIO(mq["Bytes"])), caption=mq["Filename"])
                                    except Exception:
                                        st.caption(f"Cannot display {mq['Filename']}")
                                else:
                                    st.video(mq["Bytes"])
                                    st.caption(f"Video: {mq['Filename']}")

                                cur_d = sv.get("Item_Decision", "✅ OK")
                                cur_d = cur_d if cur_d in DECISIONS else "✅ OK"
                                m_dec = st.radio("Decision:", DECISIONS, index=DECISIONS.index(cur_d), horizontal=True, key=f"mdec_{selected_farm}_{fk}")
                                old_c = sv.get("Suggested_Correction_or_Comment", "")
                                m_com = st.text_input("سبب الرفض / تعليق:", value="" if pd.isna(old_c) else str(old_c), key=f"mcom_{selected_farm}_{fk}")
                                m_att = st.checkbox("📎 إرفاق بالتقرير النهائي", value=to_bool(sv.get("Attach_to_Summary", False)) or (m_dec == "❌ Reject"), key=f"matt_{selected_farm}_{fk}")
                                media_overrides[fk] = (m_dec, m_com, m_att)

            for i, r in form_qa_df.iterrows():
                fk = r["Field_Key"]
                if fk in media_overrides:
                    form_qa_df.at[i, "Item_Decision"] = media_overrides[fk][0]
                    form_qa_df.at[i, "Suggested_Correction_or_Comment"] = media_overrides[fk][1]
                    form_qa_df.at[i, "Attach_to_Summary"] = media_overrides[fk][2]

            form_qa_df["Attach_to_Summary"] = form_qa_df["Attach_to_Summary"].astype(bool)
            form_qa_df["Suggested_Correction_or_Comment"] = form_qa_df["Suggested_Correction_or_Comment"].fillna("").astype(str)

            # الصفوف اللي ليها كارت (صورة/فيديو) بتتعدّل من الكارت بس، عشان مايبقاش فيه مصدرين للقرار
            editor_base = form_qa_df[~form_qa_df["Field_Key"].isin(list(media_overrides.keys()))]

            search_q = st.text_input("🔎 Search (Arabic / English):", "")
            if search_q.strip():
                m = (editor_base["Question_Title"].str.contains(search_q, case=False, na=False, regex=False)
                     | editor_base["Submitted_Answer"].str.contains(search_q, case=False, na=False, regex=False))
                display_df = editor_base[m].copy()
            else:
                display_df = editor_base.copy()

            edited = stretch(
                st.data_editor, display_df[EDITOR_COLS], hide_index=True, height=520,
                disabled=["Field_Key", "Question_Title", "Submitted_Answer"],
                column_config={
                    "Field_Key": None,
                    "Question_Title": st.column_config.TextColumn("📌 Question", width="large"),
                    "Submitted_Answer": st.column_config.TextColumn("✍️ Answer", width="medium"),
                    "Item_Decision": st.column_config.SelectboxColumn(
                        "⚖️ Verdict", options=DECISIONS, required=True, width="small"),
                    "Suggested_Correction_or_Comment": st.column_config.TextColumn("📝 Comment", width="large"),
                    "Attach_to_Summary": st.column_config.CheckboxColumn("📎 Attach?", default=False, width="small"),
                },
                key=f"ed_{selected_farm}_{short_hash(search_q)}_{int(show_empty)}_{int(only_photos)}_{int(load_photos)}_{int(load_videos)}",
            )

            # FIX: بندمج اللي اتعدّل في الجدول مع باقي الصفوف (البحث كان بيخلّي الحفظ يمسح القرارات المخفية)
            final_df = form_qa_df.copy()
            if not edited.empty:
                for c in EDIT_COLS:
                    final_df.loc[edited.index, c] = edited[c]
            final_df["Suggested_Correction_or_Comment"] = final_df["Suggested_Correction_or_Comment"].fillna("").astype(str)
            final_df["Attach_to_Summary"] = final_df["Attach_to_Summary"].fillna(False).astype(bool)

            merged_lines, rejected, modified = [], 0, 0
            if not final_df.empty:
                rejected = int((final_df["Item_Decision"] == "❌ Reject").sum())
                modified = int((final_df["Item_Decision"] == "✏️ Modify").sum())
                flagged = final_df["Item_Decision"].isin(["❌ Reject", "✏️ Modify", "❓ Clarify"])
                pick = final_df["Attach_to_Summary"] | (flagged & (final_df["Suggested_Correction_or_Comment"].str.strip() != ""))
                for _, r in final_df[pick].iterrows():
                    if not r["Attach_to_Summary"] and r["Item_Decision"] == "✅ OK":
                        continue
                    q_short = str(r["Question_Title"]).split("(")[0].strip()
                    ans = str(r["Submitted_Answer"])
                    ans = ans[:60] + "..." if len(ans) > 60 else ans
                    comm = r["Suggested_Correction_or_Comment"].strip() or "Flagged by auditor"
                    merged_lines.append(f"• [{r['Item_Decision']}] {q_short} (Ans: '{ans}') ➔ {comm}")

            st.divider()
            st.markdown("### 🗂️ Final Decision & Single Comment")
            c_left, c_right = st.columns([1.4, 1])
            with c_left:
                st.markdown(f"**Merged comment** ({len(merged_lines)} items | {rejected} rejected | {modified} modified)")
                default_merged = "\n".join(merged_lines) if merged_lines else str(audit_rec.get("Consolidated_Comments", ""))
                final_text = st.text_area("Edit before saving:", value=default_merged, height=180,
                                          key=f"merged_{selected_farm}_{short_hash(default_merged)}")
                general_note = st.text_input("📌 General note / rejection reason:",
                                             value=str(audit_rec.get("General_Auditor_Note", "")),
                                             key=f"note_{selected_farm}")
            with c_right:
                choices = [STATUS_PENDING, "✅ Approved Entire Form",
                           "⚠️ Conditional (Modifications Required)", "🚫 REJECTED ENTIRE FORM"]
                cur = str(audit_rec.get("Form_Status", STATUS_PENDING))
                cur = cur if cur in choices else STATUS_PENDING
                final_status = st.radio("Final Status:", choices, index=choices.index(cur), key=f"radio_{selected_farm}")
                sync_kobo = st.checkbox("🔗 Sync status to KoboToolbox", value=False)
                b1, b2 = st.columns(2)
                save_btn = stretch(b1.button, "💾 Save", type="primary")
                reject_btn = stretch(b2.button, "🚫 REJECT NOW")

                # ── زرار Airtable ──
                air_btn = stretch(st.button, "☁️ Send to Airtable", key=f"air_{selected_farm}")
                if air_btn:
                    if final_status == STATUS_PENDING:
                        st.warning("⚠️ اختار Final Status غير Pending قبل الإرسال.")
                    else:
                        with st.spinner("جاري الإرسال إلى Airtable..."):
                            ok, air_msg = append_to_airtable({
                                "farm_id": selected_farm, "submission_id": sub_id,
                                "Form_Status": final_status, "Rejected_Items_Count": rejected,
                                "Consolidated_Comments": with_general_note(final_text, general_note),
                                "General_Auditor_Note": general_note, "Auditor_Name": USER,
                                "Reviewed_At": datetime.now().strftime("%Y-%m-%d %H:%M")})
                        (st.success if ok else st.error)(("✅ " if ok else "❌ فشل الإرسال: ") + air_msg)

                if save_btn or reject_btn:
                    chosen = "🚫 REJECTED ENTIRE FORM" if reject_btn else final_status
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    report = with_general_note(final_text, general_note)

                    for col, v in (("Form_Status", chosen), ("Rejected_Items_Count", rejected),
                                   ("Consolidated_Comments", report), ("General_Auditor_Note", general_note),
                                   ("Auditor_Name", USER), ("Reviewed_At", now_ts)):
                        st.session_state["results"].loc[audit_row_idx, col] = v

                    if not final_df.empty:
                        save_field_decisions(selected_farm, final_df[[
                            "Field_Key", "Question_Title", "Submitted_Answer", "Item_Decision",
                            "Suggested_Correction_or_Comment", "Attach_to_Summary"]])
                        st.session_state.pop(f"fd_{selected_farm}", None)

                    # ── حفظ محلي (CSV) ──
                    save_master_decision({
                        "farm_id": selected_farm, "submission_id": sub_id, "Form_Status": chosen,
                        "Rejected_Items_Count": rejected, "Consolidated_Comments": report,
                        "General_Auditor_Note": general_note, "Auditor_Name": USER, "Reviewed_At": now_ts})
                    st.session_state.pop("xlsx", None)

                    msg = f"✅ Saved! Farm `{selected_farm}` → **{chosen}**"
                    if sync_kobo and sub_id:
                        code = ("validation_status_approved" if "Approved" in chosen
                                else "validation_status_not_approved" if "REJECTED" in chosen
                                else "validation_status_on_hold")
                        if update_kobo_validation_status(sub_id, code):
                            msg += " — synced to Kobo ✔"
                        else:
                            st.session_state["flash_warn"] = "⚠️ Saved, but Kobo sync failed."
                    st.session_state["flash"] = msg
                    st.rerun()

        with tab_reviews:
            rev = res[res["Form_Status"] != STATUS_PENDING]
            st.caption(f"{len(rev)} reviewed of {len(res)} loaded — اضغط 'Prepare Excel' فوق لتحميلها في ملف واحد.")
            if rev.empty:
                st.info("لسه مفيش استمارات اتراجعت.")
            else:
                show = rev[["farm_id", "engineer", "Form_Status", "Rejected_Items_Count",
                            "Consolidated_Comments", "Auditor_Name", "Reviewed_At"]]
                stretch(st.dataframe, show)
