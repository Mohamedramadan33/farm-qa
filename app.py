from __future__ import annotations

import base64
import hashlib
import hmac
import io
import os
import re
import threading
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Farm QA Auditor", layout="wide", page_icon="🛰️")

try:
    if "KOBO_TOKEN" in st.secrets:
        os.environ["KOBO_TOKEN"] = str(st.secrets["KOBO_TOKEN"])
except Exception:
    pass

import farm_qa_core as qa  # noqa: E402

if not qa.KOBO_TOKEN:
    st.error("❌ KOBO_TOKEN غير موجود. ضيفه في App settings → Secrets.")
    st.stop()


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
    st.title("🔐 Farm QA — Login")
    if not users:
        st.error("مفيش مستخدمين متسجلين. ضيف في Secrets:\n\n[users]\nاسمك = \"الباسورد\"")
        st.stop()
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
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
AI_KEYS = ["ai_count", "ai_low", "ai_high", "veg_pct", "crown_m", "img_age", "ndvi_recent",
           "ai_error", "imagery_date", "zoom_used", "ai_raw", "ai_factor", "verdict", "confidence"]
STATUS_PENDING = "⏳ Pending Review"
META_PREFIXES = ("_", "meta/", "formhub/", "__version__", "deviceid", "simserial", "phonenumber", "subscriberid")


def stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


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
def fetch_kobo_image_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers={"Authorization": f"Token {qa.KOBO_TOKEN}"}, timeout=30)
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
        rec = qa.audit_row(row, get_cols_map(), counter=get_tree_counter(), use_ai=True,
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
        dl = a.get("download_medium_url") or a.get("download_small_url") or a.get("download_url")
        if short and dl:
            lookup[short] = dl
            lookup[full] = dl
    return lookup


def is_image_filename(val) -> bool:
    return isinstance(val, str) and val.strip().lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


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


# ═══════════════════════ STORAGE (CSV على السيرفر) ═══════════════════════
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


def save_master_decision(rec: dict):
    fid = str(rec["farm_id"])
    with get_lock():
        df = _csv_read(MASTER_LOG_FILE, MASTER_COLS)
        if not df.empty:
            df = df[df["farm_id"].astype(str) != fid]
        _csv_write(MASTER_LOG_FILE, pd.concat([df, pd.DataFrame([rec])], ignore_index=True))


def save_field_decisions(farm_id: str, field_df: pd.DataFrame):
    """بنخزّن بس البنود اللي فيها قرار/تعليق."""
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
st.title("🛰️ Farm QA & Form Auditor")

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

        tab_form, tab_reviews = st.tabs(["📋 Form & Auditor", "✅ Reviewed Forms"])

        with tab_form:
            top_left, top_right = st.columns([1.5, 1])
            ai_img = st.session_state.get("ai_imgs", {}).get(selected_farm)
            ac = audit_rec.get("ai_count")
            ai_ran = bool(ai_img) or bool(audit_rec.get("ai_error")) or not (ac is None or ac != ac)

            with top_left:
                st.info(
                    f"**🤖 Verdict ({fmt(audit_rec.get('confidence', ''))}):** {audit_rec['verdict']}\n\n"
                    f"- **Area:** entered `{fmt(audit_rec.get('area_entered'))}` | GPS `{fmt(audit_rec.get('area_gps'))}` Feddan ({fmt(audit_rec.get('area_check'))})\n"
                    f"- **Trees:** engineer `{fmt(audit_rec.get('engineer_count'))}` ({fmt(audit_rec.get('tree_details'))}) | AI `{fmt(audit_rec.get('ai_count'))}` | forecast `{fmt(audit_rec.get('forecast'))}`\n"
                    f"- **NDVI:** `{fmt(audit_rec.get('ndvi_recent'))}` | **Vegetation:** `{fmt(audit_rec.get('veg_pct'))}%`"
                )
                if audit_rec.get("gps_warnings"):
                    st.warning(f"**Geometry:** {audit_rec['gps_warnings']}")
                if audit_rec.get("flags"):
                    st.error(f"**Flags:** {audit_rec['flags']}")

                if st.button("🔁 Re-run satellite AI" if ai_ran else "🛰️ Run satellite AI check", key=f"ai_{selected_farm}"):
                    with st.spinner("Running DeepForest (first run downloads the model)..."):
                        try:
                            if ai_ran:
                                run_ai_for_farm.clear()
                            rec_ai, img_b = run_ai_for_farm(sub_id, use_ndvi)
                            apply_ai(audit_row_idx, rec_ai)
                            st.session_state.setdefault("ai_imgs", {})[selected_farm] = img_b or b""
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI check failed: {e}")
                            st.exception(e)

            with top_right:
                if ai_img:
                    try:
                        stretch(st.image, Image.open(io.BytesIO(ai_img)), caption=f"Farm {selected_farm} — AI: {fmt(audit_rec.get('ai_count'))}")
                    except Exception:
                        st.caption("Could not display image.")
                else:
                    st.caption("Satellite image appears after the AI check.")

            st.divider()

            area_bad = str(audit_rec.get("area_check", "")).startswith("❌")
            verdict_bad = "❌" in str(audit_rec.get("verdict", ""))
            qa_items = [
                {"Field_Key": "_AI_AREA_CHECK", "Question_Title": "🛰️ [Check] Farm Area (Entered vs GPS)",
                 "Submitted_Answer": f"Entered: {fmt(audit_rec.get('area_entered'))} | GPS: {fmt(audit_rec.get('area_gps'))} ({fmt(audit_rec.get('area_check'))})",
                 "Photo_Preview": "", "Item_Decision": "❌ Reject" if area_bad else "✅ OK",
                 "Suggested_Correction_or_Comment": f"Area mismatch: GPS shows {audit_rec.get('area_gps')} Feddan" if area_bad else "",
                 "Attach_to_Summary": bool(area_bad)},
                {"Field_Key": "_AI_TREE_CHECK", "Question_Title": "🛰️ [Check] Tree Count (Engineer vs AI vs Forecast)",
                 "Submitted_Answer": f"Eng: {fmt(audit_rec.get('engineer_count'))} | AI: {fmt(audit_rec.get('ai_count'))} | Forecast: {fmt(audit_rec.get('forecast'))}",
                 "Photo_Preview": "", "Item_Decision": "❌ Reject" if verdict_bad else "✅ OK",
                 "Suggested_Correction_or_Comment": str(audit_rec.get("verdict", "")) if verdict_bad else "",
                 "Attach_to_Summary": bool(verdict_bad)},
            ]
            photo_questions = []

            f1, f2, f3 = st.columns(3)
            show_empty = f1.checkbox("Show empty fields", value=False)
            only_photos = f2.checkbox("📸 Only photo questions", value=False)
            load_photos = f3.checkbox("🖼️ Load photos (slower)", value=False)

            for col_name, val in flat_submission.items():
                if str(col_name).startswith(META_PREFIXES):
                    continue
                present = qa.present(val)
                if not present and not show_empty:
                    continue
                val_str = str(val).strip() if present else "— (Empty)"
                fname = val_str.split("/")[-1].strip()
                is_photo = is_image_filename(fname) and fname in att_lookup
                if only_photos and not is_photo:
                    continue
                q_label = resolve_label(col_name, label_map)
                data_uri = ""
                if is_photo and load_photos:
                    b = fetch_kobo_image_bytes(att_lookup[fname])
                    if b:
                        data_uri = "data:image/jpeg;base64," + base64.b64encode(b).decode()
                        photo_questions.append({"Question_Title": q_label, "Filename": fname, "Bytes": b})
                qa_items.append({
                    "Field_Key": str(col_name), "Question_Title": ("📸 " + q_label) if is_photo else q_label,
                    "Submitted_Answer": val_str, "Photo_Preview": data_uri, "Item_Decision": "✅ OK",
                    "Suggested_Correction_or_Comment": "", "Attach_to_Summary": False})

            form_qa_df = pd.DataFrame(qa_items)

            saved = get_field_decisions(selected_farm)
            if not saved.empty:
                smap = saved.drop_duplicates("Field_Key", keep="last").set_index("Field_Key").to_dict(orient="index")
                for i, r in form_qa_df.iterrows():
                    s = smap.get(r["Field_Key"])
                    if s:
                        dec = s.get("Item_Decision", "✅ OK")
                        form_qa_df.at[i, "Item_Decision"] = dec if isinstance(dec, str) and dec else "✅ OK"
                        cm = s.get("Suggested_Correction_or_Comment", "")
                        form_qa_df.at[i, "Suggested_Correction_or_Comment"] = "" if pd.isna(cm) else str(cm)
                        form_qa_df.at[i, "Attach_to_Summary"] = to_bool(s.get("Attach_to_Summary", False))

            search_q = st.text_input("🔎 Search (Arabic / English):", "")
            if search_q.strip():
                m = (form_qa_df["Question_Title"].str.contains(search_q, case=False, na=False, regex=False)
                     | form_qa_df["Submitted_Answer"].str.contains(search_q, case=False, na=False, regex=False))
                display_df = form_qa_df[m].copy()
            else:
                display_df = form_qa_df.copy()

            edited = stretch(
                st.data_editor, display_df, hide_index=True, height=520,
                disabled=["Field_Key", "Question_Title", "Submitted_Answer", "Photo_Preview"],
                column_config={
                    "Field_Key": None,
                    "Question_Title": st.column_config.TextColumn("📌 Question", width="large"),
                    "Submitted_Answer": st.column_config.TextColumn("✍️ Answer", width="medium"),
                    "Photo_Preview": st.column_config.ImageColumn("📷 Photo", width="medium") if load_photos else None,
                    "Item_Decision": st.column_config.SelectboxColumn(
                        "⚖️ Verdict", options=["✅ OK", "❌ Reject", "✏️ Modify", "❓ Clarify"], required=True, width="small"),
                    "Suggested_Correction_or_Comment": st.column_config.TextColumn("📝 Comment", width="large"),
                    "Attach_to_Summary": st.column_config.CheckboxColumn("📎 Attach?", default=False, width="small"),
                },
                key=f"ed_{selected_farm}_{short_hash(search_q)}_{int(show_empty)}{int(only_photos)}{int(load_photos)}",
            )

            if photo_questions:
                with st.expander(f"🖼️ Photos ({len(photo_questions)})"):
                    pc = st.columns(3)
                    for i, pq in enumerate(photo_questions):
                        with pc[i % 3]:
                            st.markdown(f"**📸 {pq['Question_Title']}**")
                            try:
                                stretch(st.image, Image.open(io.BytesIO(pq["Bytes"])), caption=pq["Filename"])
                            except Exception:
                                st.caption(f"Cannot display {pq['Filename']}")

            merged_lines, rejected, modified = [], 0, 0
            if not edited.empty:
                ef = edited.copy()
                ef["Suggested_Correction_or_Comment"] = ef["Suggested_Correction_or_Comment"].fillna("").astype(str)
                ef["Attach_to_Summary"] = ef["Attach_to_Summary"].fillna(False).astype(bool)
                rejected = int((ef["Item_Decision"] == "❌ Reject").sum())
                modified = int((ef["Item_Decision"] == "✏️ Modify").sum())
                flagged = ef["Item_Decision"].isin(["❌ Reject", "✏️ Modify", "❓ Clarify"])
                for _, r in ef[ef["Attach_to_Summary"] | (flagged & (ef["Suggested_Correction_or_Comment"].str.strip() != ""))].iterrows():
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

                if save_btn or reject_btn:
                    chosen = "🚫 REJECTED ENTIRE FORM" if reject_btn else final_status
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    report = final_text
                    if general_note.strip():
                        report = f"[General Note: {general_note.strip()}]\n" + report

                    for col, v in (("Form_Status", chosen), ("Rejected_Items_Count", rejected),
                                   ("Consolidated_Comments", report), ("General_Auditor_Note", general_note),
                                   ("Auditor_Name", USER), ("Reviewed_At", now_ts)):
                        st.session_state["results"].loc[audit_row_idx, col] = v

                    if not edited.empty:
                        save_field_decisions(selected_farm, edited[[
                            "Field_Key", "Question_Title", "Submitted_Answer", "Item_Decision",
                            "Suggested_Correction_or_Comment", "Attach_to_Summary"]])
                        st.session_state.pop(f"fd_{selected_farm}", None)

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
