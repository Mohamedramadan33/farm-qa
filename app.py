from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ── لازم أول أمر Streamlit ──
st.set_page_config(page_title="Enterprise Farm QA & Form Auditor", layout="wide", page_icon="🛰️")

# ── الـ Token من Secrets (Streamlit Cloud: App settings → Secrets) قبل استيراد الـ core ──
try:
    if "KOBO_TOKEN" in st.secrets:
        os.environ["KOBO_TOKEN"] = str(st.secrets["KOBO_TOKEN"])
except Exception:
    pass  # مفيش secrets.toml محلياً، هنعتمد على environment variable

import farm_qa_core as qa  # noqa: E402

if not qa.KOBO_TOKEN:
    st.error("❌ KOBO_TOKEN غير موجود. ضيفه في App settings → Secrets أو كـ environment variable.")
    st.stop()

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MASTER_LOG_FILE = "master_form_reviews.csv"
FIELD_LOG_FILE = "field_level_reviews.csv"
DEBUG_DIR = "debug_imgs"
MASTER_COLS = [
    "farm_id", "submission_id", "Form_Status", "Rejected_Items_Count",
    "Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At",
]


# ═══════════════════════ 0. توافق بين نسخ Streamlit ═══════════════════════
def stretch(fn, *args, **kwargs):
    """width='stretch' في النسخ الجديدة، use_container_width في القديمة."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


# ═══════════════════════ 1. CACHING & KOBO ═══════════════════════
@st.cache_resource(show_spinner="Loading DeepForest AI Model into memory...")
def get_tree_counter():
    counter = qa.TreeCounter()
    counter._load()
    return counter


@st.cache_data(ttl=600, show_spinner="Fetching raw submissions & attachments from KoboToolbox...")
def get_kobo_raw_and_df():
    raw_rows = qa.fetch_kobo()
    flat_df = qa.build_dataframe(raw_rows)
    return raw_rows, flat_df


@st.cache_data(ttl=3600, show_spinner="Fetching Form Question Labels from Kobo Schema...")
def get_kobo_labels() -> dict:
    labels = {}
    try:
        headers = {"Authorization": f"Token {qa.KOBO_TOKEN}"}
        url = f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}.json"
        r = requests.get(url, headers=headers, timeout=30)
        if r.ok:
            survey = r.json().get("content", {}).get("survey", [])
            for item in survey:
                name = item.get("name") or item.get("$autoname")
                lbl = item.get("label")
                if name and lbl:
                    if isinstance(lbl, list):
                        lbl_str = " / ".join(str(x) for x in lbl if x)
                    else:
                        lbl_str = str(lbl)
                    labels[name.lower()] = lbl_str
    except Exception:
        pass
    return labels


@st.cache_data(ttl=3600, show_spinner=False, max_entries=300)
def fetch_kobo_image_bytes(url: str) -> bytes | None:
    try:
        headers = {"Authorization": f"Token {qa.KOBO_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=30)
        if r.ok:
            return r.content
    except Exception:
        pass
    return None


def bytes_to_data_uri(img_bytes: bytes) -> str:
    if not img_bytes:
        return ""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_attachment_lookup(raw_row: dict) -> dict:
    lookup = {}
    atts = raw_row.get("_attachments", [])
    if not isinstance(atts, list):
        return lookup
    for a in atts:
        if not isinstance(a, dict):
            continue
        fname_full = str(a.get("filename", ""))
        fname_short = fname_full.split("/")[-1].strip()
        dl_url = a.get("download_medium_url") or a.get("download_small_url") or a.get("download_url")
        if fname_short and dl_url:
            lookup[fname_short] = dl_url
            lookup[fname_full] = dl_url
    return lookup


def is_image_filename(val) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def update_kobo_validation_status(submission_id: str, status_code: str) -> bool:
    try:
        headers = {"Authorization": f"Token {qa.KOBO_TOKEN}", "Content-Type": "application/json"}
        url = f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}/data/validation_statuses/"
        payload = {
            "payload": {
                "submission_ids": [int(submission_id)],
                "validation_status.uid": status_code,
            }
        }
        r = requests.patch(url, headers=headers, json=payload, timeout=20)
        return r.ok
    except Exception:
        return False


# ═══════════════════════ 2. HELPERS ═══════════════════════
META_PREFIXES = ("_", "meta/", "formhub/", "__version__", "deviceid", "simserial", "phonenumber", "subscriberid")


def resolve_label(col_path: str, label_map: dict) -> str:
    clean_leaf = re.sub(r"\[\d+\]", "", str(col_path)).split("/")[-1].lower()
    repeat_idx = re.findall(r"\[\d+\]", str(col_path))
    prefix_tag = f"{repeat_idx[0]} " if repeat_idx else ""
    if clean_leaf in label_map:
        return f"{prefix_tag}{label_map[clean_leaf]} ({clean_leaf})"
    return str(col_path)


def load_master_log() -> pd.DataFrame:
    if os.path.exists(MASTER_LOG_FILE):
        try:
            return pd.read_csv(MASTER_LOG_FILE, dtype={"farm_id": str})
        except Exception:
            pass
    return pd.DataFrame(columns=MASTER_COLS)


def save_master_decision(record: dict):
    df = load_master_log()
    fid = str(record["farm_id"])
    if not df.empty:
        df = df[df["farm_id"].astype(str) != fid]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(MASTER_LOG_FILE, index=False, encoding="utf-8-sig")


def save_field_decisions(farm_id: str, field_df: pd.DataFrame):
    f_df = field_df.copy()
    f_df["farm_id"] = str(farm_id)
    if os.path.exists(FIELD_LOG_FILE):
        try:
            old = pd.read_csv(FIELD_LOG_FILE, dtype={"farm_id": str})
            old = old[old["farm_id"].astype(str) != str(farm_id)]
            f_df = pd.concat([old, f_df], ignore_index=True)
        except Exception:
            pass
    f_df.to_csv(FIELD_LOG_FILE, index=False, encoding="utf-8-sig")


def load_field_decisions(farm_id: str) -> pd.DataFrame:
    if os.path.exists(FIELD_LOG_FILE):
        try:
            df = pd.read_csv(FIELD_LOG_FILE, dtype={"farm_id": str})
            return df[df["farm_id"].astype(str) == str(farm_id)]
        except Exception:
            pass
    return pd.DataFrame()


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]


def blank_record(farm_id, err) -> dict:
    return {
        "farm_id": farm_id, "engineer": "—", "verdict": f"❌ Processing Error: {err}",
        "confidence": "🔴 Low", "flags": [], "_poly": None,
    }


# ═══════════════════════ 3. SIDEBAR & EXECUTION ═══════════════════════
st.title("🛰️ Enterprise Farm QA & Full-Form Inspection System")
st.caption("Full Kobo Question/Answer + Inline Question Photos • Item-by-Item Rejection • Comment Merger • GeoAI")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
if "flash_warn" in st.session_state:
    st.warning(st.session_state.pop("flash_warn"))

with st.sidebar:
    st.header("👤 Auditor Profile")
    auditor_name = st.text_input("Auditor Name:", value="Senior QA Auditor")

    st.divider()
    st.header("⚙️ Audit Scope")
    mode = st.selectbox("Select Scope:", ["Specific Farm (Farm ID)", "By Submission Date", "All Farms"])

    farm_id_input, date_str = None, None
    if mode == "Specific Farm (Farm ID)":
        farm_id_input = st.text_input("Enter Farm ID:")
    elif mode == "By Submission Date":
        date_str = str(st.date_input("Select Submission Date:"))

    use_ai = st.checkbox("Run DeepForest AI (satellite tree count)", value=True)
    use_ndvi = st.checkbox("Include Sentinel-2 NDVI (slower)", value=True, disabled=not use_ai)

    if st.button("🔄 Refresh Kobo Server Data"):
        st.cache_data.clear()
        st.rerun()

    run_btn = stretch(st.button, "🚀 Load & Audit Forms", type="primary")

    st.divider()
    st.header("💾 Backup (الملفات المحلية بتتمسح عند الـ reboot)")
    for label, path in (("Master decisions", MASTER_LOG_FILE), ("Field-level decisions", FIELD_LOG_FILE)):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                st.download_button(f"⬇️ {label}", fh.read(), file_name=path, mime="text/csv", key=f"dl_{path}")
        up = st.file_uploader(f"⬆️ Restore {label}", type="csv", key=f"up_{path}")
        if up is not None and st.button(f"Restore {label}", key=f"restore_{path}"):
            with open(path, "wb") as fh:
                fh.write(up.getvalue())
            st.success("Restored ✔ — اضغط Load & Audit Forms تاني")

if run_btn:
    mode_key = "farm" if farm_id_input else "date" if date_str and mode == "By Submission Date" else "all"
    try:
        raw_rows, raw_df = get_kobo_raw_and_df()
        filtered_df = qa.filter_df(raw_df, mode=mode_key, date=date_str, farm=farm_id_input)
        if filtered_df.empty:
            st.warning("⚠️ No forms matched your search criteria.")
        else:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            run_ai = use_ai
            counter = None
            if use_ai:
                try:
                    counter = get_tree_counter()
                except Exception as e:
                    run_ai = False
                    st.error(f"❌ تعذر تحميل موديل DeepForest — التدقيق هيكمل من غير AI: {e}")
                    st.exception(e)
            cols = qa.map_all(filtered_df)
            recs = []
            total = len(filtered_df)
            prog = st.progress(0, text="Running GeoAI & Form Audit...")
            for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
                try:
                    rec = qa.audit_row(row, cols, counter=counter, use_ai=run_ai,
                                       debug_dir=DEBUG_DIR, use_ndvi=(use_ndvi and run_ai))
                except Exception as e:
                    rec = blank_record(qa.first_value(row, cols["farm_id"]) or f"row{i}", e)
                rec["_df_index"] = idx
                rec["_submission_id"] = row.get("_id", "")
                recs.append(rec)
                prog.progress(i / total, text=f"Processed {i}/{total} forms...")
            prog.empty()

            qa.cross_checks(recs)
            for r in recs:
                r["flags"] = "; ".join(r.get("flags", []))

            res_df = pd.DataFrame(recs)
            res_df["farm_id"] = res_df["farm_id"].astype(str)

            master_log = load_master_log()
            keep = ["farm_id", "Form_Status", "Rejected_Items_Count", "Consolidated_Comments",
                    "General_Auditor_Note", "Auditor_Name", "Reviewed_At"]
            if not master_log.empty:
                ml = master_log[keep].drop_duplicates("farm_id", keep="last")
                res_df = res_df.merge(ml, on="farm_id", how="left")
            else:
                for c in keep[1:]:
                    res_df[c] = ""

            res_df["Form_Status"] = res_df["Form_Status"].fillna("⏳ Pending Review").replace("", "⏳ Pending Review")
            res_df["Rejected_Items_Count"] = pd.to_numeric(res_df["Rejected_Items_Count"], errors="coerce").fillna(0).astype(int)
            for c in ("Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At"):
                res_df[c] = res_df[c].fillna("").astype(str)

            st.session_state["results"] = res_df
            st.session_state["raw_rows"] = raw_rows
            st.session_state["flat_df"] = raw_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)

# ═══════════════════════ 4. MAIN WORKSPACE ═══════════════════════
if "results" in st.session_state:
    res = st.session_state["results"]
    raw_rows = st.session_state["raw_rows"]
    flat_df = st.session_state["flat_df"]
    label_map = get_kobo_labels()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Loaded Forms", len(res))
    k2.metric("AI Excellent Match ✅", int(res["verdict"].astype(str).str.contains("Excellent Match", na=False).sum()))
    k3.metric("Approved Forms 👍", int((res["Form_Status"] == "✅ Approved Entire Form").sum()))
    k4.metric("Conditional / Modify ⚠️", int((res["Form_Status"] == "⚠️ Conditional (Modifications Required)").sum()))
    k5.metric("Rejected Entirely 🚫", int((res["Form_Status"] == "🚫 REJECTED ENTIRE FORM").sum()))

    st.divider()

    col_sel, col_dl = st.columns([2, 1])
    with col_sel:
        farm_options = res["farm_id"].astype(str).tolist()

        def _fmt(x):
            r0 = res[res["farm_id"] == x].iloc[0]
            return f"Farm {x}  |  Engineer: {r0['engineer']}  |  Status: {r0['Form_Status']}"

        selected_farm = st.selectbox(
            "🎯 Select Farm ID to Open Full Form, Question Photos & Auditor:",
            farm_options, format_func=_fmt,
        )
    with col_dl:
        st.write("")
        excel_out = "enterprise_farm_qa_report.xlsx"
        try:
            qa.export_report(res, excel_out)
            with open(excel_out, "rb") as f:
                stretch(
                    st.download_button,
                    "📥 Export Master Audit Report (Excel)", f.read(),
                    file_name=excel_out,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.caption(f"Excel export unavailable: {e}")

    if selected_farm:
        audit_row_idx = res.index[res["farm_id"].astype(str) == selected_farm][0]
        audit_rec = res.loc[audit_row_idx]
        orig_df_idx = audit_rec["_df_index"]
        flat_submission = flat_df.loc[orig_df_idx]
        raw_submission = raw_rows[orig_df_idx]
        sub_id = str(audit_rec.get("_submission_id", ""))
        att_lookup = build_attachment_lookup(raw_submission)

        tab_form, tab_warehouse = st.tabs([
            "📋 1. Full Form Questions, Inline Photos & Item-by-Item Auditor",
            "📊 2. Master Data Warehouse (All Farms)",
        ])

        with tab_form:
            top_left, top_right = st.columns([1.5, 1])
            with top_left:
                st.info(
                    f"**🤖 Automated GeoAI Verdict ({audit_rec.get('confidence', '')}):** {audit_rec['verdict']}\n\n"
                    f"- **Entered Area:** `{audit_rec.get('area_entered', '—')}` Feddan | **GPS Area:** `{audit_rec.get('area_gps', '—')}` Feddan ({audit_rec.get('area_check', '')})\n"
                    f"- **Engineer Trees:** `{audit_rec.get('engineer_count', '—')}` ({audit_rec.get('tree_details', '—')}) | **Satellite AI:** `{audit_rec.get('ai_count', '—')}` | **Agronomic Forecast:** `{audit_rec.get('forecast', '—')}`\n"
                    f"- **Sentinel-2 NDVI:** `{audit_rec.get('ndvi_recent', '—')}` | **Vegetation Cover:** `{audit_rec.get('veg_pct', '—')}%`"
                )
                if audit_rec.get("gps_warnings"):
                    st.warning(f"**Geometry Quality:** {audit_rec['gps_warnings']}")
                if audit_rec.get("flags"):
                    st.error(f"**Cross-Check Flags:** {audit_rec['flags']}")
            with top_right:
                sat_img_path = os.path.join(DEBUG_DIR, f"{selected_farm.replace('/', '_')}.jpg")
                if os.path.exists(sat_img_path):
                    stretch(
                        st.image, Image.open(sat_img_path),
                        caption=f"🛰️ DeepForest Satellite Detection — Farm {selected_farm} (AI Count: {audit_rec.get('ai_count', '—')})",
                    )
                else:
                    st.caption("No satellite AI image available for this farm.")

            st.divider()
            st.markdown("### 📝 Complete Form Questions, Answers & Inline Question Photos")
            st.caption("Every question answered in the form appears below. Photos appear as thumbnails in the **📷 Question Photo** column, and full-size photos are in the inspector under the table.")

            qa_items, photo_questions = [], []
            area_bad = str(audit_rec.get("area_check", "")).startswith("❌")
            verdict_bad = "❌" in str(audit_rec.get("verdict", ""))

            qa_items.append({
                "Field_Key": "_AI_AREA_CHECK",
                "Question_Title": "🛰️ [GeoAI Check] Farm Area (Entered vs GPS)",
                "Submitted_Answer": f"Entered: {audit_rec.get('area_entered')} Fed | GPS: {audit_rec.get('area_gps')} Fed ({audit_rec.get('area_check')})",
                "Photo_Preview": "",
                "Item_Decision": "❌ Reject" if area_bad else "✅ OK",
                "Suggested_Correction_or_Comment": f"Area mismatch: GPS shows {audit_rec.get('area_gps')} Feddan" if area_bad else "",
                "Attach_to_Summary": bool(area_bad),
            })
            qa_items.append({
                "Field_Key": "_AI_TREE_CHECK",
                "Question_Title": "🛰️ [GeoAI Check] Tree Count (Engineer vs Satellite vs Forecast)",
                "Submitted_Answer": f"Eng: {audit_rec.get('engineer_count')} | AI: {audit_rec.get('ai_count')} | Forecast: {audit_rec.get('forecast')}",
                "Photo_Preview": "",
                "Item_Decision": "❌ Reject" if verdict_bad else "✅ OK",
                "Suggested_Correction_or_Comment": str(audit_rec.get("verdict", "")) if verdict_bad else "",
                "Attach_to_Summary": bool(verdict_bad),
            })

            f_col1, f_col2 = st.columns(2)
            show_empty = f_col1.checkbox("Show empty/unanswered form fields", value=False)
            only_photos = f_col2.checkbox("📸 Show ONLY questions with uploaded photos", value=False)

            with st.spinner("Matching question answers with uploaded Kobo photos..."):
                for col_name, val in flat_submission.items():
                    if str(col_name).startswith(META_PREFIXES):
                        continue
                    is_val_present = qa.present(val)
                    if not is_val_present and not show_empty:
                        continue

                    val_str = str(val).strip() if is_val_present else "— (Empty)"
                    q_label = resolve_label(col_name, label_map)

                    data_uri = ""
                    fname_clean = val_str.split("/")[-1].strip()
                    if is_image_filename(fname_clean) and fname_clean in att_lookup:
                        raw_img_bytes = fetch_kobo_image_bytes(att_lookup[fname_clean])
                        if raw_img_bytes:
                            data_uri = bytes_to_data_uri(raw_img_bytes)
                            photo_questions.append({
                                "Field_Key": str(col_name), "Question_Title": q_label,
                                "Filename": fname_clean, "Bytes": raw_img_bytes,
                            })

                    if only_photos and not data_uri:
                        continue

                    qa_items.append({
                        "Field_Key": str(col_name),
                        "Question_Title": ("📸 " + q_label) if data_uri else q_label,
                        "Submitted_Answer": val_str,
                        "Photo_Preview": data_uri,
                        "Item_Decision": "✅ OK",
                        "Suggested_Correction_or_Comment": "",
                        "Attach_to_Summary": False,
                    })

            form_qa_df = pd.DataFrame(qa_items)

            saved_fields = load_field_decisions(selected_farm)
            if not saved_fields.empty and not form_qa_df.empty:
                saved_map = saved_fields.drop_duplicates("Field_Key", keep="last").set_index("Field_Key").to_dict(orient="index")
                for i, r in form_qa_df.iterrows():
                    fk = r["Field_Key"]
                    if fk in saved_map:
                        s = saved_map[fk]
                        dec = s.get("Item_Decision", "✅ OK")
                        form_qa_df.at[i, "Item_Decision"] = dec if isinstance(dec, str) and dec else "✅ OK"
                        cm = s.get("Suggested_Correction_or_Comment", "")
                        form_qa_df.at[i, "Suggested_Correction_or_Comment"] = "" if pd.isna(cm) else str(cm)
                        att = s.get("Attach_to_Summary", False)
                        form_qa_df.at[i, "Attach_to_Summary"] = False if pd.isna(att) else bool(att)

            search_q = st.text_input("🔎 Search questions or answers (Arabic / English):", "")
            if search_q.strip() and not form_qa_df.empty:
                mask = (
                    form_qa_df["Question_Title"].str.contains(search_q, case=False, na=False, regex=False)
                    | form_qa_df["Submitted_Answer"].str.contains(search_q, case=False, na=False, regex=False)
                )
                display_qa_df = form_qa_df[mask].copy()
            else:
                display_qa_df = form_qa_df.copy()

            editor_key = f"form_editor_{selected_farm}_{short_hash(search_q)}_{int(show_empty)}{int(only_photos)}"
            edited_fields_df = stretch(
                st.data_editor,
                display_qa_df,
                hide_index=True,
                height=520,
                disabled=["Field_Key", "Question_Title", "Submitted_Answer", "Photo_Preview"],
                column_config={
                    "Field_Key": None,
                    "Question_Title": st.column_config.TextColumn("📌 Question Title (السؤال)", width="large"),
                    "Submitted_Answer": st.column_config.TextColumn("✍️ Answer (الإجابة)", width="medium"),
                    "Photo_Preview": st.column_config.ImageColumn("📷 Question Photo", help="Double-click any thumbnail to zoom in!", width="medium"),
                    "Item_Decision": st.column_config.SelectboxColumn(
                        "⚖️ Verdict", options=["✅ OK", "❌ Reject", "✏️ Modify", "❓ Clarify"], required=True, width="small"),
                    "Suggested_Correction_or_Comment": st.column_config.TextColumn("📝 Correction / Comment (التعديل أو التعليق)", width="large"),
                    "Attach_to_Summary": st.column_config.CheckboxColumn("📎 Attach?", help="Attach this comment to the final consolidated form report", default=False, width="small"),
                },
                key=editor_key,
            )

            if photo_questions:
                with st.expander(f"🖼️ Full-Size Visual Inspector for Question Photos ({len(photo_questions)} photos found in form)", expanded=False):
                    p_cols = st.columns(3)
                    for idx_p, pq in enumerate(photo_questions):
                        with p_cols[idx_p % 3]:
                            st.markdown(f"**📸 {pq['Question_Title']}**")
                            try:
                                stretch(st.image, Image.open(io.BytesIO(pq["Bytes"])), caption=f"File: {pq['Filename']}")
                            except Exception:
                                st.caption(f"Cannot display {pq['Filename']}")

            # ── تجميع التعليقات ──
            merged_lines = []
            rejected_count = modified_count = 0
            if not edited_fields_df.empty:
                ef = edited_fields_df.copy()
                ef["Suggested_Correction_or_Comment"] = ef["Suggested_Correction_or_Comment"].fillna("").astype(str)
                ef["Attach_to_Summary"] = ef["Attach_to_Summary"].fillna(False).astype(bool)
                rejected_count = int((ef["Item_Decision"] == "❌ Reject").sum())
                modified_count = int((ef["Item_Decision"] == "✏️ Modify").sum())
                flagged = ef["Item_Decision"].isin(["❌ Reject", "✏️ Modify", "❓ Clarify"])
                sel = ef[ef["Attach_to_Summary"] | (flagged & (ef["Suggested_Correction_or_Comment"].str.strip() != ""))]
                for _, r in sel.iterrows():
                    if not r["Attach_to_Summary"] and r["Item_Decision"] == "✅ OK":
                        continue
                    q_short = str(r["Question_Title"]).split("(")[0].strip()
                    ans_short = str(r["Submitted_Answer"])
                    if len(ans_short) > 60:
                        ans_short = ans_short[:60] + "..."
                    comm = r["Suggested_Correction_or_Comment"].strip() or "Flagged by auditor"
                    merged_lines.append(f"• [{r['Item_Decision']}] {q_short} (Ans: '{ans_short}') ➔ {comm}")

            st.divider()
            st.markdown("### 🗂️ Consolidated Audit Summary & Final Form Decision")
            col_summary, col_decision = st.columns([1.4, 1])

            with col_summary:
                st.markdown(f"**🔗 Auto-Merged Field Comments ({len(merged_lines)} attached items | {rejected_count} rejected | {modified_count} modified):**")
                default_merged = "\n".join(merged_lines) if merged_lines else str(audit_rec.get("Consolidated_Comments", ""))
                # الـ hash في الـ key عشان النص يتحدث لما قرارات البنود تتغير
                final_merged_text = st.text_area(
                    "Merged Item Comments (Editable before saving):",
                    value=default_merged, height=180,
                    key=f"merged_box_{selected_farm}_{short_hash(default_merged)}",
                )
                general_note = st.text_input(
                    "📌 General Form-Level Auditor Note / Master Rejection Reason:",
                    value=str(audit_rec.get("General_Auditor_Note", "")),
                    placeholder="e.g., Form rejected due to unclear animal photos and GPS polygon mismatch...",
                    key=f"gen_note_{selected_farm}",
                )

            with col_decision:
                st.markdown("**⚖️ Final Form Action (قرار الاستمارة الكلي):**")
                status_choices = [
                    "⏳ Pending Review",
                    "✅ Approved Entire Form",
                    "⚠️ Conditional (Modifications Required)",
                    "🚫 REJECTED ENTIRE FORM",
                ]
                cur_saved_status = str(audit_rec.get("Form_Status", "⏳ Pending Review"))
                if cur_saved_status not in status_choices:
                    cur_saved_status = "⏳ Pending Review"

                final_form_status = st.radio(
                    "Select Final Status for this Form:", status_choices,
                    index=status_choices.index(cur_saved_status),
                    key=f"final_radio_{selected_farm}",
                )
                sync_kobo = st.checkbox("🔗 Sync validation status directly to KoboToolbox server", value=False)

                c_btn1, c_btn2 = st.columns(2)
                save_all_btn = stretch(c_btn1.button, "💾 Save Decision", type="primary")
                reject_now_btn = stretch(c_btn2.button, "🚫 REJECT NOW")

                if save_all_btn or reject_now_btn:
                    chosen_status = "🚫 REJECTED ENTIRE FORM" if reject_now_btn else final_form_status
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

                    full_combined_report = final_merged_text
                    if general_note.strip():
                        full_combined_report = f"[General Note: {general_note.strip()}]\n" + full_combined_report

                    for col, v in (
                        ("Form_Status", chosen_status), ("Rejected_Items_Count", rejected_count),
                        ("Consolidated_Comments", full_combined_report), ("General_Auditor_Note", general_note),
                        ("Auditor_Name", auditor_name), ("Reviewed_At", now_ts),
                    ):
                        st.session_state["results"].loc[audit_row_idx, col] = v

                    if not edited_fields_df.empty:
                        save_field_decisions(
                            selected_farm,
                            edited_fields_df[["Field_Key", "Question_Title", "Submitted_Answer", "Item_Decision",
                                              "Suggested_Correction_or_Comment", "Attach_to_Summary"]],
                        )
                    save_master_decision({
                        "farm_id": selected_farm, "submission_id": sub_id,
                        "Form_Status": chosen_status, "Rejected_Items_Count": rejected_count,
                        "Consolidated_Comments": full_combined_report, "General_Auditor_Note": general_note,
                        "Auditor_Name": auditor_name, "Reviewed_At": now_ts,
                    })

                    msg = f"✅ Saved! Farm `{selected_farm}` marked as: **{chosen_status}**"
                    if sync_kobo and sub_id:
                        kobo_code = (
                            "validation_status_approved" if "Approved" in chosen_status
                            else "validation_status_not_approved" if "REJECTED" in chosen_status
                            else "validation_status_on_hold"
                        )
                        if update_kobo_validation_status(sub_id, kobo_code):
                            msg += " — synced to KoboToolbox ✔"
                        else:
                            st.session_state["flash_warn"] = "⚠️ Saved locally, but Kobo validation sync failed."
                    st.session_state["flash"] = msg
                    st.rerun()

        with tab_warehouse:
            st.subheader("📊 Master Audit Data Warehouse (All Loaded Farms)")
            priority_cols = [
                "farm_id", "Form_Status", "Rejected_Items_Count", "Consolidated_Comments",
                "General_Auditor_Note", "Auditor_Name", "Reviewed_At", "verdict", "confidence",
                "engineer", "area_entered", "area_gps", "area_check", "engineer_count", "ai_count", "forecast",
            ]
            rem_cols = [c for c in res.columns if not str(c).startswith("_") and c not in priority_cols]
            final_cols = [c for c in priority_cols if c in res.columns] + rem_cols
            stretch(st.dataframe, res[final_cols])
