import streamlit as st
import pandas as pd
import requests
import io
import os
import re
import gc
import base64
from datetime import datetime
from PIL import Image
import farm_qa_core as qa

st.set_page_config(page_title="Enterprise Farm QA & Form Auditor", layout="wide", page_icon="🛰️")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
""", unsafe_allow_html=True)

MASTER_LOG_FILE = "master_form_reviews.csv"
FIELD_LOG_FILE = "field_level_reviews.csv"

# ═══════════════════════ 1. CACHING & SAFE MODEL LOADING ═══════════════════════
@st.cache_resource(show_spinner="Loading DeepForest AI Model into memory...")
def get_tree_counter_safe():
    """Safely loads DeepForest without crashing the app if RAM or packages are constrained."""
    try:
        counter = qa.TreeCounter()
        counter._load()
        return counter, None
    except Exception as e:
        return None, str(e)

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
                        lbl_str = " / ".join([str(x) for x in lbl if x])
                    else:
                        lbl_str = str(lbl)
                    labels[name.lower()] = lbl_str
    except Exception:
        pass
    return labels

@st.cache_data(ttl=3600, max_entries=60, show_spinner=False)
def fetch_kobo_media_bytes(url: str, is_image: bool = True) -> bytes | None:
    """Downloads protected image or video bytes from Kobo using Token, compressing images to save RAM."""
    try:
        headers = {"Authorization": f"Token {qa.KOBO_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=60)
        if not r.ok:
            return None
        if is_image:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((900, 900))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        return r.content
    except Exception:
        return None

def bytes_to_data_uri(img_bytes: bytes) -> str:
    if not img_bytes:
        return ""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def is_image_filename(val) -> bool:
    if not isinstance(val, str):
        return False
    v = val.strip().lower()
    return v.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))

def is_video_filename(val) -> bool:
    if not isinstance(val, str):
        return False
    v = val.strip().lower()
    return v.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"))

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
        mime = str(a.get("mimetype", "")).lower()
        full_url = a.get("download_url")
        preview_url = a.get("download_medium_url") or a.get("download_small_url") or full_url
        if fname_short and full_url:
            info = {
                "preview_url": preview_url,
                "full_url": full_url,
                "mimetype": mime,
                "filename": fname_short
            }
            lookup[fname_short] = info
            lookup[fname_full] = info
    return lookup

def update_kobo_validation_status(submission_id: str, status_code: str) -> bool:
    try:
        headers = {"Authorization": f"Token {qa.KOBO_TOKEN}", "Content-Type": "application/json"}
        url = f"{qa.KOBO_SERVER}/api/v2/assets/{qa.ASSET_UID}/data/validation_statuses/"
        payload = {
            "payload": {
                "submission_ids": [int(submission_id)],
                "validation_status.uid": status_code
            }
        }
        r = requests.patch(url, headers=headers, json=payload, timeout=20)
        return r.ok
    except Exception:
        return False

# ═══════════════════════ 2. HELPER FUNCTIONS ═══════════════════════
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
    return pd.DataFrame(columns=[
        "farm_id", "submission_id", "Form_Status", "Rejected_Items_Count",
        "Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At"
    ])

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

# ═══════════════════════ 3. SIDEBAR & EXECUTION ═══════════════════════
st.title("🛰️ Enterprise Farm QA & Full-Form Media Auditor")
st.caption("Full Kobo Questions • Photo & Video Inspector with Direct Reject/Approve Controls • Auto-Comment Merger • GeoAI")

with st.sidebar:
    st.header("👤 Auditor Profile")
    auditor_name = st.text_input("Auditor Name:", value="Senior QA Auditor")
    
    st.divider()
    st.header("⚙️ Audit Scope")
    mode = st.selectbox("Select Scope:", ["Specific Farm (Farm ID)", "By Submission Date", "All Farms"])
    
    farm_id_input, date_str = None, None
    max_Batch = 10
    if mode == "Specific Farm (Farm ID)":
        farm_id_input = st.text_input("Enter Farm ID:")
    elif mode == "By Submission Date":
        date_str = str(st.date_input("Select Submission Date:"))
    else:
        max_Batch = st.slider("Max farms to process at once (RAM Protection):", 1, 50, 10)

    use_ai = st.checkbox("Run DeepForest Satellite Tree AI", value=True)
    use_ndvi = st.checkbox("Run Sentinel-2 NDVI Check (Slower)", value=False)
    
    if st.button("🔄 Refresh Kobo Server Data"):
        st.cache_data.clear()
        st.rerun()
        
    run_btn = st.button("🚀 Load & Audit Forms", use_container_width=True, type="primary")

if run_btn:
    mode_key = "farm" if farm_id_input else "date" if date_str else "all"
    with st.spinner("Fetching Kobo submissions, question labels, photos/videos, and running GeoAI..."):
        try:
            raw_rows, raw_df = get_kobo_raw_and_df()
            filtered_df = qa.filter_df(raw_df, mode=mode_key, date=date_str, farm=farm_id_input)
            if mode_key == "all" and len(filtered_df) > max_Batch:
                st.info(f"ℹ️ Showing latest {max_Batch} farms to protect server memory. Adjust slider in sidebar for more.")
                filtered_df = filtered_df.tail(max_Batch)

            if filtered_df.empty:
                st.warning("⚠️ No forms matched your search criteria.")
            else:
                os.makedirs("debug_imgs", exist_ok=True)
                counter, ai_load_err = (get_tree_counter_safe() if use_ai else (None, None))
                if use_ai and ai_load_err:
                    st.warning(f"⚠️ DeepForest AI skipped due to server constraint ({ai_load_err}). Continuing with GPS & Vegetation audit.")
                
                cols = qa.map_all(filtered_df)
                recs = []
                prog = st.progress(0, text="Running GeoAI & Form Audit...")
                for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
                    rec = qa.audit_row(row, cols, counter=counter, use_ai=use_ai, use_ndvi=use_ndvi, debug_dir="debug_imgs")
                    rec["_df_index"] = idx
                    rec["_submission_id"] = row.get("_id", "")
                    recs.append(rec)
                    prog.progress(i / len(filtered_df), text=f"Processed {i}/{len(filtered_df)} forms...")
                prog.empty()
                qa.cross_checks(recs)
                for r in recs:
                    r["flags"] = "; ".join(r.get("flags", []))
                
                res_df = pd.DataFrame(recs)
                res_df["farm_id"] = res_df["farm_id"].astype(str)
                
                master_log = load_master_log()
                if not master_log.empty:
                    res_df = res_df.merge(
                        master_log[["farm_id", "Form_Status", "Rejected_Items_Count", "Consolidated_Comments", "General_Auditor_Note", "Auditor_Name", "Reviewed_At"]],
                        on="farm_id", how="left"
                    )
                else:
                    res_df["Form_Status"] = "⏳ Pending Review"
                    res_df["Rejected_Items_Count"] = 0
                    res_df["Consolidated_Comments"] = ""
                    res_df["General_Auditor_Note"] = ""
                    res_df["Auditor_Name"] = ""
                    res_df["Reviewed_At"] = ""

                res_df["Form_Status"] = res_df["Form_Status"].fillna("⏳ Pending Review")
                res_df["Rejected_Items_Count"] = res_df["Rejected_Items_Count"].fillna(0)
                res_df["Consolidated_Comments"] = res_df["Consolidated_Comments"].fillna("")
                res_df["General_Auditor_Note"] = res_df["General_Auditor_Note"].fillna("")

                st.session_state["results"] = res_df
                st.session_state["raw_rows"] = raw_rows
                st.session_state["flat_df"] = raw_df
                gc.collect()
        except Exception as e:
            st.error(f"Error loading data: {e}")

# ═══════════════════════ 4. MAIN WORKSPACE ═══════════════════════
if "results" in st.session_state:
    res = st.session_state["results"]
    raw_rows = st.session_state["raw_rows"]
    flat_df = st.session_state["flat_df"]
    label_map = get_kobo_labels()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Loaded Forms", len(res))
    k2.metric("AI Excellent Match ✅", int(res["verdict"].str.contains("Excellent Match", na=False).sum()))
    k3.metric("Approved Forms 👍", int((res["Form_Status"] == "✅ Approved Entire Form").sum()))
    k4.metric("Conditional / Modify ⚠️", int((res["Form_Status"] == "⚠️ Conditional (Modifications Required)").sum()))
    k5.metric("Rejected Entirely 🚫", int((res["Form_Status"] == "🚫 REJECTED ENTIRE FORM").sum()))

    st.divider()

    col_sel, col_dl = st.columns([2, 1])
    with col_sel:
        farm_options = res["farm_id"].astype(str).tolist()
        selected_farm = st.selectbox(
            "🎯 Select Farm ID to Open Full Form, Photos, Videos & Auditor:",
            farm_options,
            format_func=lambda x: f"Farm {x}  |  Engineer: {res[res['farm_id']==x]['engineer'].iloc[0]}  |  Status: {res[res['farm_id']==x]['Form_Status'].iloc[0]}"
        )
    with col_dl:
        st.write("")
        excel_out = "enterprise_farm_qa_report.xlsx"
        qa.export_report(res, excel_out)
        with open(excel_out, "rb") as f:
            st.download_button(
                "📥 Export Master Audit Report (Excel)",
                f, file_name=excel_out,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    if selected_farm:
        audit_row_idx = res.index[res["farm_id"].astype(str) == selected_farm][0]
        audit_rec = res.loc[audit_row_idx]
        orig_df_idx = audit_rec["_df_index"]
        flat_submission = flat_df.loc[orig_df_idx]
        raw_submission = raw_rows[orig_df_idx]
        sub_id = str(audit_rec.get("_submission_id", ""))

        att_lookup = build_attachment_lookup(raw_submission)
        saved_fields_df = load_field_decisions(selected_farm)
        saved_map = saved_fields_df.set_index("Field_Key").to_dict(orient="index") if not saved_fields_df.empty else {}

        tab_form, tab_warehouse = st.tabs([
            "📋 1. Full Form Questions, Photos/Videos Audit Cards & Decision Merger",
            "📊 2. Master Data Warehouse (All Farms)"
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
                sat_img_path = os.path.join("debug_imgs", f"{selected_farm.replace('/', '_')}.jpg")
                if os.path.exists(sat_img_path):
                    st.image(
                        Image.open(sat_img_path),
                        caption=f"🛰️ DeepForest Satellite Detection — Farm {selected_farm} (AI Count: {audit_rec.get('ai_count', '—')})",
                        use_container_width=True
                    )
                else:
                    st.caption("No satellite AI image available for this farm.")

            # ── COLLECT QUESTIONS, PHOTOS & VIDEOS ──
            qa_items = []
            media_items = []
            matched_filenames = set()

            qa_items.append({
                "Field_Key": "_AI_AREA_CHECK",
                "Question_Title": "🛰️ [GeoAI Check] Farm Area (Entered vs GPS)",
                "Submitted_Answer": f"Entered: {audit_rec.get('area_entered')} Fed | GPS: {audit_rec.get('area_gps')} Fed ({audit_rec.get('area_check')})",
                "Photo_Preview": "",
                "Item_Decision": saved_map.get("_AI_AREA_CHECK", {}).get("Item_Decision", "❌ Reject" if str(audit_rec.get("area_check", "")).startswith("❌") else "✅ OK"),
                "Suggested_Correction_or_Comment": str(saved_map.get("_AI_AREA_CHECK", {}).get("Suggested_Correction_or_Comment", "" if not str(audit_rec.get("area_check", "")).startswith("❌") else f"Area mismatch: GPS shows {audit_rec.get('area_gps')} Feddan") or ""),
                "Attach_to_Summary": bool(saved_map.get("_AI_AREA_CHECK", {}).get("Attach_to_Summary", True if str(audit_rec.get("area_check", "")).startswith("❌") else False))
            })
            qa_items.append({
                "Field_Key": "_AI_TREE_CHECK",
                "Question_Title": "🛰️ [GeoAI Check] Tree Count (Engineer vs Satellite vs Forecast)",
                "Submitted_Answer": f"Eng: {audit_rec.get('engineer_count')} | AI: {audit_rec.get('ai_count')} | Forecast: {audit_rec.get('forecast')}",
                "Photo_Preview": "",
                "Item_Decision": saved_map.get("_AI_TREE_CHECK", {}).get("Item_Decision", "❌ Reject" if "❌" in str(audit_rec.get("verdict", "")) else "✅ OK"),
                "Suggested_Correction_or_Comment": str(saved_map.get("_AI_TREE_CHECK", {}).get("Suggested_Correction_or_Comment", "" if "❌" not in str(audit_rec.get("verdict", "")) else audit_rec.get("verdict", "")) or ""),
                "Attach_to_Summary": bool(saved_map.get("_AI_TREE_CHECK", {}).get("Attach_to_Summary", True if "❌" in str(audit_rec.get("verdict", "")) else False))
            })

            st.divider()
            f_col1, f_col2 = st.columns(2)
            show_empty = f_col1.checkbox("Show empty/unanswered form fields", value=False)
            only_media = f_col2.checkbox("📸🎥 Show ONLY questions with Photos or Videos in table", value=False)

            with st.spinner("Loading form questions, photos, and videos from Kobo..."):
                for col_name, val in flat_submission.items():
                    if str(col_name).startswith(META_PREFIXES):
                        continue
                    is_val_present = qa.present(val)
                    if not is_val_present and not show_empty:
                        continue

                    val_str = str(val).strip() if is_val_present else "— (Empty)"
                    q_label = resolve_label(col_name, label_map)
                    fname_clean = val_str.split("/")[-1].strip()

                    data_uri = ""
                    media_type = None
                    if fname_clean in att_lookup:
                        att_info = att_lookup[fname_clean]
                        mime = att_info["mimetype"]
                        if is_image_filename(fname_clean) or "image" in mime:
                            raw_bytes = fetch_kobo_media_bytes(att_info["preview_url"], is_image=True)
                            if raw_bytes:
                                data_uri = bytes_to_data_uri(raw_bytes)
                                media_type = "image"
                                matched_filenames.add(fname_clean)
                                media_items.append({
                                    "Field_Key": str(col_name),
                                    "Question_Title": q_label,
                                    "Filename": fname_clean,
                                    "Media_Type": "image",
                                    "Bytes": raw_bytes
                                })
                        elif is_video_filename(fname_clean) or "video" in mime:
                            raw_bytes = fetch_kobo_media_bytes(att_info["full_url"], is_image=False)
                            if raw_bytes:
                                media_type = "video"
                                matched_filenames.add(fname_clean)
                                media_items.append({
                                    "Field_Key": str(col_name),
                                    "Question_Title": q_label,
                                    "Filename": fname_clean,
                                    "Media_Type": "video",
                                    "Bytes": raw_bytes
                                })

                    if only_media and media_type is None:
                        continue

                    icon_prefix = "📸 " if media_type == "image" else "🎥 " if media_type == "video" else ""
                    saved_entry = saved_map.get(str(col_name), {})
                    qa_items.append({
                        "Field_Key": str(col_name),
                        "Question_Title": icon_prefix + q_label,
                        "Submitted_Answer": val_str,
                        "Photo_Preview": data_uri,
                        "Item_Decision": saved_entry.get("Item_Decision", "✅ OK"),
                        "Suggested_Correction_or_Comment": str(saved_entry.get("Suggested_Correction_or_Comment", "") or ""),
                        "Attach_to_Summary": bool(saved_entry.get("Attach_to_Summary", False))
                    })

                # Unlinked attachments fallback
                for fname_k, att_info in att_lookup.items():
                    short_n = att_info["filename"]
                    if short_n not in matched_filenames and fname_k == short_n:
                        mime = att_info["mimetype"]
                        fk_unlinked = f"_UNLINKED_{short_n}"
                        if is_video_filename(short_n) or "video" in mime:
                            raw_bytes = fetch_kobo_media_bytes(att_info["full_url"], is_image=False)
                            if raw_bytes:
                                matched_filenames.add(short_n)
                                media_items.append({
                                    "Field_Key": fk_unlinked,
                                    "Question_Title": f"Uploaded Video ({short_n})",
                                    "Filename": short_n,
                                    "Media_Type": "video",
                                    "Bytes": raw_bytes
                                })
                        elif is_image_filename(short_n) or "image" in mime:
                            raw_bytes = fetch_kobo_media_bytes(att_info["preview_url"], is_image=True)
                            if raw_bytes:
                                matched_filenames.add(short_n)
                                media_items.append({
                                    "Field_Key": fk_unlinked,
                                    "Question_Title": f"Uploaded Photo ({short_n})",
                                    "Filename": short_n,
                                    "Media_Type": "image",
                                    "Bytes": raw_bytes
                                })

            # ══════════════════════════════════════════════════════════════════════
            # SECTION A: INTERACTIVE PHOTO & VIDEO AUDIT CARDS
            # ══════════════════════════════════════════════════════════════════════
            media_decisions_override = {}
            if media_items:
                st.markdown(f"### 📸🎥 Visual Media Audit Cards ({len(media_items)} Photos & Videos in Form)")
                st.caption("Inspect every uploaded photo and video below. Under each item, select **OK / Reject / Modify**, write your **rejection reason or comment**, and check **Attach Comment** to include it in the final report.")
                
                m_cols = st.columns(3)
                for idx_m, m_item in enumerate(media_items):
                    fk = m_item["Field_Key"]
                    saved_m = saved_map.get(fk, {})
                    
                    with m_cols[idx_m % 3]:
                        with st.container(border=True):
                            badge = "🎥 VIDEO" if m_item["Media_Type"] == "video" else "📸 PHOTO"
                            st.markdown(f"**{badge}: {m_item['Question_Title']}**")
                            
                            if m_item["Media_Type"] == "image":
                                st.image(Image.open(io.BytesIO(m_item["Bytes"])), caption=f"File: {m_item['Filename']}", use_container_width=True)
                            else:
                                st.video(m_item["Bytes"])
                                st.caption(f"Video File: {m_item['Filename']}")

                            dec_options = ["✅ OK", "❌ Reject", "✏️ Modify", "❓ Clarify"]
                            default_dec = saved_m.get("Item_Decision", "✅ OK")
                            if default_dec not in dec_options:
                                default_dec = "✅ OK"

                            m_verdict = st.radio(
                                "Media Decision:",
                                dec_options,
                                index=dec_options.index(default_dec),
                                horizontal=True,
                                key=f"media_dec_{selected_farm}_{fk}"
                            )
                            m_comment = st.text_input(
                                "Why rejected / Comment (سبب الرفض أو التعليق):",
                                value=str(saved_m.get("Suggested_Correction_or_Comment", "") or ""),
                                placeholder="Write rejection reason or note...",
                                key=f"media_comm_{selected_farm}_{fk}"
                            )
                            default_att = bool(saved_m.get("Attach_to_Summary", False)) or (m_verdict == "❌ Reject")
                            m_attach = st.checkbox(
                                "📎 Attach comment to final report (إرفاق بالتقرير)",
                                value=default_att,
                                key=f"media_att_{selected_farm}_{fk}"
                            )

                            media_decisions_override[fk] = {
                                "Question_Title": m_item["Question_Title"],
                                "Submitted_Answer": m_item["Filename"],
                                "Item_Decision": m_verdict,
                                "Suggested_Correction_or_Comment": m_comment,
                                "Attach_to_Summary": m_attach
                            }

            form_qa_df = pd.DataFrame(qa_items)
            if not form_qa_df.empty and media_decisions_override:
                existing_keys = set(form_qa_df["Field_Key"])
                for i, r in form_qa_df.iterrows():
                    fk = r["Field_Key"]
                    if fk in media_decisions_override:
                        form_qa_df.at[i, "Item_Decision"] = media_decisions_override[fk]["Item_Decision"]
                        form_qa_df.at[i, "Suggested_Correction_or_Comment"] = media_decisions_override[fk]["Suggested_Correction_or_Comment"]
                        form_qa_df.at[i, "Attach_to_Summary"] = media_decisions_override[fk]["Attach_to_Summary"]
                
                for fk, m_val in media_decisions_override.items():
                    if fk not in existing_keys:
                        extra_row = pd.DataFrame([{
                            "Field_Key": fk,
                            "Question_Title": m_val["Question_Title"],
                            "Submitted_Answer": m_val["Submitted_Answer"],
                            "Photo_Preview": "",
                            "Item_Decision": m_val["Item_Decision"],
                            "Suggested_Correction_or_Comment": m_val["Suggested_Correction_or_Comment"],
                            "Attach_to_Summary": m_val["Attach_to_Summary"]
                        }])
                        form_qa_df = pd.concat([form_qa_df, extra_row], ignore_index=True)

            # ══════════════════════════════════════════════════════════════════════
            # SECTION B: COMPLETE FORM QUESTIONS & ANSWERS TABLE
            # ══════════════════════════════════════════════════════════════════════
            st.divider()
            st.markdown("### 📝 Complete Form Questions & Answers Table (All Fields)")
            search_q = st.text_input("🔎 Search questions or answers (Arabic / English):", "")
            if search_q.strip() and not form_qa_df.empty:
                mask = (
                    form_qa_df["Question_Title"].str.contains(search_q, case=False, na=False) |
                    form_qa_df["Submitted_Answer"].str.contains(search_q, case=False, na=False)
                )
                display_qa_df = form_qa_df[mask].copy()
            else:
                display_qa_df = form_qa_df.copy()

            edited_fields_df = st.data_editor(
                display_qa_df,
                use_container_width=True,
                hide_index=True,
                height=480,
                disabled=["Field_Key", "Question_Title", "Submitted_Answer", "Photo_Preview"],
                column_config={
                    "Field_Key": None,
                    "Question_Title": st.column_config.TextColumn("📌 Question Title (السؤال)", width="large"),
                    "Submitted_Answer": st.column_config.TextColumn("✍️ Answer (الإجابة)", width="medium"),
                    "Photo_Preview": st.column_config.ImageColumn(
                        "📷 Photo Preview",
                        help="Double-click thumbnail to zoom in",
                        width="medium"
                    ),
                    "Item_Decision": st.column_config.SelectboxColumn(
                        "⚖️ Verdict",
                        options=["✅ OK", "❌ Reject", "✏️ Modify", "❓ Clarify"],
                        required=True,
                        width="small"
                    ),
                    "Suggested_Correction_or_Comment": st.column_config.TextColumn(
                        "📝 Correction / Comment (التعديل أو التعليق)",
                        width="large"
                    ),
                    "Attach_to_Summary": st.column_config.CheckboxColumn(
                        "📎 Attach?",
                        help="Attach this
