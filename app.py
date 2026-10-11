import streamlit as st
import pandas as pd
import os
from PIL import Image
import farm_qa_core as qa

st.set_page_config(page_title="Farm QA Central System", layout="wide", page_icon="🛰️")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Farm QA Central System (BI & GeoAI Audit)")
st.caption("Automated Geospatial & AI Auditing | KoboToolbox -> Polygon QA -> DeepForest AI -> Sentinel-2 NDVI")

with st.sidebar:
    st.header("⚙️ Audit Parameters")
    mode = st.selectbox("Audit Scope:", ["Specific Farm (Farm ID)", "By Submission Date", "All Farms"])
    
    farm_id, date_str = None, None
    if mode == "Specific Farm (Farm ID)":
        farm_id = st.text_input("Enter Farm ID:")
    elif mode == "By Submission Date":
        date_str = str(st.date_input("Select Submission Date:"))

    use_ai = st.checkbox("Enable Satellite AI & Sentinel-2 NDVI", value=True)
    run_btn = st.button("🚀 Run QA Audit", use_container_width=True, type="primary")

if run_btn:
    mode_key = "farm" if farm_id else "date" if date_str else "all"
    with st.spinner("Fetching KoboToolbox data & running Geospatial/Satellite AI models..."):
        try:
            df = qa.load_data(mode=mode_key, date=date_str, farm=farm_id)
            if df.empty:
                st.warning("⚠️ No matching records found for the selected criteria.")
            else:
                os.makedirs("debug_imgs", exist_ok=True)
                res_df = qa.audit_df(df, use_ai=use_ai, debug_dir="debug_imgs", verbose=False)
                st.session_state["results"] = res_df
        except Exception as e:
            st.error(f"Error during connection or processing: {e}")

if "results" in st.session_state:
    res = st.session_state["results"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Audited Farms", len(res))
    c2.metric("Excellent Match ✅", int(res["verdict"].str.contains("Excellent Match", na=False).sum()))
    c3.metric("Needs Review ⚠️", int(res["verdict"].str.contains("Review|Variance|Theoretical|Unverified", na=False).sum()))
    c4.metric("Critical Discrepancy ❌", int(res["verdict"].str.contains("Discrepancy|Under-counting|Over Capacity", na=False).sum()))

    st.divider()
    st.subheader("📊 Audit Data Warehouse")
    cols_to_show = [c for c in res.columns if not str(c).startswith("_")]
    st.dataframe(res[cols_to_show], use_container_width=True)

    excel_file = "farm_qa_report.xlsx"
    qa.export_report(res, excel_file)
    with open(excel_file, "rb") as f:
        st.download_button(
            "📥 Download Full Audit Report (Excel)",
            f,
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.divider()
    st.subheader("🔍 DeepForest Visual Crown Detection & Farm Drill-Down")
    selected = st.selectbox("Select Farm ID to inspect:", res["farm_id"].astype(str).unique())
    if selected:
        row = res[res["farm_id"].astype(str) == selected].iloc[0]
        col_text, col_pic = st.columns([1.2, 1])
        with col_text:
            st.info(f"**System Verdict (Confidence: {row.get('confidence', '')}):**\n\n{row['verdict']}")
            st.write(f"- **Engineer:** {row['engineer']} | **Reported Count:** {row['engineer_count']} ({row.get('tree_details', '—')})")
            st.write(f"- **Satellite AI Count:** {row.get('ai_count', '—')} (Band: {row.get('ai_low', '—')} – {row.get('ai_high', '—')}) | **Agronomic Forecast:** {row.get('forecast', '—')}")
            st.write(f"- **Entered Area:** {row.get('area_entered', '—')} Feddan | **GPS Calculated Area:** {row.get('area_gps', '—')} Feddan ({row.get('area_check', '')})")
            st.write(f"- **Recent Sentinel-2 NDVI:** {row.get('ndvi_recent', '—')} | **Vegetation Cover (ExG):** {row.get('veg_pct', '—')}%")
            if row.get("gps_warnings"):
                st.warning(f"**Geometry Quality Warnings:** {row['gps_warnings']}")
            if row.get("flags"):
                st.error(f"**Cross-Check Flags:** {row['flags']}")
        with col_pic:
            img_path = os.path.join("debug_imgs", f"{selected.replace('/', '_')}.jpg")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"DeepForest Crown Detection - Farm {selected}", use_container_width=True)
            else:
                st.caption("No satellite detection image saved for this farm.")
