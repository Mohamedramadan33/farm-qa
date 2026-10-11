import streamlit as st
import pandas as pd
import os
from PIL import Image
import farm_qa_core as qa

st.set_page_config(page_title="Farm QA Central System", layout="wide", page_icon="🛰️")

st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ نظام تدقيق المزارع المركزي (Farm QA Central System)")
st.caption("التدقيق الفضائي الآلي | Kobo -> Polygon -> DeepForest -> Sentinel-2 NDVI")

with st.sidebar:
    st.header("⚙️ معايير الفحص")
    mode = st.selectbox("نطاق الفحص:", ["مزرعة محددة (Farm ID)", "حسب تاريخ الإرسال", "جميع المزارع"])
    
    farm_id, date_str = None, None
    if mode == "مزرعة محددة (Farm ID)":
        farm_id = st.text_input("أدخل كود المزرعة:")
    elif mode == "حسب تاريخ الإرسال":
        date_str = str(st.date_input("اختر تاريخ الإرسال:"))

    use_ai = st.checkbox("تشغيل التحليل الفضائي (DeepForest + NDVI)", value=True)
    run_btn = st.button("🚀 بدء الفحص والتدقيق", use_container_width=True, type="primary")

if run_btn:
    mode_key = "farm" if farm_id else "date" if date_str else "all"
    with st.spinner("جاري سحب البيانات من Kobo وتحليل الأقمار الصناعية..."):
        try:
            df = qa.load_data(mode=mode_key, date=date_str, farm=farm_id)
            if df.empty:
                st.warning("⚠️ لم يتم العثور على استمارات مطابقة للبحث.")
            else:
                os.makedirs("debug_imgs", exist_ok=True)
                res_df = qa.audit_df(df, use_ai=use_ai, debug_dir="debug_imgs", verbose=False)
                st.session_state["results"] = res_df
        except Exception as e:
            st.error(f"حدث خطأ أثناء الاتصال أو المعالجة: {e}")

if "results" in st.session_state:
    res = st.session_state["results"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي المزارع", len(res))
    c2.metric("تطابق ممتاز ✅", int(res["verdict"].str.contains("تطابق ممتاز", na=False).sum()))
    c3.metric("مراجعة ⚠️", int(res["verdict"].str.contains("مراجعة|تباين|نظري", na=False).sum()))
    c4.metric("مخالفة / خطأ جسيم ❌", int(res["verdict"].str.contains("تلاعب|إهمال|تجاوز", na=False).sum()))

    st.divider()
    st.subheader("📊 مستودع بيانات التدقيق (Data Warehouse)")
    cols_to_show = [c for c in res.columns if not str(c).startswith("_")]
    st.dataframe(res[cols_to_show], use_container_width=True)

    excel_file = "farm_qa_report.xlsx"
    qa.export_report(res, excel_file)
    with open(excel_file, "rb") as f:
        st.download_button("📥 تحميل التقرير الشامل بصيغة Excel", f, file_name=excel_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.subheader("🔍 تفاصيل المزرعة والرصد البصري")
    selected = st.selectbox("اختر المزرعة:", res["farm_id"].astype(str).unique())
    if selected:
        row = res[res["farm_id"].astype(str) == selected].iloc[0]
        col_text, col_pic = st.columns([1.2, 1])
        with col_text:
            st.info(f"**قرار التدقيق ({row.get('confidence', '')}):**\n\n{row['verdict']}")
            st.write(f"- **المهندس:** {row['engineer']} | **العدد المحصور:** {row['engineer_count']} ({row.get('tree_details', '—')})")
            st.write(f"- **رصد القمر الصناعي:** {row.get('ai_count', '—')} | **التوقع الزراعي:** {row.get('forecast', '—')}")
            st.write(f"- **المساحة المدخلة:** {row.get('area_entered', '—')} فدان | **مساحة GPS:** {row.get('area_gps', '—')} فدان")
            st.write(f"- **مؤشر NDVI الحديث:** {row.get('ndvi_recent', '—')} | **الغطاء النباتي:** {row.get('veg_pct', '—')}%")
            if row.get("flags"):
                st.warning(f"ملاحظات التدقيق المتقاطع: {row['flags']}")
        with col_pic:
            img_path = os.path.join("debug_imgs", f"{selected.replace('/', '_')}.jpg")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"رصد التيجان للمزرعة {selected}", use_container_width=True)
            else:
                st.caption("لا توجد صورة فضائية ملتقطة لهذه المزرعة.")