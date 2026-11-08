# -*- coding: utf-8 -*-
"""
app.py — EBDA & Kobo Farm QA Audit Dashboard
Streamlit Cloud & Local Deployment Entry Point
Fully wired to live KoboToolbox API and Airtable via st.secrets with safe fallbacks.
"""

import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

import openpyxl
import pandas as pd
import requests
import streamlit as st
from folium import Map, Polygon as FoliumPolygon, Marker, TileLayer
from streamlit_folium import st_folium

import farm_qa_core as core
import pgs_rules as pgs
import team_store as store
import work_divider as wd

st.set_page_config(
    page_title="EBDA & Kobo — Farm QA Audit",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom RTL & Modern Arabic Typography Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
    }
    div.block-container { padding-top: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_image(img_or_url: Any, caption: Optional[str] = None):
    """
    Safely renders image in Streamlit across all versions (supports both
    use_container_width and use_column_width without throwing TypeError).
    """
    if img_or_url is None:
        return
    try:
        st.image(img_or_url, caption=caption, use_column_width=True)
    except TypeError:
        try:
            st.image(img_or_url, caption=caption, use_container_width=True)
        except Exception:
            st.image(img_or_url, caption=caption)
    except Exception as e:
        st.caption(f"صورة مضلع القمر الصناعي: {e}")


def get_secret(key: str, default: str = "") -> str:
    """
    Safely extracts credentials from st.secrets across all possible nesting patterns
    without throwing KeyError or AttributeError.
    """
    try:
        if not hasattr(st, "secrets"):
            return default
        # 1. Direct top-level uppercase or lowercase
        if key in st.secrets:
            return str(st.secrets[key]).strip()
        low_key = key.lower()
        if low_key in st.secrets:
            return str(st.secrets[low_key]).strip()

        # 2. Nested sections e.g. [kobo] token = "..." or [airtable] api_key = "..."
        for section in st.secrets:
            sec_val = st.secrets[section]
            if isinstance(sec_val, dict):
                if key in sec_val:
                    return str(sec_val[key]).strip()
                if low_key in sec_val:
                    return str(sec_val[low_key]).strip()
                if "kobo" in section.lower() and "token" in sec_val:
                    if "TOKEN" in key:
                        return str(sec_val["token"]).strip()
                if "airtable" in section.lower():
                    if "KEY" in key and "api_key" in sec_val:
                        return str(sec_val["api_key"]).strip()
                    if "BASE" in key and "base_id" in sec_val:
                        return str(sec_val["base_id"]).strip()
                    if "TABLE" in key and "table_name" in sec_val:
                        return str(sec_val["table_name"]).strip()
    except Exception:
        pass
    return default


# Read Secrets
SECRET_KOBO_TOKEN = get_secret("KOBO_TOKEN")
SECRET_KOBO_SERVER = get_secret("KOBO_SERVER", "https://kf.kobotoolbox.org")
SECRET_KOBO_ASSET = get_secret("KOBO_ASSET_UID", "aX4NJWgge6tooXjfSYXhrq")
SECRET_AIRTABLE_KEY = get_secret("AIRTABLE_API_KEY") or get_secret("AIRTABLE_TOKEN")
SECRET_AIRTABLE_BASE = get_secret("AIRTABLE_BASE_ID")
SECRET_AIRTABLE_TABLE = get_secret("AIRTABLE_TABLE_NAME")


# -------------------------------------------------------------------
# Demo Fallback Seed Submissions (Used only if no token or offline)
# -------------------------------------------------------------------
DEMO_FARMS = [
    {
        "_id": 1001,
        "farm_id": "EG-GZ-2024-001",
        "farm_name": "مزرعة الهدى العضوية",
        "engineer_name": "م. أحمد عثمان",
        "rep_name": "الحاج رضوان الأباصيري",
        "rep_phone": "01012345678",
        "governorate": "الجيزة",
        "visit_date": "2024-05-12",
        "submission_time": "2024-05-14",
        "audit_type": "تجديد شهادة PGS السنوية",
        "season": "صيفي 2024",
        "farm_area": 12.5,
        "soil_type": "طميية رملية خفيفة",
        "irrig_type": "تنقيط حديث مع طلمبة طاقة شمسية",
        "tree_type": "جازورين",
        "tree_number": 420,
        "gps_polygon": "30.0125 31.1820 18 3; 30.0145 31.1855 18 3; 30.0110 31.1870 18 3; 30.0090 31.1830 18 3; 30.0125 31.1820 18 3",
        "soil_points": "30.0120 31.1835; 30.0135 31.1850; 30.0115 31.1860; 30.0100 31.1835",
        "attachments": [],
        "flat": {},
    },
    {
        "_id": 1002,
        "farm_id": "EG-BH-2024-004",
        "farm_name": "حوض الخير - النوبارية",
        "engineer_name": "م. مصطفى الشافعي",
        "rep_name": "عبد الله الشبراوي",
        "rep_phone": "01234567890",
        "governorate": "البحيرة",
        "visit_date": "2024-06-02",
        "submission_time": "2024-06-03",
        "audit_type": "معاينة أولية انضمام PGS",
        "season": "صيفي 2024",
        "farm_area": 8.0,
        "soil_type": "رملية مستصلحة",
        "irrig_type": "ري بالرش المحوري والتنقيط",
        "tree_type": "زيتون",
        "tree_number": 190,
        "gps_polygon": "30.6520 30.1210 22 2; 30.6550 30.1245 22 2; 30.6530 30.1280 22 2; 30.6495 30.1240 22 2; 30.6520 30.1210 22 2",
        "soil_points": "30.6525 30.1230; 30.6540 30.1250; 30.6520 30.1265; 30.6505 30.1235",
        "attachments": [],
        "flat": {},
    },
    {
        "_id": 1003,
        "farm_id": "EG-MN-2024-019",
        "farm_name": "مزرعة وادي النيل للتمور",
        "engineer_name": "م. هاني السعدني",
        "rep_name": "الحاج ممدوح سويلم",
        "rep_phone": "01188887777",
        "governorate": "المنيا",
        "visit_date": "2024-06-18",
        "submission_time": "2024-06-20",
        "audit_type": "تفتيش مفاجئ سنوي",
        "season": "صيفي 2024",
        "farm_area": 15.0,
        "soil_type": "طميية كلسية",
        "irrig_type": "غمر مقنن بالبوابات",
        "tree_type": "نخيل",
        "tree_number": 310,
        "gps_polygon": "28.1150 30.7420 40 4; 28.1180 30.7460 40 4; 28.1140 30.7490 40 4; 28.1110 30.7440 40 4; 28.1150 30.7420 40 4",
        "soil_points": "28.1155 30.7435; 28.1170 30.7455; 28.1135 30.7470",
        "attachments": [],
        "flat": {},
    },
    {
        "_id": 1004,
        "farm_id": "EG-FY-2024-008",
        "farm_name": "مزارع قارون الحيوية",
        "engineer_name": "م. كريم البنا",
        "rep_name": "سيد عبد التواب",
        "rep_phone": "01543219876",
        "governorate": "الفيوم",
        "visit_date": "2024-07-05",
        "submission_time": "2024-07-06",
        "audit_type": "مراجعة موسمية",
        "season": "صيفي 2024",
        "farm_area": 5.2,
        "soil_type": "طينية ثقيلة",
        "irrig_type": "تنقيط تحت سطحي",
        "tree_type": "كافور",
        "tree_number": 180,
        "gps_polygon": "29.3120 30.8250 12 3; 29.3150 30.8280 12 3; 29.3130 30.8310 12 3; 29.3100 30.8270 12 3; 29.3120 30.8250 12 3",
        "soil_points": "29.3125 30.8260; 29.3140 30.8285; 29.3128 30.8295; 29.3110 30.8265",
        "attachments": [],
        "flat": {},
    },
]


@st.cache_data(ttl=300)
def load_kobo_data(token: str, server_url: str, asset_uid: str) -> List[Dict[str, Any]]:
    if not token:
        return []
    raw_list = store.fetch_kobo_submissions(server_url, asset_uid, token)
    if not raw_list:
        return []
    return [core.extract_farm_record(r, idx) for idx, r in enumerate(raw_list)]


# -------------------------------------------------------------------
# Session State & Cache Initialization
# -------------------------------------------------------------------
if "current_user" not in st.session_state:
    st.session_state.current_user = "mohamed"

if "all_reviews" not in st.session_state:
    # Hydrate reviews from local store and Airtable
    local_reviews = store.get_all_reviews()
    if SECRET_AIRTABLE_KEY and SECRET_AIRTABLE_BASE and SECRET_AIRTABLE_TABLE:
        try:
            airtable_master = store.fetch_airtable_reviews(
                SECRET_AIRTABLE_KEY, SECRET_AIRTABLE_BASE, SECRET_AIRTABLE_TABLE
            )
            for fid, rec in airtable_master.items():
                local_reviews["master"][fid] = rec
        except Exception:
            pass
    st.session_state.all_reviews = local_reviews

if "assignments" not in st.session_state:
    st.session_state.assignments = store.get_assignments()

# -------------------------------------------------------------------
# Sidebar: User Authentication & Data Connection
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 نظام تدقيق المزارع")
    st.caption("الجمعية المصرية للزراعة الحيوية (EBDA) — نظام PGS")

    st.markdown("---")
    st.subheader("👤 حساب المراجع")
    current_key = st.selectbox(
        "اختر المراجع الحالي:",
        options=wd.TEAM_KEYS,
        index=wd.TEAM_KEYS.index(st.session_state.current_user)
        if st.session_state.current_user in wd.TEAM_KEYS
        else 0,
        format_func=lambda k: f"{wd.TEAM[k]['name']} ({wd.TEAM[k]['email']})",
    )
    st.session_state.current_user = current_key

    st.markdown("---")
    st.subheader("🔌 حالة الربط السحابي (API)")

    token_to_use = SECRET_KOBO_TOKEN
    server_to_use = SECRET_KOBO_SERVER
    asset_to_use = SECRET_KOBO_ASSET

    if not token_to_use:
        token_to_use = st.text_input(
            "أدخل Kobo Token:",
            type="password",
            help="إذا كان مسجلاً في st.secrets فلن تحتاج لإدخاله هنا.",
        )

    # Fetch live data
    live_farms = []
    if token_to_use:
        with st.spinner("جاري الاتصال بـ KoboToolbox..."):
            live_farms = load_kobo_data(token_to_use, server_to_use, asset_to_use)

    if live_farms:
        raw_farms = live_farms
        st.success(f"🟢 متصل بـ Kobo API: {len(live_farms)} استمارة")
    else:
        raw_farms = DEMO_FARMS
        st.info(f"⚪ الوضع الحالي: بيانات تجريبية ({len(DEMO_FARMS)} مزارع)")

    if SECRET_AIRTABLE_KEY:
        st.success("🟢 متصل بـ Airtable")
    else:
        st.caption("⚪ Airtable: محلي (أضف المفاتيح في secrets للمزامنة)")

    if st.button("🔄 تحديث البيانات الحية (Refresh)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    nav_choice = st.radio(
        "📌 الأقسام الرئيسية:",
        [
            "📋 قائمة المهام والمزارع",
            "🌾 فحص المزرعة وقواعد PGS",
            "📊 سجل المراجعات وتصدير Excel",
            "⚖️ التوزيع العادل للمهام",
        ],
    )

# -------------------------------------------------------------------
# View 1: قائمة المهام والمزارع (Audit Worklist)
# -------------------------------------------------------------------
if nav_choice == "📋 قائمة المهام والمزارع":
    st.header("📋 قائمة المزارع ومهام التدقيق")
    st.write(
        f"مرحباً **{wd.display_name(st.session_state.current_user)}**، يمكنك هنا الاطلاع على جميع المزارع المسندة إليك ومتابعة حالات التدقيق."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_reviewer = st.selectbox(
            "تصفية حسب المراجع المسند إليه:",
            options=["الكل"] + wd.TEAM_KEYS,
            format_func=lambda k: "جميع المراجعين" if k == "الكل" else wd.display_name(k),
        )
    with col2:
        govs = sorted(list(set(f.get("governorate", "غير محدد") for f in raw_farms)))
        filter_gov = st.selectbox("تصفية حسب المحافظة:", options=["الكل"] + govs)
    with col3:
        filter_status = st.selectbox(
            "تصفية حسب قرار الاعتماد:",
            options=["الكل", "Approved Entire Form", "Conditional (Modifications Required)", "REJECTED ENTIRE FORM", "Pending Review"],
            format_func=lambda s: {
                "الكل": "جميع الحالات",
                "Approved Entire Form": "معتمدة بالكامل (Approved)",
                "Conditional (Modifications Required)": "مشروطة بتعديلات (Conditional)",
                "REJECTED ENTIRE FORM": "مرفوضة بالكامل (Rejected)",
                "Pending Review": "قيد المراجعة (Pending)",
            }.get(s, s),
        )

    table_rows = []
    reviews_master = st.session_state.all_reviews.get("master", {})

    for idx, f in enumerate(raw_farms):
        sid = str(f.get("_id", idx + 1))
        fid = f.get("farm_id", f"FARM-{sid}")
        assign_info = st.session_state.assignments.get(sid, {})
        assigned_to = assign_info.get("to") or wd.TEAM_KEYS[idx % len(wd.TEAM_KEYS)]
        rev = reviews_master.get(fid, {})
        rev_status = rev.get("Form_Status", "Pending Review")

        if filter_reviewer != "الكل" and assigned_to != filter_reviewer:
            continue
        if filter_gov != "الكل" and f.get("governorate") != filter_gov:
            continue
        if filter_status != "الكل" and rev_status != filter_status:
            continue

        table_rows.append(
            {
                "كود المزرعة": fid,
                "اسم المزرعة": f.get("farm_name", "—"),
                "المحافظة": f.get("governorate", "—"),
                "اسم المزارع": f.get("rep_name", "—"),
                "المهندس المراجع": f.get("engineer_name", "—"),
                "المساحة (فدان)": f.get("farm_area", 0.0),
                "المراجع المسند": wd.display_name(assigned_to),
                "تاريخ الزيارة": f.get("visit_date", "—"),
                "حالة الاعتماد": rev_status,
                "sub_id": sid,
            }
        )

    df_display = pd.DataFrame(table_rows)
    if not df_display.empty:
        st.dataframe(
            df_display.drop(columns=["sub_id"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("لا توجد مزارع تطابق شروط التصفية المختارة.")

# -------------------------------------------------------------------
# View 2: فحص المزرعة وقواعد PGS (Farm Inspection & Rules)
# -------------------------------------------------------------------
elif nav_choice == "🌾 فحص المزرعة وقواعد PGS":
    st.header("🌾 فحص المزرعة والتحقق الهندسي وقواعد PGS")

    farm_choices = {f"{f.get('farm_id')} — {f.get('farm_name')} ({f.get('governorate')})": f for f in raw_farms}
    selected_label = st.selectbox("اختر المزرعة للمعاينة والتدقيق:", options=list(farm_choices.keys()))
    farm = farm_choices[selected_label]

    sub_id = str(farm.get("_id", ""))
    farm_id = farm.get("farm_id", "")
    reviews_master = st.session_state.all_reviews.get("master", {})
    existing_rev = reviews_master.get(farm_id, {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("كود المزرعة", farm.get("farm_id", "—"))
    with col2:
        st.metric("المحافظة", farm.get("governorate", "—"))
    with col3:
        st.metric("المساحة المسجلة", f"{farm.get('farm_area', 0.0)} فدان")
    with col4:
        st.metric("مهندس المعاينة", farm.get("engineer_name", "—"))

    st.markdown("---")

    poly_coords, is_repaired = core.parse_polygon_coords(farm.get("gps_polygon", ""))
    utm_metrics = core.calculate_utm_metrics(poly_coords) if poly_coords else {}
    area_gps_feddan = utm_metrics.get("area_feddan", 0.0)
    perimeter_m = utm_metrics.get("perimeter_m", 0.0)
    gps_quality = core.evaluate_gps_quality(poly_coords, utm_metrics.get("poly_utm"))

    t1, t2, t3 = st.tabs(["📐 القياسات الهندسية ومضلع GPS", "🛰️ تحليل الأقمار الصناعية والأشجار", "📜 قواعد PGS واعتماد الاستمارة"])

    with t1:
        st.subheader("1. تدقيق مضلع GPS وحسابات UTM")
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.metric("مساحة مضلع GPS", f"{area_gps_feddan:.2f} فدان")
            farm_ent_area = float(farm.get("farm_area") or 0.0)
            diff_feddan = abs(farm_ent_area - area_gps_feddan)
            diff_pct = (diff_feddan / max(farm_ent_area, 0.01)) * 100
            if diff_pct <= 10:
                st.success(f"مطابقة ممتازة (الفارق: {diff_pct:.1f}%)")
            else:
                st.error(f"فارق ({diff_pct:.1f}%) بين المسجل ومضلع GPS!")
        with g_col2:
            st.metric("محيط المزرعة", f"{perimeter_m:.1f} متر")
        with g_col3:
            st.metric("جودة رفع GPS", f"{gps_quality['rating']} ({gps_quality['score']}/100)")

        if is_repaired:
            st.warning("⚠️ تم رصد تقاطع ذاتي (Bowtie) في تسلسل النقاط، وتم فكه وإصلاحه هندسياً تلقائياً.")

        if poly_coords and len(poly_coords) >= 3:
            mean_lat = sum(p[1] for p in poly_coords) / len(poly_coords)
            mean_lon = sum(p[0] for p in poly_coords) / len(poly_coords)

            m = Map(location=[mean_lat, mean_lon], zoom_start=17, tiles=None)
            TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="ArcGIS Satellite",
                overlay=False,
                control=True,
            ).add_to(m)

            folium_coords = [[p[1], p[0]] for p in poly_coords]
            FoliumPolygon(
                locations=folium_coords,
                color="#10b981",
                weight=3,
                fill=True,
                fill_color="#10b981",
                fill_opacity=0.25,
                tooltip=f"مزرعة {farm.get('farm_name')}",
            ).add_to(m)

            Marker([mean_lat, mean_lon], tooltip=f"مركز {farm.get('farm_id')}").add_to(m)
            st_folium(m, height=420, use_container_width=True)
        else:
            st.warning("لا توجد إحداثيات GPS صالحة لرسم المضلع.")

    with t2:
        st.subheader("2. فحص الغطاء النباتي وتعداد الأشجار بالأقمار الصناعية")
        st.caption("تحليل بصري فائق السرعة عبر مؤشر الخضرة الزائد (ExG) على ممر الحدود (8 أمتار) دون استنزاف المعالج.")

        canopy_result = core.analyze_farm_satellite_canopy(
            poly_coords,
            perimeter_m,
            tree_type_str=farm.get("tree_type", ""),
        )

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            st.metric("نوع شجر الحدود", farm.get("tree_type", "غير محدد"))
        with c_col2:
            st.metric("عدد المهندس", farm.get("tree_number", 0))
        with c_col3:
            st.metric("توقع المسافات الهندسية", canopy_result["forecast"] or 0)
        with c_col4:
            st.metric(
                "تقدير القمر الصناعي (AI)",
                f"{canopy_result['ai_count']} شجرة",
                f"نطاق: {canopy_result['ai_low']} - {canopy_result['ai_high']}",
            )

        st.info(
            f"🌱 نسبة الغطاء النباتي على الحدود: **{canopy_result['veg_pct']}%** | 📅 تاريخ التقاط القمر الصناعي: **{canopy_result['imagery_date']}**"
        )

        # SAFE IMAGE RENDERING (No TypeError across any Streamlit version)
        if canopy_result.get("annotated_img") is not None:
            safe_image(
                canopy_result["annotated_img"],
                caption="صورة القمر الصناعي عالية الدقة مع تتبع حدود المزرعة",
            )

    with t3:
        st.subheader("3. تقرير محرك قواعد PGS والاعتماد النهائي")

        flat_payload = farm.get("flat") or farm
        rule_findings = pgs.evaluate_pgs_rules(farm, flat_payload, poly_coords)

        for finding in rule_findings:
            if finding["passed"]:
                st.success(f"✅ **{finding['rule']}**: {finding['message']}")
            elif finding["severity"] == "error":
                st.error(f"❌ **{finding['rule']}**: {finding['message']}")
            else:
                st.warning(f"⚠️ **{finding['rule']}**: {finding['message']}")

        st.markdown("---")
        st.subheader("⚖️ قرار المراجع والاعتماد النهائي")

        status_options = [
            "Approved Entire Form",
            "Conditional (Modifications Required)",
            "REJECTED ENTIRE FORM",
            "Pending Review",
        ]
        curr_status = existing_rev.get("Form_Status", "Pending Review")
        status_idx = status_options.index(curr_status) if curr_status in status_options else 3

        selected_status = st.selectbox(
            "القرار الإجمالي للمزرعة:",
            options=status_options,
            index=status_idx,
            format_func=lambda s: {
                "Approved Entire Form": "✅ اعتماد كامل للاستمارة (Approved Entire Form)",
                "Conditional (Modifications Required)": "⚠️ اعتماد مشروط باستيفاء ملاحظات (Conditional)",
                "REJECTED ENTIRE FORM": "❌ رفض الاستمارة بالكامل (REJECTED ENTIRE FORM)",
                "Pending Review": "⏳ قيد المراجعة والتدقيق (Pending Review)",
            }.get(s, s),
        )

        auditor_notes = st.text_area(
            "ملاحظات المراجع العامة وتوصيات اللجنة:",
            value=existing_rev.get("General_Auditor_Note", ""),
            placeholder="اكتب أي ملاحظات فنية أو تصحيحية مطلوبة من المهندس...",
            height=100,
        )

        save_col1, save_col2, save_col3 = st.columns([1, 1, 2])
        with save_col1:
            if st.button("💾 حفظ القرار محلياً", type="primary", use_container_width=True):
                new_record = {
                    "Form_Status": selected_status,
                    "General_Auditor_Note": auditor_notes,
                    "Auditor_Name": wd.display_name(st.session_state.current_user),
                    "Rejected_Items_Count": sum(1 for f in rule_findings if not f["passed"]),
                    "Consolidated_Comments": "; ".join(f["message"] for f in rule_findings if not f["passed"]),
                }
                store.save_master_review(farm_id, sub_id, new_record)
                st.session_state.all_reviews["master"][farm_id] = new_record
                st.success("تم حفظ القرار محلياً!")

        with save_col2:
            if SECRET_AIRTABLE_KEY and st.button("☁️ مزامنة مع Airtable", use_container_width=True):
                new_record = {
                    "farm_id": farm_id,
                    "submission_id": sub_id,
                    "Form_Status": selected_status,
                    "General_Auditor_Note": auditor_notes,
                    "Auditor_Name": wd.display_name(st.session_state.current_user),
                    "Rejected_Items_Count": sum(1 for f in rule_findings if not f["passed"]),
                    "Consolidated_Comments": "; ".join(f["message"] for f in rule_findings if not f["passed"]),
                    "Reviewed_At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                air_res = store.sync_to_airtable(
                    SECRET_AIRTABLE_KEY,
                    SECRET_AIRTABLE_BASE,
                    SECRET_AIRTABLE_TABLE,
                    new_record,
                )
                if air_res.get("ok"):
                    st.success(air_res.get("message"))
                else:
                    st.error(air_res.get("message"))

        with save_col3:
            st.caption(
                f"آخر تحديث: {existing_rev.get('Reviewed_At', 'لم تُراجع بعد')} بواسطة {existing_rev.get('Auditor_Name', '—')}"
            )

# -------------------------------------------------------------------
# View 3: سجل المراجعات وتصدير Excel (Audit Log & Export)
# -------------------------------------------------------------------
elif nav_choice == "📊 سجل المراجعات وتصدير Excel":
    st.header("📊 سجل المراجعات الشامل وتصدير تقرير Excel")
    st.write("يمكنك تنزيل تقرير كامل بجميع المزارع وقرارات التدقيق والنتائج الهندسية بصيغة `.xlsx`.")

    reviews_master = st.session_state.all_reviews.get("master", {})
    report_rows = []

    for f in raw_farms:
        fid = f.get("farm_id")
        rev = reviews_master.get(fid, {})
        assign = st.session_state.assignments.get(str(f.get("_id")), {})

        report_rows.append(
            {
                "كود المزرعة": fid,
                "اسم المزرعة": f.get("farm_name", "—"),
                "المحافظة": f.get("governorate", "—"),
                "المساحة المسجلة": f.get("farm_area", 0.0),
                "المهندس المراجع": f.get("engineer_name", "—"),
                "المراجع المسند": wd.display_name(assign.get("to")),
                "قرار الاستمارة": rev.get("Form_Status", "Pending Review"),
                "اسم المراجع المعتمد": rev.get("Auditor_Name", "—"),
                "تاريخ المراجعة": rev.get("Reviewed_At", "—"),
                "ملاحظات المراجع": rev.get("General_Auditor_Note", "—"),
                "المخالفات المرصودة": rev.get("Consolidated_Comments", "لا توجد"),
            }
        )

    df_report = pd.DataFrame(report_rows)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_report.to_excel(writer, index=False, sheet_name="EBDA_Farm_Audit_Report")
    excel_data = output.getvalue()

    st.download_button(
        label="📥 تنزيل تقرير التدقيق بصيغة Excel (.xlsx)",
        data=excel_data,
        file_name=f"EBDA_Farm_QA_Report_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

# -------------------------------------------------------------------
# View 4: التوزيع العادل للمهام (Fair Work Distribution)
# -------------------------------------------------------------------
elif nav_choice == "⚖️ التوزيع العادل للمهام":
    st.header("⚖️ خط توزيع العمل المتوازن على مراجعي الفريق")
    st.write(
        "تقسيم المزارع تلقائياً على أعضاء الفريق الأربعة (Mohamed, Mansy, Haidy, Islam) بالتساوي مع مراعاة حزم المحافظات لمنع تشتت المراجع بين محافظات متعددة."
    )

    unval_items = [
        {
            "subId": str(f.get("_id")),
            "farmCode": f.get("farm_id"),
            "governorate": f.get("governorate"),
        }
        for f in raw_farms
    ]

    div_result = wd.divide_work_fairly(unval_items, wd.TEAM_KEYS, max_gov_ratio=1.4)

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    cols = [col_w1, col_w2, col_w3, col_w4]

    for idx, worker in enumerate(wd.TEAM_KEYS):
        farms_for_w = div_result["worker_farms"][worker]
        govs_for_w = div_result["worker_govs"][worker]
        with cols[idx]:
            st.metric(
                label=wd.display_name(worker),
                value=f"{len(farms_for_w)} مزارع",
                delta=f"المحافظات: {', '.join(govs_for_w) if govs_for_w else 'لا يوجد'}",
            )

    if st.button("🔄 تطبيق التوزيع العادل وحفظه في سجل المهام", type="primary"):
        for worker, farms_list in div_result["worker_farms"].items():
            for farm_item in farms_list:
                store.set_assignment(
                    sub_id=farm_item["subId"],
                    to_member=worker,
                    by_user="pipeline",
                    farm_id=farm_item["farmCode"],
                )
        st.session_state.assignments = store.get_assignments()
        st.success("تم تحديث وحفظ توزيع المهام بنجاح!")

    with st.expander("📜 سجل خطوات التوزيع الحسابي (Logs):", expanded=False):
        for log_line in div_result["logs"]:
            st.text(log_line)
