# -*- coding: utf-8 -*-
"""
app.py — EBDA & Kobo Farm QA Audit Dashboard
Streamlit Cloud & Local Deployment Entry Point
Full Feature Set:
- Secure Login & Team Management (Mohamed, Mansy, Haidy, Islam)
- Live Kobo API & Airtable Integration with Secrets Auto-Detection
- Multi-Criteria Search & Filter (Farm ID, Name, Status, Validation, Reviewer, Gov)
- Automated Fair Work Distribution + Instant Farm Reassignment
- All-in-One Consolidated Farm Audit Dashboard
- Tree Counts (Engineer vs Perimeter Forecast vs AI Satellite Estimate)
- Satellite Boundary Image & Interactive Folium Map
- Full Kobo Photos Gallery with Individual Decisions (OK/Reject/Modify) and Comment Input
- PGS Rules Engine & Excel Export
"""

import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional

import openpyxl
import pandas as pd
import requests
from PIL import Image
import streamlit as st
from folium import Map, Polygon as FoliumPolygon, Marker, TileLayer, CircleMarker
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
    .login-box {
        max-width: 440px;
        margin: 60px auto;
        padding: 30px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .badge-approved { background-color: #10b981; color: white; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    .badge-conditional { background-color: #f59e0b; color: white; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    .badge-rejected { background-color: #ef4444; color: white; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    .badge-pending { background-color: #6b7280; color: white; padding: 3px 10px; border-radius: 6px; font-weight: bold; }
    div.block-container { padding-top: 1.5rem; }
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
        st.caption(f"الصورة: {e}")


def get_secret(key: str, default: str = "") -> str:
    """
    Safely extracts credentials from st.secrets across all possible nesting patterns
    without throwing KeyError or AttributeError.
    """
    try:
        if not hasattr(st, "secrets"):
            return default
        if key in st.secrets:
            return str(st.secrets[key]).strip()
        low_key = key.lower()
        if low_key in st.secrets:
            return str(st.secrets[low_key]).strip()

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
# Demo Fallback Seed Submissions (Used only if offline or no token)
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
        "photos": [
            {"filename": "boundary_photo.jpg", "url": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&auto=format&fit=crop", "title": "صورة أشجار الحدود (جازورين)"},
            {"filename": "soil_photo.jpg", "url": "https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=600&auto=format&fit=crop", "title": "صورة أخذ عينة التربة"},
            {"filename": "tenure_photo.jpg", "url": "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=600&auto=format&fit=crop", "title": "مستند الحيازة والملكية"},
        ],
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
        "photos": [
            {"filename": "olive_trees.jpg", "url": "https://images.unsplash.com/photo-1541888946425-d0fbb186f5f7?w=600&auto=format&fit=crop", "title": "صورة أشجار الزيتون على الحدود"},
        ],
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
        "photos": [
            {"filename": "palm_field.jpg", "url": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=600&auto=format&fit=crop", "title": "صورة النخيل ومصدات الرياح"},
        ],
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
        "photos": [],
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


@st.cache_data(ttl=3600)
def fetch_authenticated_kobo_image(url: str, token: str) -> Optional[bytes]:
    """
    Downloads private KoboToolbox images using the Authorization header.
    """
    if not url:
        return None
    headers = {}
    if token:
        headers["Authorization"] = f"Token {token}"
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.ok and len(resp.content) > 300:
            return resp.content
    except Exception:
        pass
    return None


# -------------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_user" not in st.session_state:
    st.session_state.current_user = "mohamed"

if "all_reviews" not in st.session_state:
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
# SCREEN 1: LOGIN PAGE (واجهة الدخول باسم وكلمة سر)
# -------------------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 1.8, 1])

    with col_l2:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #10b981; font-size: 2.2rem; font-weight: 800; margin-bottom: 4px;">🌾 نظام تدقيق مزارع EBDA</h1>
                <p style="color: #94a3b8; font-size: 1.05rem;">الجمعية المصرية للزراعة الحيوية — التحقق الهندسي وتدقيق PGS</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            st.subheader("🔐 تسجيل الدخول إلى النظام")
            user_choice = st.selectbox(
                "اسم المراجع (المستخدم):",
                options=wd.TEAM_KEYS,
                format_func=lambda k: f"{wd.TEAM[k]['name']} ({wd.TEAM[k]['email']})",
            )
            user_pass = st.text_input(
                "كلمة المرور:",
                type="password",
                placeholder="أدخل كلمة المرور (الافتراضية: 000000)",
            )
            submit_login = st.form_submit_button("تسجيل الدخول", type="primary", use_container_width=True)

            if submit_login:
                verified = store.verify_user(user_choice, user_pass)
                # Allow default 000000 if not set yet
                if verified or user_pass == "000000":
                    st.session_state.authenticated = True
                    st.session_state.current_user = user_choice
                    st.success(f"مرحباً بك يا {wd.display_name(user_choice)}! جاري الدخول...")
                    st.rerun()
                else:
                    st.error("كلمة المرور غير صحيحة. كلمة المرور الافتراضية هي: 000000")

    st.stop()


# -------------------------------------------------------------------
# SCREEN 2: MAIN DASHBOARD (لوحة التحكم الرئيسية بعد الدخول)
# -------------------------------------------------------------------

# Data Connection
token_to_use = SECRET_KOBO_TOKEN
server_to_use = SECRET_KOBO_SERVER
asset_to_use = SECRET_KOBO_ASSET

# Fetch live data
live_farms = []
if token_to_use:
    with st.spinner("جاري سحب بيانات الاستمارات الحية من KoboToolbox..."):
        live_farms = load_kobo_data(token_to_use, server_to_use, asset_to_use)

if live_farms:
    raw_farms = live_farms
    data_source_badge = f"🟢 Kobo API حي ({len(live_farms)} استمارة)"
else:
    raw_farms = DEMO_FARMS
    data_source_badge = f"⚪ وضع تجريبي ({len(DEMO_FARMS)} مزارع)"

# Ensure default assignments for unassigned farms
saved_assigns = st.session_state.assignments
needs_save = False
for idx, f in enumerate(raw_farms):
    sid = str(f.get("_id", idx + 1))
    if sid not in saved_assigns:
        saved_assigns[sid] = {
            "to": wd.TEAM_KEYS[idx % len(wd.TEAM_KEYS)],
            "by": "auto",
            "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "farm_id": f.get("farm_id", f"FARM-{sid}"),
        }
        needs_save = True
if needs_save:
    store.save_assignments(saved_assigns)
    st.session_state.assignments = saved_assigns


# -------------------------------------------------------------------
# Sidebar: Navigation & Controls
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 تدقيق المزارع PGS")
    st.write(f"المستخدم: **{wd.display_name(st.session_state.current_user)}**")
    st.caption(f"حالة المصدر: {data_source_badge}")

    if st.button("🚪 تسجيل الخروج", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ إجراءات سريعة")

    if st.button("🔄 تحديث من Kobo و Airtable", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Fair distribution button
    if st.button("⚡ تقسيم الشغل العادل على الفريق", type="primary", use_container_width=True):
        unval = [
            {"subId": str(f.get("_id")), "farmCode": f.get("farm_id"), "governorate": f.get("governorate")}
            for f in raw_farms
        ]
        div_res = wd.divide_work_fairly(unval, wd.TEAM_KEYS, max_gov_ratio=1.4)
        for worker, items in div_res["worker_farms"].items():
            for item in items:
                store.set_assignment(item["subId"], worker, by_user="auto_pipeline", farm_id=item["farmCode"])
        st.session_state.assignments = store.get_assignments()
        st.success("تم توزيع المزارع بالتساوي على الفريق ومراعاة المحافظات!")
        st.rerun()

    st.markdown("---")
    nav_choice = st.radio(
        "📌 التنقل بين الشاشات:",
        [
            "📋 قائمة وبحث وتوزيع المزارع",
            "🌾 فحص المزرعة الشامل والصور والأشجار",
            "📊 التقرير النهائي وتصدير Excel",
            "⚖️ موازنة محافظات الفريق",
        ],
    )


# -------------------------------------------------------------------
# VIEW 1: قائمة وبحث وتوزيع المزارع (Search, Filter & Allocation)
# -------------------------------------------------------------------
if nav_choice == "📋 قائمة وبحث وتوزيع المزارع":
    st.header("📋 قائمة وبحث المزارع وتوزيع المهام")
    st.caption("يمكنك البحث بالاسم أو كود المزرعة أو حالة الاعتماد وتعيين أي مزرعة لأي عضو في الفريق فوراً.")

    # Top Search Bar
    s_col1, s_col2 = st.columns([2.5, 1])
    with s_col1:
        search_query = st.text_input(
            "🔍 ابحث عن مزرعة (بالكود، الاسم، اسم المزارع، أو اسم المهندس):",
            placeholder="مثال: 001 أو الهدى أو رضوان أو أحمد...",
        ).strip().lower()

    # Filters row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        filter_status = st.selectbox(
            "حالة المراجعة:",
            options=["الكل", "Approved Entire Form", "Conditional (Modifications Required)", "REJECTED ENTIRE FORM", "Pending Review"],
            format_func=lambda s: {
                "الكل": "جميع الحالات",
                "Approved Entire Form": "معتمدة (Approved)",
                "Conditional (Modifications Required)": "مشروطة (Conditional)",
                "REJECTED ENTIRE FORM": "مرفوضة (Rejected)",
                "Pending Review": "قيد المراجعة (Pending)",
            }.get(s, s),
        )
    with f_col2:
        filter_reviewer = st.selectbox(
            "المراجع المسند إليه:",
            options=["الكل"] + wd.TEAM_KEYS,
            format_func=lambda k: "جميع المراجعين" if k == "الكل" else wd.display_name(k),
        )
    with f_col3:
        govs = sorted(list(set(f.get("governorate", "غير محدد") for f in raw_farms)))
        filter_gov = st.selectbox("المحافظة:", options=["الكل"] + govs)
    with f_col4:
        val_options = ["الكل", "Approved", "Not Approved", "On Hold", "لم تحدد"]
        filter_val = st.selectbox("حالة Kobo Validation:", options=val_options)

    # Filter Logic
    matched_farms = []
    reviews_master = st.session_state.all_reviews.get("master", {})

    for idx, f in enumerate(raw_farms):
        sid = str(f.get("_id", idx + 1))
        fid = str(f.get("farm_id", f"FARM-{sid}"))
        fname = str(f.get("farm_name", ""))
        farmer = str(f.get("rep_name", ""))
        eng = str(f.get("engineer_name", ""))
        gov = str(f.get("governorate", ""))
        assign_info = st.session_state.assignments.get(sid, {})
        assigned_to = assign_info.get("to") or wd.TEAM_KEYS[idx % len(wd.TEAM_KEYS)]

        rev = reviews_master.get(fid, {})
        rev_status = rev.get("Form_Status", "Pending Review")

        kobo_val_raw = f.get("_validation_status", {})
        kobo_val_label = kobo_val_raw.get("label", "لم تحدد") if isinstance(kobo_val_raw, dict) else "لم تحدد"

        # Apply search query
        if search_query:
            text_corpus = f"{fid} {fname} {farmer} {eng}".lower()
            if search_query not in text_corpus:
                continue

        # Apply filters
        if filter_status != "الكل" and rev_status != filter_status:
            continue
        if filter_reviewer != "الكل" and assigned_to != filter_reviewer:
            continue
        if filter_gov != "الكل" and gov != filter_gov:
            continue
        if filter_val != "الكل" and filter_val not in kobo_val_label:
            continue

        matched_farms.append({
            "idx": idx,
            "sub_id": sid,
            "farm_id": fid,
            "farm_name": fname,
            "farmer": farmer,
            "engineer": eng,
            "gov": gov,
            "area": f.get("farm_area", 0.0),
            "assigned_to": assigned_to,
            "status": rev_status,
            "val_status": kobo_val_label,
            "raw": f,
        })

    st.write(f"تم العثور على **{len(matched_farms)}** مزرعة مطابقة لشروط البحث.")

    # Work table with inline quick assignment selector
    if matched_farms:
        display_data = []
        for m in matched_farms:
            display_data.append({
                "كود المزرعة": m["farm_id"],
                "اسم المزرعة": m["farm_name"],
                "المزارع": m["farmer"],
                "المهندس": m["engineer"],
                "المحافظة": m["gov"],
                "المساحة (فدان)": m["area"],
                "المراجع المسند": wd.display_name(m["assigned_to"]),
                "قرار المراجعة": m["status"],
                "اعتماد Kobo": m["val_status"],
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("⚡ إسناد وتغيير مسؤول المزرعة يدوياً")
        assign_col1, assign_col2, assign_col3 = st.columns([2, 1.5, 1])

        with assign_col1:
            farm_to_reassign = st.selectbox(
                "اختر المزرعة لإعادة الإسناد:",
                options=[m["farm_id"] for m in matched_farms],
                format_func=lambda fid: f"{fid} — {next((m['farm_name'] for m in matched_farms if m['farm_id'] == fid), '')}",
            )
        with assign_col2:
            target_worker = st.selectbox(
                "تعيين المزرعة إلى:",
                options=wd.TEAM_KEYS,
                format_func=lambda k: wd.display_name(k),
            )
        with assign_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("تحديث الإسناد", use_container_width=True, type="primary"):
                target_m = next((m for m in matched_farms if m["farm_id"] == farm_to_reassign), None)
                if target_m:
                    store.set_assignment(
                        sub_id=target_m["sub_id"],
                        to_member=target_worker,
                        by_user=st.session_state.current_user,
                        farm_id=farm_to_reassign,
                    )
                    st.session_state.assignments = store.get_assignments()
                    st.success(f"تم إسناد مزرعة {farm_to_reassign} إلى {wd.display_name(target_worker)}!")
                    st.rerun()
    else:
        st.info("لا توجد مزارع تطابق كلمات البحث أو الفلاتر.")


# -------------------------------------------------------------------
# VIEW 2: فحص المزرعة الشامل والصور والأشجار (Consolidated Single View)
# -------------------------------------------------------------------
elif nav_choice == "🌾 فحص المزرعة الشامل والصور والأشجار":
    st.header("🌾 فحص المزرعة الشامل — القياسات، الأشجار، الصور وقواعد PGS")

    farm_keys = {f"{f.get('farm_id')} — {f.get('farm_name')} ({f.get('governorate')})": f for f in raw_farms}
    selected_farm_label = st.selectbox("اختر المزرعة للمعاينة والتدقيق:", options=list(farm_keys.keys()))
    farm = farm_keys[selected_farm_label]

    sub_id = str(farm.get("_id", ""))
    farm_id = farm.get("farm_id", "")
    reviews_master = st.session_state.all_reviews.get("master", {})
    existing_rev = reviews_master.get(farm_id, {})
    current_assign = st.session_state.assignments.get(sub_id, {}).get("to", "غير مسند")

    # Header Card
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
    with h_col1:
        st.metric("كود المزرعة", farm.get("farm_id", "—"))
    with h_col2:
        st.metric("المحافظة", farm.get("governorate", "—"))
    with h_col3:
        st.metric("المساحة المسجلة", f"{farm.get('farm_area', 0.0)} فدان")
    with h_col4:
        st.metric("مهندس المعاينة", farm.get("engineer_name", "—"))
    with h_col5:
        st.metric("المراجع المسند", wd.display_name(current_assign))

    st.markdown("---")

    # Quick inline reassignment option
    reassign_exp = st.expander("👤 تغيير المراجع المسند لهذه المزرعة", expanded=False)
    with reassign_exp:
        r_c1, r_c2 = st.columns([3, 1])
        with r_c1:
            new_assignee = st.selectbox(
                "إسناد المزرعة إلى:",
                options=wd.TEAM_KEYS,
                index=wd.TEAM_KEYS.index(current_assign) if current_assign in wd.TEAM_KEYS else 0,
                format_func=lambda k: wd.display_name(k),
                key=f"reassign_{farm_id}",
            )
        with r_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("حفظ التعيين", key=f"btn_reassign_{farm_id}", use_container_width=True):
                store.set_assignment(sub_id, new_assignee, by_user=st.session_state.current_user, farm_id=farm_id)
                st.session_state.assignments = store.get_assignments()
                st.success(f"تم إسناد المزرعة إلى {wd.display_name(new_assignee)}")
                st.rerun()

    # Geometry & Polygon
    poly_coords, is_repaired = core.parse_polygon_coords(farm.get("gps_polygon", ""))
    utm_metrics = core.calculate_utm_metrics(poly_coords) if poly_coords else {}
    area_gps_feddan = utm_metrics.get("area_feddan", 0.0)
    perimeter_m = utm_metrics.get("perimeter_m", 0.0)
    gps_quality = core.evaluate_gps_quality(poly_coords, utm_metrics.get("poly_utm"))

    # Tabs for consolidated view
    # Tabs for consolidated view
    tab_trees, tab_photos, tab_qa, tab_pgs, tab_decision = st.tabs([
        "🌳 رصد وتعداد الأشجار وصورة القمر الصناعي",
        "📸 الأسئلة المصورة من استمارة Kobo",
        "📝 جميع أسئلة وأجوبة الاستمارة",
        "📜 تدقيق قواعد PGS",
        "⚖️ قرار الاعتماد والمزامنة السحابية",
    ])

    # 1. Trees & Satellite Visual Detection
    with tab_trees:
        st.subheader("🌳 رصد وتعداد الأشجار على صورة القمر الصناعي وخريطة المضلع")
        st.caption("يتم رصد وترقيم كل شجرة على حدود المزرعة بنقاط خضراء واضحة، مع خوارزمية القمم الضوئية وتحديد دقيق للمسافات.")

        # Interactive Calibration Controls
        with st.expander("🎚️ إعدادات ومعايرة دقة الرصد وحساسية التيجان (Detection Settings)", expanded=False):
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                species_presets = {
                    "تلقائي من نوع شجر الاستمارة": None,
                    "جازورين (مصدات كثيفة - 2.0 متر)": 2.0,
                    "كافور / أوكالبتوس (3.0 متر)": 3.0,
                    "موالح / حمضيات (4.0 متر)": 4.0,
                    "زيتون (5.0 متر)": 5.0,
                    "مانجو / فاكهة كبيرة (6.0 متر)": 6.0,
                    "نخيل (8.0 متر)": 8.0,
                    "تخصيص يدوي للمسافة": "custom",
                }
                chosen_species = st.selectbox(
                    "نوع شجر الحدود والمسافة:",
                    options=list(species_presets.keys()),
                    key=f"tree_spec_sel_{farm_id}",
                )
            with t_col2:
                sel_sp = species_presets[chosen_species]
                if sel_sp == "custom":
                    custom_spacing = st.number_input(
                        "المسافة بين الأشجار (متر):",
                        min_value=1.0,
                        max_value=25.0,
                        value=3.0,
                        step=0.5,
                        key=f"custom_sp_inp_{farm_id}",
                    )
                elif sel_sp is not None:
                    custom_spacing = float(sel_sp)
                    st.text_input("المسافة المعتمدة:", value=f"{custom_spacing} متر", disabled=True, key=f"sp_disp_{farm_id}")
                else:
                    custom_spacing = None
                    st.text_input("المسافة المعتمدة:", value="تلقائي حسب الاستمارة", disabled=True, key=f"sp_disp_{farm_id}")
            with t_col3:
                sens_val = st.slider(
                    "حساسية قمم التيجان (Sensitivity):",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.50,
                    step=0.05,
                    key=f"tree_sens_val_{farm_id}",
                    help="زيادة القيمة ترصد الأشجار الصغيرة والمتباعدة، وتقليلها يركز على الأشجار الكبيرة فقط.",
                )

        # Compute satellite canopy & visual tree detections
        canopy_result = core.analyze_farm_satellite_canopy(
            poly_coords,
            perimeter_m,
            tree_type_str=farm.get("tree_type", ""),
            custom_spacing_m=custom_spacing,
            sensitivity=sens_val,
        )

        tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
        with tr_c1:
            st.metric(
                "الأشجار المرصودة بالقمر الصناعي (AI)",
                f"{canopy_result['ai_count']} شجرة",
                f"نطاق: {canopy_result['ai_low']} - {canopy_result['ai_high']}",
            )
        with tr_c2:
            st.metric("عدد المهندس بالاستمارة", f"{farm.get('tree_number', 0)} شجرة")
        with tr_c3:
            st.metric("توقع المسافات الهندسية", f"{canopy_result['forecast'] or 0} شجرة")
        with tr_c4:
            st.metric("نوع شجر الحدود", farm.get("tree_type", "غير محدد"))

        st.info(
            f"🌱 نسبة الغطاء النباتي على الحدود: **{canopy_result['veg_pct']}%** | 📏 مسافة الشجر المعتمدة: **{canopy_result['spacing_m']} متر** | 📅 تاريخ الالتقاط: **{canopy_result['imagery_date']}**"
        )

        # High-Resolution Satellite Image with Counted Trees Drawn on it
        if canopy_result.get("annotated_img") is not None:
            st.markdown("##### 🛰️ صورة القمر الصناعي محدد عليها الأشجار المرصودة بالأرقام:")
            safe_image(
                canopy_result["annotated_img"],
                caption=f"صورة القمر الصناعي — تم رصد وتحديد {canopy_result['ai_count']} شجرة على امتداد محيط المزرعة",
            )

        st.markdown("---")
        st.subheader("📐 مضلع GPS وخريطة المزرعة التفاعلية (مع نقاط الأشجار)")
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            st.metric("مساحة مضلع GPS", f"{area_gps_feddan:.2f} فدان")
            ent_area = float(farm.get("farm_area") or 0.0)
            diff_fed = abs(ent_area - area_gps_feddan)
            diff_pct = (diff_fed / max(ent_area, 0.01)) * 100
            if diff_pct <= 10:
                st.success(f"مطابقة ممتازة (الفارق: {diff_pct:.1f}%)")
            else:
                st.error(f"فارق ({diff_pct:.1f}%) بين المسجل ومضلع GPS!")
        with g_c2:
            st.metric("محيط المزرعة", f"{perimeter_m:.1f} متر")
        with g_c3:
            st.metric("جودة رفع GPS", f"{gps_quality['rating']} ({gps_quality['score']}/100)")

        if is_repaired:
            st.warning("⚠️ تم إصلاح وفك تقاطع ذاتي (Bowtie) في تسلسل نقاط GPS تلقائياً.")

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

            # Add each individual detected tree point as an interactive green CircleMarker
            for tp in canopy_result.get("tree_points", []):
                CircleMarker(
                    location=[tp["lat"], tp["lon"]],
                    radius=4,
                    color="#047857",
                    fill=True,
                    fill_color="#10b981",
                    fill_opacity=0.9,
                    tooltip=f"شجرة #{tp['id']}",
                ).add_to(m)

            Marker([mean_lat, mean_lon], tooltip=f"مركز {farm.get('farm_id')}").add_to(m)
            st_folium(m, height=420, use_container_width=True)

        # GPS Coordinates Table for Detected Trees
        tree_pts = canopy_result.get("tree_points", [])
        if tree_pts:
            with st.expander(f"📋 جدول إحداثيات كل شجرة مرصودة ({len(tree_pts)} شجرة)", expanded=False):
                pts_df = pd.DataFrame([
                    {
                        "رقم الشجرة": p["id"],
                        "خط العرض (Latitude)": f"{p['lat']:.6f}",
                        "خط الطول (Longitude)": f"{p['lon']:.6f}",
                    }
                    for p in tree_pts
                ])
                st.dataframe(pts_df, use_container_width=True, height=220)
                csv_bytes = pts_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 تحميل إحداثيات الأشجار بصيغة CSV",
                    data=csv_bytes,
                    file_name=f"trees_{farm_id}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{farm_id}",
                )

        # Custom AI Model Guide
        with st.expander("🧠 تعليم وتدريب نموذج ذكاء اصطناعي مخصص فائق الدقة (AI Custom Model Guide)", expanded=False):
            st.markdown(
                """
                **لرفع الدقة لأعلى مستوى ممكن بناءً على صور حقيقية من مزارعك:**
                1. **تجهيز الصور**: قم بتحميل لقطات عالية الدقة للتيجان ومحيط المزارع.
                2. **التعليم (Labeling)**: ارفع 20 - 30 صورة مجاناً على منصة [Roboflow](https://roboflow.com) وضع مربعات أو نقاط حول تيجان الأشجار تحت فئة `tree`.
                3. **التدريب (Training)**: اختر تدريب **YOLOv8 Nano** (سريع جداً وخفيف ولا يستهلك ذاكرة) في Google Colab في أقل من 10 دقائق.
                4. **الملف الجاهز**: عند وضع ملف الأوزان الناتج `best.pt` في مجلد التطبيق، سيتم تفعيله وقراءته تلقائياً!
                """
            )

    # 2. Questions with Photos (الأسئلة المصورة من Kobo)
    with tab_photos:
        st.subheader("📸 الأسئلة المصورة من استمارة Kobo مع فحص وقرار كل صورة")
        st.caption("عرض كافة الصور المرفوعة ميدانياً مع نص السؤال والإجابة، وإمكانية تحديد (مقبول / مرفوض) وكتابة سبب التعليق.")

        farm_photos = farm.get("photos", [])
        photo_decisions = store.get_photo_decisions(farm_id)

        if not farm_photos:
            st.info("لا توجد صور مرفقة في هذه الاستمارة حالياً.")
        else:
            for p_idx, p_item in enumerate(farm_photos):
                fn = p_item.get("filename", f"photo_{p_idx+1}")
                title = p_item.get("title", fn)
                url = p_item.get("url", "")
                existing_p_dec = photo_decisions.get(fn, {})

                with st.container():
                    st.markdown(f"#### 📷 {p_idx+1}. {title}")
                    col_img, col_act = st.columns([1.5, 2])

                    with col_img:
                        if "kobotoolbox" in url or "kf.kobo" in url:
                            img_bytes = fetch_authenticated_kobo_image(url, token_to_use)
                            if img_bytes:
                                safe_image(Image.open(BytesIO(img_bytes)), caption=fn)
                            else:
                                st.caption("تعذر تحميل الصورة مباشرة من Kobo. اضغط للفتح بالمتصفح:")
                                st.markdown(f"[🔗 فتح الصورة ({fn})]({url})")
                        else:
                            safe_image(url, caption=fn)

                    with col_act:
                        dec_options = ["مقبول", "مرفوض", "يحتاج تعديل", "استيضاح"]
                        curr_dec = existing_p_dec.get("decision", "مقبول")
                        dec_idx = dec_options.index(curr_dec) if curr_dec in dec_options else 0

                        p_choice = st.radio(
                            f"قرار السؤال / الصورة ({fn}):",
                            options=dec_options,
                            index=dec_idx,
                            horizontal=True,
                            key=f"p_dec_{farm_id}_{p_idx}",
                        )

                        p_comment = st.text_input(
                            f"سبب التعليق / الملاحظة على {title}:",
                            value=existing_p_dec.get("comment", ""),
                            placeholder="مثال: الصورة واضحة / غير واضحة / تحتاج إعادة التقاط...",
                            key=f"p_com_{farm_id}_{p_idx}",
                        )

                        if st.button(f"💾 حفظ قرار وتعليق الصورة {p_idx+1}", key=f"btn_save_p_{farm_id}_{p_idx}"):
                            store.save_photo_decision(farm_id, fn, p_choice, p_comment)
                            st.success(f"تم حفظ تعليق وقرار '{title}' بنجاح!")

                    st.markdown("<hr style='margin: 14px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # 3. All Form Questions & Answers (جميع أسئلة وأجوبة استمارة Kobo)
    with tab_qa:
        st.subheader("📝 جميع أسئلة وأجوبة استمارة Kobo بالتفصيل")
        st.caption("استعراض وتدقيق كافة الأسئلة والأجوبة الواردة في استمارة المعاينة الميدانية.")

        all_qa_items = core.get_form_qa_items(farm.get("flat") or farm, farm.get("photos", []))

        qa_search_col1, qa_search_col2 = st.columns([2, 1.5])
        with qa_search_col1:
            qa_search = st.text_input("🔍 بحث في الأسئلة أو الأجوبة:", placeholder="اكتب للبحث في نص السؤال أو الإجابة...").strip().lower()

        sections = sorted(list(set(item["section"] for item in all_qa_items)))
        with qa_search_col2:
            section_filter = st.selectbox("تصفية حسب القسم / المحور:", options=["جميع الأقسام"] + sections)

        filtered_qa = []
        for item in all_qa_items:
            if qa_search:
                combined_txt = f"{item['title']} {item['answer']} {item['field_key']}".lower()
                if qa_search not in combined_txt:
                    continue
            if section_filter != "جميع الأقسام" and item["section"] != section_filter:
                continue
            filtered_qa.append(item)

        st.write(f"عرض **{len(filtered_qa)}** سؤال وإجابة:")

        # Render Q&A Accordion / Cards
        for q_idx, q_item in enumerate(filtered_qa):
            with st.expander(f"🔹 **{q_item['title']}** — ({q_item['section']})", expanded=True if q_idx < 4 else False):
                qa_c1, qa_c2 = st.columns([2.5, 1.5])
                with qa_c1:
                    st.markdown(f"**السؤال:** {q_item['title']}")
                    st.markdown(f"**مسار الحقل (Field):** `{q_item['field_key']}`")
                    st.markdown(f"**الإجابة المسجلة بالاستمارة:**")
                    st.info(q_item['answer'])

                    if q_item.get("photo_url"):
                        st.markdown("**📸 صورة مرفقة بهذا السؤال:**")
                        safe_image(q_item["photo_url"], caption=q_item.get("photo_filename"))

                with qa_c2:
                    st.markdown("**تدقيق المراجع:**")
                    item_decision = st.selectbox(
                        "قرار الحقل:",
                        options=["OK (سليم)", "Reject (مرفوض)", "Modify (يحتاج تعديل)", "Clarify (استيضاح)"],
                        key=f"item_dec_{farm_id}_{q_idx}",
                    )
                    item_note = st.text_input(
                        "ملاحظة / تصحيح مقترح:",
                        placeholder="اكتب ملاحظتك على هذا السؤال...",
                        key=f"item_note_{farm_id}_{q_idx}",
                    )


    # 4. PGS Rules
    with tab_pgs:
        st.subheader("📜 تقرير محرك قواعد PGS الزراعية المصرية")
        flat_payload = farm.get("flat") or farm
        rule_findings = pgs.evaluate_pgs_rules(farm, flat_payload, poly_coords)

        for finding in rule_findings:
            if finding["passed"]:
                st.success(f"✅ **{finding['rule']}**: {finding['message']}")
            elif finding["severity"] == "error":
                st.error(f"❌ **{finding['rule']}**: {finding['message']}")
            else:
                st.warning(f"⚠️ **{finding['rule']}**: {finding['message']}")

    # 5. Master Decision & Cloud Sync
    with tab_decision:
        st.subheader("⚖️ القرار النهائي للمزرعة والمزامنة مع Airtable و Kobo")

        status_options = [
            "Approved Entire Form",
            "Conditional (Modifications Required)",
            "REJECTED ENTIRE FORM",
            "Pending Review",
        ]
        curr_status = existing_rev.get("Form_Status", "Pending Review")
        status_idx = status_options.index(curr_status) if curr_status in status_options else 3

        selected_status = st.selectbox(
            "القرار الشامل للاستمارة:",
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
            "الملاحظات العامة وتوصيات لجنة المراجعة:",
            value=existing_rev.get("General_Auditor_Note", ""),
            placeholder="اكتب التوصيات الفنية والملاحظات المطلوبة لتصحيح وضع المزرعة...",
            height=120,
        )

        btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])

        with btn_c1:
            if st.button("💾 حفظ القرار محلياً", type="primary", use_container_width=True):
                new_record = {
                    "Form_Status": selected_status,
                    "General_Auditor_Note": auditor_notes,
                    "Auditor_Name": wd.display_name(st.session_state.current_user),
                    "Rejected_Items_Count": 0,
                    "Consolidated_Comments": "; ".join(f["message"] for f in rule_findings if not f["passed"]),
                }
                store.save_master_review(farm_id, sub_id, new_record)
                st.session_state.all_reviews["master"][farm_id] = new_record
                st.success("تم حفظ قرار المراجعة محلياً بنجاح!")

        with btn_c2:
            if SECRET_AIRTABLE_KEY and st.button("☁️ إرسال ومزامنة مع Airtable", use_container_width=True):
                new_record = {
                    "farm_id": farm_id,
                    "submission_id": sub_id,
                    "Form_Status": selected_status,
                    "General_Auditor_Note": auditor_notes,
                    "Auditor_Name": wd.display_name(st.session_state.current_user),
                    "Rejected_Items_Count": 0,
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

        with btn_c3:
            if token_to_use and st.button("🌐 تحديث حالة استمارة Kobo", use_container_width=True):
                status_map = {
                    "Approved Entire Form": "validation_status_approved",
                    "Conditional (Modifications Required)": "validation_status_on_hold",
                    "REJECTED ENTIRE FORM": "validation_status_not_approved",
                    "Pending Review": "validation_status_on_hold",
                }
                k_code = status_map.get(selected_status, "validation_status_on_hold")
                k_res = store.sync_to_kobo_validation(server_to_use, asset_to_use, token_to_use, sub_id, k_code)
                if k_res.get("ok"):
                    st.success("تم تحديث حالة الاستمارة في KoboToolbox مباشرة!")
                else:
                    st.warning(f"تعذر تحديث Kobo: {k_res.get('error', 'Status code failure')}")

        st.caption(
            f"آخر تحديث: {existing_rev.get('Reviewed_At', 'لم تُراجع بعد')} بواسطة {existing_rev.get('Auditor_Name', '—')}"
        )


# -------------------------------------------------------------------
# VIEW 3: التقرير النهائي وتصدير Excel (Audit Log & Export)
# -------------------------------------------------------------------
elif nav_choice == "📊 التقرير النهائي وتصدير Excel":
    st.header("📊 التقرير النهائي لتدقيق المزارع وتصدير Excel")
    st.caption("ملف إكسيل (.xlsx) شامل يحتوي على كافة المزارع، قرارات المراجعين، نتائج الأقمار الصناعية، وتعليقات الصور.")

    reviews_master = st.session_state.all_reviews.get("master", {})
    all_photo_decs = st.session_state.all_reviews.get("photos", {})
    report_rows = []

    for f in raw_farms:
        fid = f.get("farm_id")
        rev = reviews_master.get(fid, {})
        assign = st.session_state.assignments.get(str(f.get("_id")), {})
        f_photos = all_photo_decs.get(fid, {})

        # Summarize photo notes
        photo_summary = []
        for p_fn, p_dat in f_photos.items():
            if p_dat.get("decision") != "مقبول" or p_dat.get("comment"):
                photo_summary.append(f"{p_fn} [{p_dat.get('decision')}]: {p_dat.get('comment')}")

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
                "تعليقات الصور المرفقة": " | ".join(photo_summary) if photo_summary else "كافة الصور مقبولة",
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
# VIEW 4: موازنة محافظات الفريق (Fair Workload Distribution)
# -------------------------------------------------------------------
elif nav_choice == "⚖️ موازنة محافظات الفريق":
    st.header("⚖️ موازنة محافظات الفريق والتوزيع الإقليمي")
    st.write(
        "تقسيم المزارع بالتساوي على أعضاء الفريق الأربعة (Mohamed, Mansy, Haidy, Islam) مع الحفاظ على حزم المحافظات لتفادي تشتت المراجع."
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

    if st.button("🔄 تطبيق التوزيع العادل الآن", type="primary"):
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
        st.rerun()

    with st.expander("📜 سجل خطوات التوزيع الحسابي (Logs):", expanded=False):
        for log_line in div_result["logs"]:
            st.text(log_line)
