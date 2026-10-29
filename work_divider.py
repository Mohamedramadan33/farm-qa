import streamlit as st
from work_divider import run_divide_work_pipeline, get_submissions
import team_store as ts

st.set_page_config(page_title="EBDA & Kobo", page_icon="🌱", layout="wide")

# التحقق من تسجيل الدخول
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

if not st.session_state.logged_in:
    st.title("🔐 تسجيل الدخول — نظام المتابعة")
    username = st.selectbox("اختر اسم المستخدم", list(ts.TEAM.keys()), format_func=lambda x: ts.display_name(x))
    password = st.text_input("كلمة المرور", type="password")
    
    if st.button("دخول"):
        ok, changed = ts.login(username, password)
        if ok:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.rerun()
        else:
            st.error("❌ كلمة المرور غير صحيحة")
    st.stop()

# الواجهة الرئيسية بعد تسجيل الدخول
user = st.session_state.user
st.sidebar.title(f"مرحباً، {ts.display_name(user)}")
if st.sidebar.button("تسجيل الخروج"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

st.title("🌱 EBDA & Kobo — لوحة متابعة وتوزيع الاستمارات")

# عرض لوحة التحكم المباشرة
st.info("📊 يتم جلب الاستمارات غير المراجعة مباشرة وتوزيعها من KoboToolbox.")

if st.button("🔄 تحديث البيانات وتشغيل التوزيع"):
    with st.spinner("جاري جلب البيانات من Kobo..."):
        try:
            run_divide_work_pipeline(st.write)
            st.success("✅ تم تحديث وتوزيع الاستمارات بنجاح!")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحديث: {e}")

# عرض الاستمارات الحالية من كوبو
st.markdown("### 📋 الاستمارات غير المراجعة في Kobo")
try:
    with st.spinner("جاري تحميل الاستمارات..."):
        subs = get_submissions(st.write)
        unvalidated = [s for s in subs if not (s.get("_validation_status") or {}).get("label")]
        st.metric("إجمالي الاستمارات المتبقية بدون Validation", len(unvalidated))
        
        if unvalidated:
            # عرض عينة أو تفاصيل بسيطة
            st.write(f"عدد الاستمارات المعلقة: {len(unvalidated)}")
        else:
            st.success("🎉 مفيش استمارات متبقية، كله تم مراجعته!")
except Exception as e:
    st.warning(f"تعذر جلب الاستمارات حالياً: {e}")
