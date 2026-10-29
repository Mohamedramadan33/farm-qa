import json
from collections import Counter
import requests
import streamlit as st

SERVER_URL = "https://kf.kobotoolbox.org"
FORM_NAME = "PGS - New"
FARM_CODE_KEY = "at_house/farm_id"
GOV_KEY = "engineer_data/__002"
GOVERNORATE_HINTS = ["governorate", "gov", "المحافظة", "المحافظه"]

TEAM_MEMBERS = [
    "mohamed@ebda.earth",
    "mansy.ayman@ebda.earth",
    "haidy.abdelnasser@ebda.earth",
    "eslam.ashraf@ebda.earth"
]

def get_api_token():
    return st.secrets["KOBO_TOKEN"]

# 🚀 تسريع التطبيق: حفظ البيانات لمدة 5 دقائق لتجنب الثقل
@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_data_fast():
    headers = {"Authorization": f"Token {get_api_token()}"}
    server = SERVER_URL.rstrip("/")
    
    # 1. جلب UID الخاص بالنموذج
    assets_url = f"{server}/api/v2/assets/?format=json&limit=50"
    r = requests.get(assets_url, headers=headers, timeout=10)
    r.raise_for_status()
    assets = r.json().get("results", [])
    
    uid = None
    for a in assets:
        if (a.get("name") or "").strip().lower() == FORM_NAME.strip().lower():
            uid = a["uid"]
            break
            
    if not uid:
        return []

    # 2. جلب الحقول الأساسية فقط لتقليل حجم البيانات
    fields = ["_id", "_validation_status", FARM_CODE_KEY, GOV_KEY]
    from urllib.parse import quote
    data_url = f"{server}/api/v2/assets/{uid}/data/?format=json&limit=2000&fields={quote(json.dumps(fields))}"
    
    r_data = requests.get(data_url, headers=headers, timeout=15)
    r_data.raise_for_status()
    return r_data.json().get("results", [])

def divide_work_fairly(unvalidated_farms, active_workers, max_gov_ratio=1.4):
    if not active_workers: return {}, {}
    gov_groups = {}
    for code, gov in unvalidated_farms:
        gov_groups.setdefault(gov, []).append(code)
    
    total_farms = len(unvalidated_farms)
    num_workers = len(active_workers)
    target_share = total_farms / num_workers if num_workers > 0 else 0
    
    worker_farms = {w: [] for w in active_workers}
    worker_govs = {w: [] for w in active_workers}
    worker_counts = {w: 0 for w in active_workers}
    
    assignment_blocks = []
    for gov, codes in gov_groups.items():
        if len(codes) > (target_share * max_gov_ratio) and num_workers > 1:
            chunk_size = max(1, int(target_share))
            for i in range(0, len(codes), chunk_size):
                chunk_codes = codes[i:i + chunk_size]
                assignment_blocks.append((f"{gov} (P{i//chunk_size + 1})", gov, chunk_codes))
        else:
            assignment_blocks.append((gov, gov, codes))
            
    assignment_blocks.sort(key=lambda x: (-len(x[2]), x[0]))
    
    for block_label, real_gov, codes in assignment_blocks:
        target_worker = min(active_workers, key=lambda w: (worker_counts[w], w))
        for code in codes:
            worker_farms[target_worker].append((code, real_gov))
        if real_gov not in worker_govs[target_worker]:
            worker_govs[target_worker].append(real_gov)
        worker_counts[target_worker] += len(codes)
        
    return worker_farms, worker_govs

def run_divide_work_pipeline(logger):
    logger("⚡ [FAST RUN] جاري جلب وتوزيع البيانات...")
    subs = fetch_kobo_data_fast()

    unvalidated = []
    for s in subs:
        if not (s.get("_validation_status") or {}).get("label"):
            code = s.get(FARM_CODE_KEY) or "N/A"
            gov = s.get(GOV_KEY) or "N/A"
            unvalidated.append((code, gov))

    logger(f"📊 عدد الاستمارات المتبقية: {len(unvalidated)}")
    worker_farms, worker_govs = divide_work_fairly(unvalidated, TEAM_MEMBERS)
    
    for w in TEAM_MEMBERS:
        farms = worker_farms.get(w, [])
        w_gov_counts = Counter(gov for _, gov in farms)
        details = ", ".join([f"{gov}: {cnt}" for gov, cnt in w_gov_counts.items()])
        logger(f" ➜ {w.split('@')[0]} ({len(farms)}) -> [{details or 'لا يوجد'}]")
