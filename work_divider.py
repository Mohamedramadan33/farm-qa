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

def get_all_pages(url, logger, max_pages=None):
    results, page = [], 1
    headers = {"Authorization": f"Token {get_api_token()}"}
    while url:
        logger(f"[GET] Fetching page {page}...")
        r = requests.get(url, headers=headers, timeout=(10, 60))
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        page += 1
        if max_pages and page > max_pages: break
    return results

def find_asset_uid(logger):
    server = SERVER_URL.rstrip("/")
    url = f"{server}/api/v2/assets/?format=json&limit=100"
    assets = get_all_pages(url, logger)
    wanted = FORM_NAME.strip().lower()
    for a in assets:
        if (a.get("name") or "").strip().lower() == wanted:
            return a["uid"]
    raise RuntimeError(f'Form "{FORM_NAME}" not found.')

def get_submissions(logger):
    uid = find_asset_uid(logger)
    server = SERVER_URL.rstrip("/")
    from urllib.parse import quote
    fields = ["_id", "_validation_status", FARM_CODE_KEY, GOV_KEY]
    url = f"{server}/api/v2/assets/{uid}/data/?format=json&limit=3000&fields={quote(json.dumps(fields))}"
    return get_all_pages(url, logger)

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

def find_value(sub, hints):
    for key, value in sub.items():
        last = key.split("/")[-1].lower()
        if any(h.lower() in last for h in hints):
            if value not in (None, ""): return str(value)
    return ""

def run_divide_work_pipeline(logger):
    logger("🚀 [START] Divide Work Pipeline")
    logger("[STEP] Fetching submissions from KoboToolbox...")
    subs = get_submissions(logger)

    unvalidated = []
    for s in subs:
        if not (s.get("_validation_status") or {}).get("label"):
            code = s.get(FARM_CODE_KEY) or find_value(s, ["farm_id"]) or "?"
            gov = s.get(GOV_KEY) or find_value(s, GOVERNORATE_HINTS) or "?"
            unvalidated.append((code, gov))

    logger(f"[INFO] Found {len(unvalidated)} unvalidated farms.")
    worker_farms, worker_govs = divide_work_fairly(unvalidated, TEAM_MEMBERS)
    
    logger("\n[ASSIGNMENT BREAKDOWN]")
    for w in TEAM_MEMBERS:
        farms = worker_farms.get(w, [])
        w_gov_counts = Counter(gov for _, gov in farms)
        details = ", ".join([f"{gov}: {cnt}" for gov, cnt in w_gov_counts.items()])
        logger(f" ➜ {w.split('@')[0]} (Total: {len(farms)}) -> [{details or 'No farms'}]")

    logger("✅ [SUCCESS] Divide Work pipeline completed!")
