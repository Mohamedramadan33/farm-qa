# -*- coding: utf-8 -*-
"""
work_divider.py — EBDA Farm Review Fair Distribution Engine
Allocates farm audit submissions across the 4 audit team members (Mohamed, Mansy, Haidy, Islam)
while balancing governorate clusters and preventing regional silos.
"""

from typing import Dict, List, Any, Optional

TEAM = {
    "mohamed": {"key": "mohamed", "email": "mohamed@ebda.earth", "name": "Mohamed"},
    "mansy": {"key": "mansy", "email": "mansy.ayman@ebda.earth", "name": "Mansy"},
    "haidy": {"key": "haidy", "email": "haidy.abdelnasser@ebda.earth", "name": "Haidy"},
    "islam": {"key": "islam", "email": "eslam.ashraf@ebda.earth", "name": "Islam"},
}

TEAM_KEYS = ["mohamed", "mansy", "haidy", "islam"]

ALIASES = {
    "eslam": "islam",
    "islam.ashraf": "islam",
    "eslam.ashraf": "islam",
    "mansy.ayman": "mansy",
    "haidy.abdelnasser": "haidy",
}

DEFAULT_PASSWORD = "000000"


def display_name(key: Optional[str]) -> str:
    if not key:
        return "—"
    k = str(key).strip().lower()
    k = ALIASES.get(k, k)
    if k in TEAM:
        return TEAM[k]["name"]
    return key


def resolve_user(ident: Optional[str]) -> Optional[str]:
    if not ident:
        return None
    i = str(ident).strip().lower()
    if not i:
        return None
    if i in TEAM:
        return i
    if i in ALIASES:
        return ALIASES[i]
    for k, member in TEAM.items():
        email = member["email"].lower()
        if i == email or i == email.split("@")[0]:
            return k
    return None


def divide_work_fairly(
    unvalidated_farms: List[Dict[str, Any]],
    active_workers: Optional[List[str]] = None,
    max_gov_ratio: float = 1.4,
) -> Dict[str, Any]:
    """
    Fair governorate-aware assignment algorithm:
    1. Groups farms by governorate.
    2. Chunks governorates exceeding target_share * max_gov_ratio to maintain fairness.
    3. Sorts blocks by size (largest first) and greedily assigns to workers with lowest current workload.
    """
    if active_workers is None:
        active_workers = list(TEAM_KEYS)

    logs: List[str] = ["[FAST RUN] جاري توزيع المزارع على مراجعي الفريق..."]

    worker_farms: Dict[str, List[Dict[str, Any]]] = {w: [] for w in TEAM_KEYS}
    worker_govs: Dict[str, List[str]] = {w: [] for w in TEAM_KEYS}
    worker_counts: Dict[str, int] = {w: 0 for w in TEAM_KEYS}

    if not active_workers or not unvalidated_farms:
        return {
            "worker_farms": worker_farms,
            "worker_govs": worker_govs,
            "logs": logs,
        }

    # Group farms by governorate
    gov_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in unvalidated_farms:
        gov = item.get("governorate") or "غير محدد"
        if gov == "—":
            gov = "غير محدد"
        gov_groups.setdefault(gov, []).append(item)

    total_farms = len(unvalidated_farms)
    num_workers = len(active_workers)
    target_share = total_farms / num_workers if num_workers > 0 else 0

    class AssignmentBlock:
        def __init__(self, block_label: str, real_gov: str, items: List[Dict[str, Any]]):
            self.block_label = block_label
            self.real_gov = real_gov
            self.items = items

    assignment_blocks: List[AssignmentBlock] = []

    for gov, items in gov_groups.items():
        if len(items) > target_share * max_gov_ratio and num_workers > 1:
            chunk_size = max(1, int(target_share))
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                part_num = (i // chunk_size) + 1
                assignment_blocks.append(
                    AssignmentBlock(f"{gov} (P{part_num})", gov, chunk)
                )
        else:
            assignment_blocks.append(AssignmentBlock(gov, gov, items))

    # Sort blocks by farm count descending
    assignment_blocks.sort(key=lambda b: (-len(b.items), b.block_label))

    for block in assignment_blocks:
        sorted_workers = sorted(
            active_workers,
            key=lambda w: (worker_counts[w], TEAM[w]["email"]),
        )
        chosen_worker = sorted_workers[0]

        worker_farms[chosen_worker].extend(block.items)
        worker_counts[chosen_worker] += len(block.items)
        if block.real_gov not in worker_govs[chosen_worker]:
            worker_govs[chosen_worker].append(block.real_gov)

        logs.append(
            f"تم إسناد بلوك '{block.block_label}' ({len(block.items)} مزرعة) إلى {display_name(chosen_worker)}"
        )

    logs.append(
        f"اكتمل التوزيع: {', '.join(f'{display_name(w)}: {worker_counts[w]}' for w in active_workers)}"
    )

    return {
        "worker_farms": worker_farms,
        "worker_govs": worker_govs,
        "logs": logs,
    }
