# -*- coding: utf-8 -*-
"""
evaluate_against_kobo.py
-------------------------
Computes MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error)
comparing predicted tree counts against ground-truth KoboToolbox farm records,
overall per farm and broken down by class (orchard_tree, palm_tree, windbreak_tree).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

TREE_CLASSES = ("orchard_tree", "palm_tree", "windbreak_tree")


def compute_mae_mape(pairs: Iterable[Dict[str, float]]) -> Dict[str, Any]:
    abs_errors: List[float] = []
    pct_errors: List[float] = []
    n = 0

    for row in pairs:
        actual = float(row["actual"])
        pred = float(row["predicted"])
        err = abs(pred - actual)
        abs_errors.append(err)
        if actual > 0:
            pct_errors.append((err / actual) * 100.0)
        n += 1

    mae = sum(abs_errors) / n if n > 0 else 0.0
    mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0
    return {
        "count": n,
        "mae": round(mae, 3),
        "mape_pct": round(mape, 2),
    }


def evaluate_kobo_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    farm_rows: List[Dict[str, Any]] = []
    total_pairs: List[Dict[str, float]] = []
    class_pairs: Dict[str, List[Dict[str, float]]] = {cls: [] for cls in TREE_CLASSES}

    for rec in records:
        farm_id = str(rec.get("farm_id") or rec.get("farmId") or "unknown")
        actual_total = float(rec.get("actual_total", rec.get("kobo_count", 0)) or 0)
        pred_total = float(rec.get("pred_total", rec.get("ai_count", 0)) or 0)
        abs_err = abs(pred_total - actual_total)
        pct_err = round((abs_err / actual_total) * 100.0, 2) if actual_total > 0 else None

        total_pairs.append({"actual": actual_total, "predicted": pred_total})
        farm_rows.append(
            {
                "farm_id": farm_id,
                "actual_total": actual_total,
                "pred_total": pred_total,
                "abs_error": round(abs_err, 2),
                "ape_pct": pct_err,
            }
        )

        actual_cls = rec.get("actual_by_class") or {}
        pred_cls = rec.get("pred_by_class") or {}
        for cls in TREE_CLASSES:
            if cls in actual_cls or cls in pred_cls:
                class_pairs[cls].append(
                    {
                        "actual": float(actual_cls.get(cls, 0) or 0),
                        "predicted": float(pred_cls.get(cls, 0) or 0),
                    }
                )

    per_class = {cls: compute_mae_mape(items) for cls, items in class_pairs.items() if items}
    return {
        "overall": compute_mae_mape(total_pairs),
        "per_class": per_class,
        "per_farm": farm_rows,
    }


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("records", [])

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rec: Dict[str, Any] = {
                "farm_id": r.get("farm_id") or r.get("farmId"),
                "actual_total": float(r.get("actual_total") or r.get("kobo_count") or 0),
                "pred_total": float(r.get("pred_total") or r.get("ai_count") or 0),
                "actual_by_class": {},
                "pred_by_class": {},
            }
            for cls in TREE_CLASSES:
                ak = f"actual_{cls}"
                pk = f"pred_{cls}"
                if ak in r and r[ak] != "":
                    rec["actual_by_class"][cls] = float(r[ak])
                if pk in r and r[pk] != "":
                    rec["pred_by_class"][cls] = float(r[pk])
            rows.append(rec)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark predicted tree counts against Kobo ground truth (MAE & MAPE).")
    parser.add_argument("--input", required=True, help="JSON or CSV benchmark file")
    args = parser.parse_args()

    records = _load_records(Path(args.input))
    report = evaluate_kobo_records(records)
    print(json.dumps(report, indent=2))
