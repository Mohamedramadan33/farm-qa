# -*- coding: utf-8 -*-
"""
scripts/optimize_thresholds.py
-------------------------------
Sweeps confidence thresholds over validation predictions to select the PR-curve F1-optimal
confidence threshold and optionally update config/detection.config.json.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scripts.evaluate_detection import evaluate_predictions

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "detection.config.json"


def sweep_confidence_thresholds(
    gt_dir: Path,
    pred_dir: Path,
    img_sz: int = 512,
    iou_thresh: float = 0.45,
) -> Dict[str, object]:
    best_conf = 0.55
    best_f1 = -1.0
    curve: List[Dict[str, object]] = []

    for step in range(25, 86, 5):
        conf = round(step / 100.0, 2)
        res = evaluate_predictions(gt_dir, pred_dir, img_sz=img_sz, iou_thresh=iou_thresh, conf_thresh=conf)
        f1 = float(res["f1"])
        curve.append({"conf": conf, "precision": res["precision"], "recall": res["recall"], "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf

    return {"optimal_conf": best_conf, "best_f1": round(best_f1, 4), "curve": curve}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize confidence threshold on validation PR curve.")
    parser.add_argument("--gt-dir", default="dataset/labels/val")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    out = sweep_confidence_thresholds(Path(args.gt_dir), Path(args.pred_dir), img_sz=args.imgsz, iou_thresh=args.iou)
    print(json.dumps(out, indent=2))

    if args.write_config and CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg["aiModelConfThresh"] = out["optimal_conf"]
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"✅ Updated aiModelConfThresh={out['optimal_conf']} in {CONFIG_PATH}")
