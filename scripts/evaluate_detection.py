# -*- coding: utf-8 -*-
"""
scripts/evaluate_detection.py
------------------------------
Evaluates YOLO tree detections against validation labels, reporting:
  - Precision, Recall, F1 per class & overall
  - Small-crown Recall (8-25px boxes at 512px scale / ~0.26m/px)
  - Hard-negative false positive rate on empty-label crop tiles
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "detection.config.json"


def _load_cfg() -> Dict[str, object]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"smallCrownPxRange": [8, 25], "nmsIouThresh": 0.40, "classes": ["orchard_tree", "palm_tree", "windbreak_tree"]}


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_sz: int) -> Tuple[float, float, float, float]:
    half_w = (w * img_sz) / 2.0
    half_h = (h * img_sz) / 2.0
    x = cx * img_sz
    y = cy * img_sz
    return (x - half_w, y - half_h, x + half_w, y + half_h)


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, aa + bb - inter)


def parse_yolo_file(path: Path, img_sz: int, is_pred: bool = False) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    out: List[Dict[str, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        conf = float(parts[5]) if is_pred and len(parts) >= 6 else 1.0
        xyxy = _yolo_to_xyxy(cx, cy, w, h, img_sz)
        size_px = max(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
        out.append({"cls": cid, "conf": conf, "xyxy": xyxy, "size_px": size_px})
    return out


def evaluate_predictions(
    gt_dir: Path,
    pred_dir: Path,
    img_sz: int = 512,
    iou_thresh: float = 0.45,
    conf_thresh: float = 0.0,
    small_px_range: Tuple[int, int] = (8, 25),
) -> Dict[str, object]:
    cfg = _load_cfg()
    classes = list(cfg.get("classes", ["orchard_tree", "palm_tree", "windbreak_tree"]))

    tp_by_cls: Dict[int, int] = {i: 0 for i in range(len(classes))}
    fp_by_cls: Dict[int, int] = {i: 0 for i in range(len(classes))}
    fn_by_cls: Dict[int, int] = {i: 0 for i in range(len(classes))}
    small_gt_total = 0
    small_gt_matched = 0
    neg_tiles = 0
    neg_fp_tiles = 0

    for gt_file in sorted(gt_dir.glob("*.txt")):
        gts = parse_yolo_file(gt_file, img_sz, is_pred=False)
        preds = [
            p for p in parse_yolo_file(pred_dir / gt_file.name, img_sz, is_pred=True) if p["conf"] >= conf_thresh
        ]
        preds.sort(key=lambda d: d["conf"], reverse=True)

        if not gts:
            neg_tiles += 1
            if preds:
                neg_fp_tiles += 1

        matched_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_idx: Optional[int] = None
            for gi, gt in enumerate(gts):
                if gi in matched_gt or gt["cls"] != pred["cls"]:
                    continue
                iou = box_iou(pred["xyxy"], gt["xyxy"])
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            cid = int(pred["cls"])
            if best_idx is not None:
                matched_gt.add(best_idx)
                tp_by_cls[cid] = tp_by_cls.get(cid, 0) + 1
            else:
                fp_by_cls[cid] = fp_by_cls.get(cid, 0) + 1

        for gi, gt in enumerate(gts):
            cid = int(gt["cls"])
            is_small = small_px_range[0] <= gt["size_px"] <= small_px_range[1]
            if is_small:
                small_gt_total += 1
                if gi in matched_gt:
                    small_gt_matched += 1
            if gi not in matched_gt:
                fn_by_cls[cid] = fn_by_cls.get(cid, 0) + 1

    tp_tot = sum(tp_by_cls.values())
    fp_tot = sum(fp_by_cls.values())
    fn_tot = sum(fn_by_cls.values())
    precision = tp_tot / max(1, tp_tot + fp_tot)
    recall = tp_tot / max(1, tp_tot + fn_tot)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    small_recall = (small_gt_matched / small_gt_total) if small_gt_total > 0 else None

    per_class = {}
    for cid, cname in enumerate(classes):
        tp, fp, fn = tp_by_cls.get(cid, 0), fp_by_cls.get(cid, 0), fn_by_cls.get(cid, 0)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        per_class[cname] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round((2 * p * r) / max(1e-9, p + r), 4),
        }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "small_crown_range_px": list(small_px_range),
        "small_crown_gt": small_gt_total,
        "small_crown_recall": round(small_recall, 4) if small_recall is not None else None,
        "hard_negative_tiles": neg_tiles,
        "hard_negative_fp_rate": round(neg_fp_tiles / max(1, neg_tiles), 4) if neg_tiles > 0 else 0.0,
        "per_class": per_class,
    }


if __name__ == "__main__":
    cfg = _load_cfg()
    parser = argparse.ArgumentParser(description="Evaluate tree detections including small-crown recall (8-25px).")
    parser.add_argument("--gt-dir", default="dataset/labels/val")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted YOLO .txt files (with conf)")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--iou", type=float, default=float(cfg.get("nmsIouThresh", 0.40)))
    parser.add_argument("--conf", type=float, default=float(cfg.get("aiModelConfThresh", 0.55)))
    args = parser.parse_args()

    s_rng = tuple(cfg.get("smallCrownPxRange", [8, 25]))
    metrics = evaluate_predictions(
        Path(args.gt_dir),
        Path(args.pred_dir),
        img_sz=args.imgsz,
        iou_thresh=args.iou,
        conf_thresh=args.conf,
        small_px_range=(int(s_rng[0]), int(s_rng[1])),
    )
    print(json.dumps(metrics, indent=2))
