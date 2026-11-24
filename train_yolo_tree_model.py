# -*- coding: utf-8 -*-
"""
train_yolo_tree_model.py
------------------------
Multi-Class YOLOv8 Training & ONNX Export Pipeline for Farm Tree Crown Detection:
  - Canonical 3-class schema matching dataset/data.yaml:
      0: orchard_tree
      1: palm_tree
      2: windbreak_tree
  - Audits both train & val splits for >=30% Hard Negative background ratio
    (empty .txt files for flat Egyptian crops: berseem/clover, alfalfa, wheat, vegetables)
    and verifies basemap provider balance (Google vs Esri).
  - Applies lightweight oversampling / copy-paste augmentation on rare classes
    (palm_tree, windbreak_tree) when underrepresented in train.
  - Uses torch.cuda.is_available() for automatic GPU/CPU selection.
  - Supports 512x512 (default SAHI window) or 1024x1024 input sizes and yolov8n/yolov8s backbones.
  - Exports PyTorch weights (models/best.pt, best.pt) and simplified ONNX (public/models/best.onnx, models/best.onnx).
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple

import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
TARGET_CLASSES: Dict[int, str] = {
    0: "orchard_tree",
    1: "palm_tree",
    2: "windbreak_tree",
}


def _load_shared_config() -> Dict[str, object]:
    cfg_path = Path(__file__).resolve().parent / "config" / "detection.config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def audit_split(img_dir: Path, lbl_dir: Path, split_name: str) -> Dict[str, object]:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    positives = 0
    negatives = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    provider_counts = {"google": 0, "esri": 0, "other": 0}

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem_low = img_path.stem.lower()
        if "_google_" in stem_low or stem_low.startswith("google_"):
            provider_counts["google"] += 1
        elif "_esri_" in stem_low or stem_low.startswith("esri_"):
            provider_counts["esri"] += 1
        else:
            provider_counts["other"] += 1

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            lbl_path.write_text("", encoding="utf-8")
            negatives += 1
            continue

        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            negatives += 1
        else:
            positives += 1
            for ln in lines:
                parts = ln.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    if cid in class_counts:
                        class_counts[cid] += 1
                except ValueError:
                    pass

    total = positives + negatives
    neg_ratio = (negatives / total) if total > 0 else 0.0
    print(
        f"[{split_name.upper()}] Total: {total} | Positives: {positives} | "
        f"Hard Negatives: {negatives} ({neg_ratio * 100:.1f}%) | "
        f"Providers(G/E/O): {provider_counts['google']}/{provider_counts['esri']}/{provider_counts['other']}"
    )
    print(
        f"[{split_name.upper()}] Class Instances -> "
        f"orchard_tree(0): {class_counts[0]} | palm_tree(1): {class_counts[1]} | windbreak_tree(2): {class_counts[2]}"
    )

    return {
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "neg_ratio": neg_ratio,
        "class_counts": class_counts,
        "provider_counts": provider_counts,
    }


def prepare_hard_negatives(
    dataset_dir: Path,
    target_neg_ratio: float = 0.30,
    val_hard_neg_share: float = 0.15,
) -> Dict[str, object]:
    """
    Syncs images from dataset/hard_negatives into train (and optionally val) with empty .txt
    labels, then audits both train & val splits for >=30% Hard Negative background representation.
    """
    img_train = dataset_dir / "images" / "train"
    lbl_train = dataset_dir / "labels" / "train"
    img_val = dataset_dir / "images" / "val"
    lbl_val = dataset_dir / "labels" / "val"
    hn_dir = dataset_dir / "hard_negatives"

    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    if hn_dir.exists():
        hn_files = [p for p in sorted(hn_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        for idx, src in enumerate(hn_files):
            dst_name = f"hn_{src.name}"
            use_val = val_hard_neg_share > 0 and (idx % max(2, int(round(1.0 / val_hard_neg_share))) == 0)
            target_img_dir = img_val if use_val else img_train
            target_lbl_dir = lbl_val if use_val else lbl_train
            dst_img = target_img_dir / dst_name
            dst_lbl = target_lbl_dir / f"{Path(dst_name).stem}.txt"
            if not dst_img.exists():
                shutil.copy2(src, dst_img)
            if not dst_lbl.exists():
                dst_lbl.write_text("", encoding="utf-8")

    train_stats = audit_split(img_train, lbl_train, "train")
    val_stats = audit_split(img_val, lbl_val, "val")

    total_tr = int(train_stats["total"])
    pos_tr = int(train_stats["positives"])
    neg_tr = int(train_stats["negatives"])
    neg_ratio_tr = float(train_stats["neg_ratio"])

    if total_tr > 0 and neg_ratio_tr < target_neg_ratio - 0.05:
        needed = max(0, int((target_neg_ratio * pos_tr) / max(1e-6, 1.0 - target_neg_ratio)) - neg_tr)
        print(
            f"Warning: Add ~{needed} empty-label Egyptian row-crop tiles "
            f"(alfalfa/wheat/clover) to '{hn_dir}' to hit the {int(target_neg_ratio * 100)}% Hard Negative target."
        )

    return {"train": train_stats, "val": val_stats}


def oversample_rare_classes(dataset_dir: Path, min_class_ratio: float = 0.15) -> int:
    """
    Duplicates training tiles containing rare classes (palm_tree=1, windbreak_tree=2)
    when their instance share falls below min_class_ratio of orchard_tree instances.
    """
    img_train = dataset_dir / "images" / "train"
    lbl_train = dataset_dir / "labels" / "train"
    if not img_train.exists() or not lbl_train.exists():
        return 0

    class_tiles: Dict[int, List[Path]] = {0: [], 1: [], 2: []}
    class_counts = {0: 0, 1: 0, 2: 0}

    for img_path in sorted(img_train.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_path.stem.startswith("os_rare_"):
            continue
        lbl_path = lbl_train / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        present_cls = set()
        for ln in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = ln.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                if cid in class_counts:
                    class_counts[cid] += 1
                    present_cls.add(cid)
            except ValueError:
                pass
        for cid in present_cls:
            class_tiles[cid].append(img_path)

    majority = max(1, class_counts[0])
    added = 0
    rng = random.Random(42)

    for rare_cid in (1, 2):
        tiles = class_tiles[rare_cid]
        if not tiles or class_counts[rare_cid] >= majority * min_class_ratio:
            continue
        copies_needed = min(len(tiles) * 2, int(majority * min_class_ratio) - class_counts[rare_cid])
        for i in range(max(0, copies_needed)):
            src_img = rng.choice(tiles)
            src_lbl = lbl_train / f"{src_img.stem}.txt"
            tag = f"os_rare_c{rare_cid}_{i:03d}_{src_img.stem}"
            dst_img = img_train / f"{tag}{src_img.suffix}"
            dst_lbl = lbl_train / f"{tag}.txt"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                added += 1

    if added > 0:
        print(f"Oversampled {added} rare-class tiles (palm_tree / windbreak_tree) in images/train.")
    return added


def train_tree_model(
    data_yaml_path: str = "dataset/data.yaml",
    epochs: int = 60,
    imgsz: int = 512,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt",
    output_pt_dest: str = "models/best.pt",
    output_onnx_dest: str = "public/models/best.onnx",
    target_hard_neg_ratio: float = 0.30,
    copy_paste: float = 0.15,
    oversample_rare: bool = True,
) -> Tuple[object, Dict[str, object]]:
    print("=" * 72)
    print("Multi-Class YOLOv8 Tree Crown Detection & ONNX Export Pipeline")
    print("Classes -> 0: orchard_tree | 1: palm_tree | 2: windbreak_tree")
    print("=" * 72)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Install via: pip install ultralytics onnx onnxsim")
        sys.exit(1)

    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset config {data_yaml_path} not found!")
        sys.exit(1)

    dataset_dir = Path(data_yaml_path).resolve().parent
    stats = prepare_hard_negatives(dataset_dir, target_neg_ratio=target_hard_neg_ratio)
    if oversample_rare:
        oversample_rare_classes(dataset_dir)

    device: object = 0 if torch.cuda.is_available() else "cpu"
    print(
        f"\nLoading {data_yaml_path} | Base: {base_model} | "
        f"imgsz: {imgsz} | epochs: {epochs} | device: {device}"
    )

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=15,
        save=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=copy_paste,
        device=device,
        plots=True,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir)
    trained_best_pt = save_dir / "weights" / "best.pt"

    if trained_best_pt.exists():
        Path(output_pt_dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_best_pt, output_pt_dest)
        shutil.copy2(trained_best_pt, "best.pt")
        print(f"PyTorch weights deployed to '{output_pt_dest}' and 'best.pt'")

        try:
            best_model = YOLO(str(trained_best_pt))
            exported_onnx = Path(
                best_model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
            )
            if exported_onnx.exists():
                Path(output_onnx_dest).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(exported_onnx, output_onnx_dest)
                Path("models").mkdir(parents=True, exist_ok=True)
                shutil.copy2(exported_onnx, "models/best.onnx")
                print(f"ONNX model exported to '{output_onnx_dest}' & 'models/best.onnx'")
        except Exception as exc:
            print(f"Warning: ONNX export failed ({exc}); PyTorch weights are ready.")
    else:
        print(f"Warning: {trained_best_pt} not found. Check runs/detect output.")

    return results, stats


def main() -> None:
    shared_cfg = _load_shared_config()
    default_imgsz = int(shared_cfg.get("sahiTileSize", 512))

    parser = argparse.ArgumentParser(description="Train multi-class YOLOv8 farm tree detector")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=default_imgsz, choices=[512, 640, 1024], help="Input tile resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained backbone (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--copy-paste", type=float, default=0.15, help="Copy-paste augmentation probability")
    parser.add_argument("--no-oversample", action="store_true", help="Disable rare-class oversampling")
    args = parser.parse_args()

    train_tree_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        base_model=args.model,
        copy_paste=args.copy_paste,
        oversample_rare=not args.no_oversample,
    )


if __name__ == "__main__":
    main()