# -*- coding: utf-8 -*-
"""
scripts/split_dataset_by_farm.py
---------------------------------
Farm-level GroupShuffleSplit dataset splitter so all tiles from a single farm
stay strictly in either `train` or `val` (zero spatial tile leakage).
Supports Google & Esri provider tagging/auditing.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def extract_farm_id(filename: str, meta_dir: Optional[Path] = None) -> str:
    stem = Path(filename).stem
    if meta_dir is not None:
        meta_file = meta_dir / f"{stem}.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                if data.get("farm_id"):
                    return str(data["farm_id"])
            except Exception:
                pass
    m = re.match(r"^(?:hn_)?(?:cp_c\d+_\d+_)?(?:farm_)?([A-Za-z0-9-]+?)_(?:google|esri|tile)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].lower() == "farm":
        return parts[1]
    return stem


def extract_provider(filename: str) -> str:
    low = Path(filename).stem.lower()
    if "esri" in low:
        return "esri"
    if "google" in low:
        return "google"
    return "unknown"


def group_split_indices(
    groups: List[str],
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Splits item indices by group ID with zero group overlap between train and val.
    Uses sklearn GroupShuffleSplit when available, with deterministic pure-Python fallback.
    """
    if not groups:
        return [], []
    unique_groups = sorted(set(groups))
    if len(unique_groups) == 1:
        return list(range(len(groups))), []

    try:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(splitter.split(range(len(groups)), groups=groups))
        return train_idx.tolist(), val_idx.tolist()
    except ImportError:
        scored = sorted(
            unique_groups,
            key=lambda g: hashlib.sha256(f"{random_state}:{g}".encode("utf-8")).hexdigest(),
        )
        n_val_groups = max(1, min(len(unique_groups) - 1, int(round(len(unique_groups) * val_size))))
        val_groups = set(scored[:n_val_groups])
        train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
        val_idx = [i for i, g in enumerate(groups) if g in val_groups]
        return train_idx, val_idx


def split_dataset_by_farm(
    source_images_dir: Path,
    source_labels_dir: Path,
    output_dataset_dir: Path,
    meta_dir: Optional[Path] = None,
    val_size: float = 0.2,
    seed: int = 42,
    move_files: bool = False,
) -> Dict[str, object]:
    images = [p for p in sorted(source_images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    groups = [extract_farm_id(p.name, meta_dir=meta_dir) for p in images]
    train_idx, val_idx = group_split_indices(groups, val_size=val_size, random_state=seed)

    for split in ("train", "val"):
        (output_dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if move_files else shutil.copy2
    split_farms: Dict[str, set] = {"train": set(), "val": set()}
    split_providers: Dict[str, Counter] = {"train": Counter(), "val": Counter()}

    def _place(indices: Iterable[int], split_name: str) -> int:
        cnt = 0
        for i in indices:
            img_src = images[i]
            lbl_src = source_labels_dir / f"{img_src.stem}.txt"
            img_dst = output_dataset_dir / "images" / split_name / img_src.name
            lbl_dst = output_dataset_dir / "labels" / split_name / f"{img_src.stem}.txt"
            if img_src.resolve() != img_dst.resolve():
                transfer(str(img_src), str(img_dst))
            if lbl_src.exists():
                if lbl_src.resolve() != lbl_dst.resolve():
                    transfer(str(lbl_src), str(lbl_dst))
            else:
                lbl_dst.write_text("", encoding="utf-8")
            split_farms[split_name].add(groups[i])
            split_providers[split_name][extract_provider(img_src.name)] += 1
            cnt += 1
        return cnt

    n_train = _place(train_idx, "train")
    n_val = _place(val_idx, "val")
    overlap = split_farms["train"] & split_farms["val"]
    assert len(overlap) == 0, f"Farm leakage detected across splits: {overlap}"

    summary = {
        "train_tiles": n_train,
        "val_tiles": n_val,
        "train_farms": len(split_farms["train"]),
        "val_farms": len(split_farms["val"]),
        "farm_overlap": len(overlap),
        "train_providers": dict(split_providers["train"]),
        "val_providers": dict(split_providers["val"]),
    }
    print(f"✅ Farm-grouped split complete: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split satellite tree tiles by Farm ID without leakage.")
    parser.add_argument("--src-images", default="dataset/feedback_queue/images")
    parser.add_argument("--src-labels", default="dataset/feedback_queue/labels")
    parser.add_argument("--meta-dir", default="dataset/feedback_queue/meta")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()

    meta_p = Path(args.meta_dir)
    split_dataset_by_farm(
        Path(args.src_images),
        Path(args.src_labels),
        Path(args.out_dir),
        meta_dir=meta_p if meta_p.exists() else None,
        val_size=args.val_size,
        seed=args.seed,
        move_files=args.move,
    )
