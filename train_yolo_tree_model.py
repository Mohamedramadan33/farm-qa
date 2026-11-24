# -*- coding: utf-8 -*-

## """
train\_yolo\_tree\_model.py

Multi-Class YOLOv8 Nano Training &amp; ONNX Export Pipeline for Farm Tree Crown Detection:

- Multi-class schema matching dataset/data.yaml:
  0: orchard\_tree
  1: palm\_tree
  2: windbreak\_tree
- Enforces &amp; audits a 30% Hard Negative background ratio (Egyptian row crops:
alfalfa, wheat, clover, vegetables with empty .txt label files) to prevent false positives.
- Trains at 512x512 resolution matching SAHI 512x512 sliding-window inference.
- Exports both PyTorch weights (models/best.pt, best.pt) and ONNX (public/models/best.onnx)
for browser onnxruntime-web and backend API inference.
"""

import os
import shutil
import sys

import torch
from pathlib import Path
from typing import Dict, Tuple

IMAGE\_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
TARGET\_CLASSES = {0: "orchard\_tree", 1: "palm\_tree", 2: "windbreak\_tree"}

def prepare\_hard\_negatives(dataset\_dir: Path, target\_neg\_ratio: float = 0.30) -&gt; Dict\[str, int\]:
    """
    Ensures every image in dataset/hard\_negatives (if present) is copied into
    images/train with an empty .txt file in labels/train, and audits the
    resulting ratio of Hard Negative background images (default target &gt;= 30%).
    """
    img\_train = dataset\_dir / "images" / "train"
    lbl\_train = dataset\_dir / "labels" / "train"
    hn\_dir = dataset\_dir / "hard\_negatives"

```
img_train.mkdir(parents=True, exist_ok=True)
lbl_train.mkdir(parents=True, exist_ok=True)

# Create empty .txt label files for any hard-negative crop images placed in dataset/hard_negatives
if hn_dir.exists():
    for src in sorted(hn_dir.iterdir()):
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dst_name = f"hn_{src.name}"
        dst_img = img_train / dst_name
        dst_lbl = lbl_train / f"{Path(dst_name).stem}.txt"
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        if not dst_lbl.exists():
            dst_lbl.write_text("", encoding="utf-8")

positives = 0
negatives = 0
class_counts = {0: 0, 1: 0, 2: 0}

for img_path in sorted(img_train.iterdir()):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue
    lbl_path = lbl_train / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        # Unlabeled image in train -> treat explicitly as empty background label
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
            if parts:
                try:
                    cid = int(parts[0])
                    if cid in class_counts:
                        class_counts[cid] += 1
                except ValueError:
                    pass

total = positives + negatives
neg_ratio = (negatives / total) if total > 0 else 0.0
print(f"📊 Dataset Audit -> Total Train Images: {total} | Positives: {positives} | Hard Negatives (empty .txt): {negatives} ({neg_ratio * 100:.1f}%)")
print(f"🏷️ Class Instances -> orchard_tree(0): {class_counts[0]} | palm_tree(1): {class_counts[1]} | windbreak_tree(2): {class_counts[2]}")
if total > 0 and neg_ratio < target_neg_ratio - 0.05:
    needed = max(0, int((target_neg_ratio * positives) / max(1e-6, 1.0 - target_neg_ratio)) - negatives)
    print(
        f"⚠️ Recommendation: Add ~{needed} empty-label Egyptian row-crop tiles "
        f"(alfalfa/wheat/clover) to '{hn_dir}' to reach the 30% Hard Negative target."
    )

return {"total": total, "positives": positives, "negatives": negatives, **class_counts}
```

def train\_tree\_model(
    data\_yaml\_path: str = "dataset/data.yaml",
    epochs: int = 50,
    imgsz: int = 512,
    batch\_size: int = 16,
    base\_model: str = "yolov8n.pt",
    output\_pt\_dest: str = "models/best.pt",
    output\_onnx\_dest: str = "public/models/best.onnx",
    target\_hard\_neg\_ratio: float = 0.30,
) -&gt; Tuple\[object, Dict\[str, int\]\]:
    print("=" \* 72)
    print("🌲 Multi-Class YOLOv8 Nano Tree Detection &amp; ONNX Export Pipeline")
    print("   Classes: 0: orchard\_tree | 1: palm\_tree | 2: windbreak\_tree")
    print("=" \* 72)

```
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics is not installed.")
    print("💡 Install it via: pip install ultralytics onnx onnxsim")
    sys.exit(1)

if not os.path.exists(data_yaml_path):
    print(f"❌ Error: Dataset config {data_yaml_path} not found!")
    sys.exit(1)

dataset_dir = Path(data_yaml_path).resolve().parent
stats = prepare_hard_negatives(dataset_dir, target_neg_ratio=target_hard_neg_ratio)

print(f"\n📁 Loading dataset config: {data_yaml_path}")
print(f"🚀 Base model: {base_model} | SAHI tile size: {imgsz}x{imgsz} | Epochs: {epochs}")

model = YOLO(base_model)
results = model.train(
    data=data_yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    patience=12,
    save=True,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=180.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.8,
    device=0 if torch.cuda.is_available() else "cpu",
    plots=True,
    verbose=True,
)

save_dir = Path(model.trainer.save_dir)
trained_best_pt = save_dir / "weights" / "best.pt"

if trained_best_pt.exists():
    Path(output_pt_dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_best_pt, output_pt_dest)
    shutil.copy2(trained_best_pt, "best.pt")
    print(f"✅ PyTorch weights deployed to '{output_pt_dest}' and 'best.pt'")

    # Export static 512x512 ONNX for onnxruntime-web & Node backend
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
            print(f"🌐 ONNX model exported to '{output_onnx_dest}' & 'models/best.onnx' for Web AI!")
    except Exception as exc:
        print(f"⚠️ ONNX export failed ({exc}); PyTorch weights are still ready.")
else:
    print(f"⚠️ Warning: {trained_best_pt} not found. Check runs/detect output.")

return results, stats
```

if **name** == "**main**":
    train\_tree\_model()