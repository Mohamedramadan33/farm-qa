# -*- coding: utf-8 -*-
"""
train_yolo_tree_model.py
-------------------------
Turnkey YOLOv8 Nano Training Script for Farm Tree Crown Detection.
- Fast, lightweight training (< 10 minutes on Google Colab or modern CPU/GPU).
- Automatically exports the trained weights as 'best.pt' directly to the app directory.
- Fully compatible with Ultralytics YOLOv8 architecture.
"""

import os
import shutil
import sys
from pathlib import Path

def train_tree_model(
    data_yaml_path: str = "dataset/data.yaml",
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt",
    output_dest: str = "best.pt",
):
    print("=" * 65)
    print("🌲 YOLOv8 Nano Tree Crown Detection Training Pipeline")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Error: ultralytics is not installed.")
        print("💡 Install it via: pip install ultralytics")
        sys.exit(1)

    # 1. Check data.yaml
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: Dataset config {data_yaml_path} not found!")
        sys.exit(1)

    print(f"📁 Loading dataset config: {data_yaml_path}")
    print(f"🚀 Initializing base model: {base_model}")
    print(f"⚡ Target epochs: {epochs} | Image size: {imgsz} | Batch: {batch_size}")

    # 2. Initialize YOLOv8 Nano model
    model = YOLO(base_model)

    # 3. Start Training
    print("\n⏳ Starting model training (Fast YOLOv8n)...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=10,
        save=True,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        plots=True,
        verbose=True,
    )

    # 4. Locate best.pt from run outputs
    save_dir = Path(model.trainer.save_dir)
    trained_best_pt = save_dir / "weights" / "best.pt"

    if trained_best_pt.exists():
        print(f"\n✅ Training completed successfully!")
        print(f"📦 Source weights: {trained_best_pt}")
        shutil.copy(trained_best_pt, output_dest)
        print(f"🎯 Successfully deployed: '{output_dest}' in application directory!")
        print("🚀 Streamlit will now automatically load and use this AI model for farm audits!")
    else:
        print(f"⚠️ Warning: {trained_best_pt} not found. Check runs/detect output.")

    return results

if __name__ == "__main__":
    train_tree_model()
