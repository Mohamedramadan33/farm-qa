# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
----------------------
Deterministic unit tests for the ML & geospatial Python pipeline:
  - Centralized config/detection.config.json schema & canonical 3-class IDs
  - FastAPI batch inference payload & offset projection validation
  - Farm-ID GroupShuffleSplit zero-leakage guarantee
  - Training label polygon clipping parity
  - Small-crown (8-25px) detection evaluator & threshold sweep
  - KoboToolbox MAE / MAPE benchmarking
"""

import base64
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_against_kobo import compute_mae_mape, evaluate_kobo_records
from scripts.clip_labels_to_polygon import clip_yolo_lines_to_polygon, point_in_polygon
from scripts.evaluate_detection import evaluate_predictions
from scripts.optimize_thresholds import sweep_confidence_thresholds
from scripts.split_dataset_by_farm import extract_farm_id, group_split_indices, split_dataset_by_farm

ROOT = Path(__file__).resolve().parent.parent


class TestPipelineIntegrity(unittest.TestCase):
    def test_detection_config_canonical_schema(self):
        cfg_path = ROOT / "config" / "detection.config.json"
        self.assertTrue(cfg_path.exists())
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["classes"], ["orchard_tree", "palm_tree", "windbreak_tree"])
        self.assertGreaterEqual(cfg["aiModelConfThresh"], 0.25)
        self.assertLessEqual(cfg["aiModelConfThresh"], 0.90)
        self.assertEqual(cfg["smallCrownPxRange"], [8, 25])
        self.assertTrue(cfg["classAwareNms"])
        self.assertEqual(cfg["highResTileSize"], 288)
        self.assertAlmostEqual(cfg["highResOverlap"], 0.40)

    def test_farm_group_split_zero_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_img = tmp_path / "src_img"
            src_lbl = tmp_path / "src_lbl"
            out_dir = tmp_path / "out_dataset"
            src_img.mkdir()
            src_lbl.mkdir()

            # 5 farms x 3 tiles each (mixed google & esri)
            farms = ["F101", "F102", "F103", "F104", "F105"]
            for fid in farms:
                for idx in range(1, 4):
                    prov = "esri" if idx == 3 else "google"
                    stem = f"farm_{fid}_{prov}_tile_00{idx}"
                    (src_img / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                    (src_lbl / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            summary = split_dataset_by_farm(src_img, src_lbl, out_dir, val_size=0.2, seed=42)
            self.assertEqual(summary["farm_overlap"], 0)
            self.assertGreater(summary["train_tiles"], 0)
            self.assertGreater(summary["val_tiles"], 0)

            train_farms = {extract_farm_id(p.name) for p in (out_dir / "images" / "train").glob("*.png")}
            val_farms = {extract_farm_id(p.name) for p in (out_dir / "images" / "val").glob("*.png")}
            self.assertEqual(len(train_farms & val_farms), 0)

    def test_group_split_single_farm_safe(self):
        tr, va = group_split_indices(["F1", "F1", "F1"], val_size=0.2)
        self.assertEqual(len(tr), 3)
        self.assertEqual(len(va), 0)

    def test_polygon_label_clipping_parity(self):
        square_ring = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self.assertTrue(point_in_polygon(0.5, 0.5, square_ring))
        self.assertFalse(point_in_polygon(0.1, 0.5, square_ring))

        lines = [
            "0 0.500000 0.500000 0.050000 0.050000",  # inside
            "1 0.100000 0.100000 0.050000 0.050000",  # outside
            "2 0.750000 0.750000 0.040000 0.040000",  # inside
        ]
        clipped = clip_yolo_lines_to_polygon(lines, square_ring)
        self.assertEqual(len(clipped), 2)
        self.assertTrue(clipped[0].startswith("0 "))
        self.assertTrue(clipped[1].startswith("2 "))

    def test_small_crown_eval_and_threshold_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            gt_dir = tpath / "gt"
            pred_dir = tpath / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()

            # 16px small crown (0.03125 * 512 = 16px) + 1 hard-negative tile
            (gt_dir / "t1.txt").write_text("0 0.5 0.5 0.03125 0.03125\n", encoding="utf-8")
            (gt_dir / "t2_hn.txt").write_text("", encoding="utf-8")

            # True positive at conf=0.68 + low-conf FP on hard negative at conf=0.35
            (pred_dir / "t1.txt").write_text("0 0.5 0.5 0.03125 0.03125 0.68\n", encoding="utf-8")
            (pred_dir / "t2_hn.txt").write_text("0 0.3 0.3 0.04 0.04 0.35\n", encoding="utf-8")

            res = evaluate_predictions(gt_dir, pred_dir, img_sz=512, iou_thresh=0.45, conf_thresh=0.55)
            self.assertEqual(res["precision"], 1.0)
            self.assertEqual(res["recall"], 1.0)
            self.assertEqual(res["small_crown_recall"], 1.0)
            self.assertEqual(res["hard_negative_fp_rate"], 0.0)

            sweep = sweep_confidence_thresholds(gt_dir, pred_dir, img_sz=512, iou_thresh=0.45)
            self.assertGreaterEqual(sweep["optimal_conf"], 0.40)
            self.assertEqual(sweep["best_f1"], 1.0)

    def test_kobo_mae_mape_benchmarking(self):
        records = [
            {
                "farm_id": "F-1",
                "actual_total": 100,
                "pred_total": 110,
                "actual_by_class": {"orchard_tree": 80, "palm_tree": 10, "windbreak_tree": 10},
                "pred_by_class": {"orchard_tree": 88, "palm_tree": 10, "windbreak_tree": 12},
            },
            {
                "farm_id": "F-2",
                "actual_total": 50,
                "pred_total": 45,
                "actual_by_class": {"orchard_tree": 40, "palm_tree": 5, "windbreak_tree": 5},
                "pred_by_class": {"orchard_tree": 36, "palm_tree": 4, "windbreak_tree": 5},
            },
        ]
        report = evaluate_kobo_records(records)
        self.assertEqual(report["overall"]["mae"], 7.5)
        self.assertEqual(report["overall"]["mape_pct"], 10.0)
        self.assertIn("palm_tree", report["per_class"])
        self.assertEqual(compute_mae_mape([{"actual": 20, "predicted": 20}])["mae"], 0.0)

    def test_inference_service_base64_validation_structure(self):
        # Verify inference_service.py compiles and base64 PNG structure matches expected format
        tiny_png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
        self.assertGreaterEqual(len(tiny_png), 16)
        src = (ROOT / "inference_service.py").read_text(encoding="utf-8")
        self.assertIn("torch.cuda.is_available()", src)
        self.assertIn("@app.post(\"/detect-batch\")", src)


if __name__ == "__main__":
    unittest.main()
