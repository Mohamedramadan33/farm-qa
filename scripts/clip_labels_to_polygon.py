# -*- coding: utf-8 -*-
"""
scripts/clip_labels_to_polygon.py
----------------------------------
Ensures training YOLO label files apply the exact same polygon-center clipping
used at inference time in polygonCounter.ts / ndvi_tree_analyzer.py.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple


def point_in_polygon(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    if len(ring) < 3:
        return True
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi)) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def clip_yolo_lines_to_polygon(
    yolo_lines: Sequence[str],
    ring_norm_xy: Sequence[Tuple[float, float]],
) -> List[str]:
    """
    Filters normalized YOLO label lines ('cls cx cy w h') so only tree centers
    strictly inside `ring_norm_xy` (0..1 normalized tile coordinates) are retained.
    """
    kept: List[str] = []
    for raw in yolo_lines:
        ln = raw.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cx = float(parts[1])
            cy = float(parts[2])
        except ValueError:
            continue
        if point_in_polygon(cx, cy, ring_norm_xy):
            kept.append(ln)
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip YOLO label file using normalized polygon JSON.")
    parser.add_argument("--label", required=True, help="Path to YOLO .txt label file")
    parser.add_argument("--polygon-json", required=True, help="JSON file containing [[x,y],...] normalized 0..1 ring")
    parser.add_argument("--out", default="", help="Optional output path (defaults to overwriting --label)")
    args = parser.parse_args()

    ring_raw = json.loads(Path(args.polygon_json).read_text(encoding="utf-8"))
    ring = [(float(pt[0]), float(pt[1])) for pt in ring_raw]
    src_lines = Path(args.label).read_text(encoding="utf-8").splitlines()
    clipped = clip_yolo_lines_to_polygon(src_lines, ring)
    out_path = Path(args.out) if args.out else Path(args.label)
    out_path.write_text("\n".join(clipped), encoding="utf-8")
    print(f"Clipped {len(src_lines)} -> {len(clipped)} annotations into {out_path}")
