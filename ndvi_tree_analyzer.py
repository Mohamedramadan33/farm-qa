# -*- coding: utf-8 -*-
"""
ndvi_tree_analyzer.py
---------------------
Two-Stage Geospatial Structural Tree Crown & Cast-Shadow Detection Engine:
- Stage 1: Flat-Crop Texture & Shadow Rejection (4m x 4m grid cells masking out smooth crop basins).
- Stage 2:
    * Primary AI Path: SAHI 512x512 sliding window (20% overlap, NMS IoU=0.45, conf>=0.55) with multi-class YOLO.
    * Algorithmic Fallback: Multi-Scale DoG crown blobs (2.0m..6.5m diameter, circularity > 0.65)
      paired with directional adjacent cast shadows (>= 35% luminance drop relative to crown center).
- UTF-8 Pillow text renderer using dual Arabic + Latin TTF fonts with arabic_reshaper & python-bidi.
"""

from collections import deque
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_get_display
    HAS_ARABIC_SHAPER = True
except ImportError:
    arabic_reshaper = None
    _bidi_get_display = None
    HAS_ARABIC_SHAPER = False

HAS_YOLO = False
_YOLO_MODEL_CACHE: Dict[str, Any] = {}
_FONT_CACHE: Dict[Tuple[str, int], Any] = {}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_shared_detection_config() -> Dict[str, Any]:
    cfg_path = os.path.join(_BASE_DIR, "config", "detection.config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


_CFG = _load_shared_detection_config()
AI_MODEL_CONF_THRESH = float(_CFG.get("aiModelConfThresh", 0.55))
SAHI_TILE_SIZE = int(_CFG.get("sahiTileSize", 512))
SAHI_OVERLAP = float(_CFG.get("sahiOverlap", 0.20))
NMS_IOU_THRESH = float(_CFG.get("nmsIouThresh", 0.40))
CLASS_AWARE_NMS = bool(_CFG.get("classAwareNms", True))
CROSS_CLASS_SUPPRESSION_IOU = float(_CFG.get("crossClassSuppressionIou", 0.85))
MIN_CAST_SHADOW_DROP = float(_CFG.get("minCastShadowDrop", 0.35))
MIN_BLOB_CIRCULARITY = float(_CFG.get("minWatershedCircularity", 0.68))
MIN_CROWN_DIAM_M = float(_CFG.get("minCrownDiamM", 1.8))
MAX_CROWN_DIAM_M = float(_CFG.get("maxCrownDiamM", 7.0))
CROP_SHEET_AREA_M2 = float(_CFG.get("cropSheetAreaM2", 85.0))
MIN_CROWN_SPACING_M = float(_CFG.get("minCrownSpacingM", 2.5))

TREE_CLASS_NAMES: Dict[int, str] = {
    0: "orchard_tree",
    1: "palm_tree",
    2: "windbreak_tree",
}
TREE_CLASS_AR: Dict[str, str] = {
    "orchard_tree": "شجرة بستان",
    "palm_tree": "نخيل",
    "windbreak_tree": "مصدات رياح",
}

_TREE_CLASS_KEYWORDS = {
    "tree", "trees", "palm", "palms", "canopy", "crown",
    "orchard", "orchard_tree", "palm_tree", "windbreak_tree", "windbreak",
    "shrub", "sapling", "wood", "شجرة", "أشجار", "نخيل",
}

_ARABIC_FONT_CANDIDATES = [
    os.path.join(_BASE_DIR, "assets", "fonts", "NotoSansArabic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    os.path.join(_BASE_DIR, "assets", "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_LATIN_FONT_CANDIDATES = [
    os.path.join(_BASE_DIR, "assets", "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    os.path.join(_BASE_DIR, "assets", "fonts", "NotoSansArabic-Regular.ttf"),
]


def _load_ttf(candidates: List[str], kind: str, size: int) -> Any:
    key = (kind, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                fnt = ImageFont.truetype(fp, size=size)
                _FONT_CACHE[key] = fnt
                return fnt
            except Exception:
                continue
    fnt = ImageFont.load_default()
    _FONT_CACHE[key] = fnt
    return fnt


def _get_arabic_pil_font(size: int = 14) -> Any:
    return _load_ttf(_ARABIC_FONT_CANDIDATES, "ar", size)


def _get_latin_pil_font(size: int = 14) -> Any:
    return _load_ttf(_LATIN_FONT_CANDIDATES, "lat", size)


def _format_arabic_text(text: str) -> str:
    """Reshapes Arabic ligatures and applies BiDi ordering so PIL renders legible Arabic."""
    if not text:
        return text
    if HAS_ARABIC_SHAPER and arabic_reshaper is not None and _bidi_get_display is not None:
        try:
            reshaped = arabic_reshaper.reshape(str(text))
            return _bidi_get_display(reshaped)
        except Exception:
            return str(text)
    return str(text)


def _is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def _draw_mixed_bidi_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fill: Tuple[int, int, int] = (0, 255, 200),
    size: int = 14,
) -> None:
    """
    Renders reshaped mixed Arabic + ASCII/numeric/% text cleanly in Pillow by routing
    Arabic glyphs to NotoSansArabic-Regular.ttf and symbols/digits (%, (, ), |) to DejaVuSans.ttf,
    eliminating [▯] missing-glyph boxes.
    """
    visual = _format_arabic_text(text)
    ar_font = _get_arabic_pil_font(size)
    lat_font = _get_latin_pil_font(size)

    runs: List[Tuple[bool, str]] = []
    for ch in visual:
        is_ar = _is_arabic_char(ch)
        if not runs or runs[-1][0] != is_ar:
            runs.append((is_ar, ch))
        else:
            runs[-1] = (is_ar, runs[-1][1] + ch)

    cur_x, y = xy
    for is_ar, segment in runs:
        fnt = ar_font if is_ar else lat_font
        draw.text((cur_x, y), segment, fill=fill, font=fnt)
        try:
            adv = draw.textlength(segment, font=fnt)
        except Exception:
            bbox = draw.textbbox((0, 0), segment, font=fnt)
            adv = max(1, bbox[2] - bbox[0])
        cur_x += adv


def _get_validated_tree_yolo(model_path: str) -> Tuple[Optional[Any], str]:
    """
    Loads YOLO model (cached & lazy-imported) and verifies it defines tree/canopy classes.
    """
    global HAS_YOLO
    if not model_path or not os.path.exists(model_path):
        return None, "no_model"
    try:
        from ultralytics import YOLO
        HAS_YOLO = True
    except ImportError:
        HAS_YOLO = False
        return None, "ultralytics_not_installed"
    try:
        model = _YOLO_MODEL_CACHE.get(model_path)
        if model is None:
            model = YOLO(model_path)
            _YOLO_MODEL_CACHE[model_path] = model

        names_raw = getattr(model, "names", {}) or {}
        if isinstance(names_raw, dict):
            cls_names = [str(v).strip().lower() for v in names_raw.values()]
        elif isinstance(names_raw, (list, tuple)):
            cls_names = [str(v).strip().lower() for v in names_raw]
        else:
            cls_names = []

        has_tree_cls = any(any(kw in cname for kw in _TREE_CLASS_KEYWORDS) for cname in cls_names)
        if not has_tree_cls:
            return None, f"incompatible_classes:{','.join(cls_names)}"
        return model, ""
    except Exception as exc:
        return None, f"load_error:{exc}"


def compute_ndvi_or_vari(
    image_input: Union[np.ndarray, Image.Image],
    nir_channel: Optional[np.ndarray] = None,
    red_channel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes vegetation health index matrix [-1.0 to 1.0] ONLY for coloring/health telemetry
    of already-detected tree crowns (never used for proposing tree locations).
    """
    if nir_channel is not None and red_channel is not None:
        nir = nir_channel.astype(np.float32)
        red = red_channel.astype(np.float32)
        return np.clip((nir - red) / (nir + red + 1e-7), -1.0, 1.0)

    if isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("RGB"), dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
    else:
        raise ValueError("image_input must be PIL Image or NumPy array")

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    denom = g + r - b
    denom = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0 + 1e-5, denom)
    vari = (g - r) / denom
    gli = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-5)
    return np.clip(0.5 * np.clip(vari, -1.0, 1.0) + 0.5 * np.clip(gli, -1.0, 1.0), -1.0, 1.0)


def generate_ndvi_colormap_image(ndvi_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> Image.Image:
    h, w = ndvi_matrix.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    norm_val = np.clip((ndvi_matrix + 0.1) / 0.6, 0.0, 1.0)

    low_mask = norm_val < 0.5
    high_mask = ~low_mask

    t_low = norm_val[low_mask] * 2.0
    colored[low_mask, 0] = (215 + t_low * (254 - 215)).astype(np.uint8)
    colored[low_mask, 1] = (48 + t_low * (224 - 48)).astype(np.uint8)
    colored[low_mask, 2] = (39 + t_low * (139 - 39)).astype(np.uint8)

    t_high = (norm_val[high_mask] - 0.5) * 2.0
    colored[high_mask, 0] = (254 - t_high * (254 - 26)).astype(np.uint8)
    colored[high_mask, 1] = (224 - t_high * (224 - 152)).astype(np.uint8)
    colored[high_mask, 2] = (139 - t_high * (139 - 80)).astype(np.uint8)

    if mask is not None:
        outside = ~mask
        colored[outside] = (colored[outside] * 0.45).astype(np.uint8)

    return Image.fromarray(colored)


def _generate_sahi_slices(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    width_px: int,
    height_px: int,
    slice_size: int = SAHI_TILE_SIZE,
    overlap: float = SAHI_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    s_size = max(128, int(slice_size))
    ov = float(np.clip(overlap, 0.0, 0.50))
    stride = max(64, int(round(s_size * (1.0 - ov))))

    bx1 = max(0, min_x)
    by1 = max(0, min_y)
    bx2 = min(width_px, max_x + 1)
    by2 = min(height_px, max_y + 1)

    if (bx2 - bx1) <= s_size and (by2 - by1) <= s_size:
        return [(bx1, by1, bx2, by2)]

    xs: List[int] = []
    x = bx1
    while x < bx2:
        x_end = min(width_px, x + s_size)
        x_start = max(0, x_end - s_size)
        if not xs or xs[-1] != x_start:
            xs.append(x_start)
        if x_end >= bx2:
            break
        x += stride

    ys: List[int] = []
    y = by1
    while y < by2:
        y_end = min(height_px, y + s_size)
        y_start = max(0, y_end - s_size)
        if not ys or ys[-1] != y_start:
            ys.append(y_start)
        if y_end >= by2:
            break
        y += stride

    return [
        (sx, sy, min(width_px, sx + s_size), min(height_px, sy + s_size))
        for sy in ys
        for sx in xs
    ]


def _box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    a1 = float(max(1, (b1[2] - b1[0]) * (b1[3] - b1[1])))
    a2 = float(max(1, (b2[2] - b2[0]) * (b2[3] - b2[1])))
    return inter / max(1e-6, a1 + a2 - inter)


def _apply_sahi_wbf_nms(
    candidates: List[Dict[str, Any]],
    iou_thresh: float = NMS_IOU_THRESH,
    min_dist_px: float = 8.0,
    class_aware: bool = CLASS_AWARE_NMS,
    cross_class_iou: float = CROSS_CLASS_SUPPRESSION_IOU,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: float(c.get("conf", 0.5)), reverse=True)
    clusters: List[Dict[str, Any]] = []

    for cand in ordered:
        cx, cy = cand["center"]
        cbox = cand["bbox"]
        cconf = max(0.01, float(cand.get("conf", 0.5)))
        crad = float(cand.get("crown_radius", 4))
        c_cls_id = int(cand.get("class_id", 0))
        c_cls_name = str(cand.get("class_name", TREE_CLASS_NAMES.get(c_cls_id, "orchard_tree")))

        matched_cluster = None
        for cl in clusters:
            dist = math.hypot(cx - cl["cx"], cy - cl["cy"])
            iou = _box_iou(cbox, cl["bbox"])
            same_class = int(cl["class_id"]) == c_cls_id
            if class_aware and not same_class:
                if iou >= cross_class_iou:
                    matched_cluster = cl
                    break
            elif dist < min_dist_px or iou >= iou_thresh:
                matched_cluster = cl
                break

        if matched_cluster is None:
            clusters.append({
                "w_sum": cconf,
                "max_conf": cconf,
                "wx1": cbox[0] * cconf,
                "wy1": cbox[1] * cconf,
                "wx2": cbox[2] * cconf,
                "wy2": cbox[3] * cconf,
                "wcx": cx * cconf,
                "wcy": cy * cconf,
                "wrad": crad * cconf,
                "cx": float(cx),
                "cy": float(cy),
                "bbox": cbox,
                "class_id": c_cls_id,
                "class_name": c_cls_name,
                "circularity": cand.get("circularity"),
                "crown_diameter_m": cand.get("crown_diameter_m"),
            })
        else:
            mc = matched_cluster
            if cconf > mc["max_conf"]:
                mc["class_id"] = c_cls_id
                mc["class_name"] = c_cls_name
                mc["circularity"] = cand.get("circularity", mc.get("circularity"))
                mc["crown_diameter_m"] = cand.get("crown_diameter_m", mc.get("crown_diameter_m"))
            mc["w_sum"] += cconf
            mc["max_conf"] = max(mc["max_conf"], cconf)
            mc["wx1"] += cbox[0] * cconf
            mc["wy1"] += cbox[1] * cconf
            mc["wx2"] += cbox[2] * cconf
            mc["wy2"] += cbox[3] * cconf
            mc["wcx"] += cx * cconf
            mc["wcy"] += cy * cconf
            mc["wrad"] += crad * cconf
            w = mc["w_sum"]
            mc["cx"] = mc["wcx"] / w
            mc["cy"] = mc["wcy"] / w
            mc["bbox"] = (
                int(round(mc["wx1"] / w)),
                int(round(mc["wy1"] / w)),
                int(round(mc["wx2"] / w)),
                int(round(mc["wy2"] / w)),
            )

    fused: List[Dict[str, Any]] = []
    for cl in clusters:
        w = cl["w_sum"]
        fused.append({
            "bbox": (
                int(round(cl["wx1"] / w)),
                int(round(cl["wy1"] / w)),
                int(round(cl["wx2"] / w)),
                int(round(cl["wy2"] / w)),
            ),
            "center": (int(round(cl["wcx"] / w)), int(round(cl["wcy"] / w))),
            "conf": round(min(1.0, cl["max_conf"]), 4),
            "crown_radius": max(4, int(round(cl["wrad"] / w))),
            "class_id": cl["class_id"],
            "class_name": cl["class_name"],
            "circularity": cl.get("circularity"),
            "crown_diameter_m": cl.get("crown_diameter_m"),
        })
    return fused


# ---------------------------------------------------------------------------
# 4-Gate Canopy Watershed Algorithm (Offline Morphological Fallback)
# ---------------------------------------------------------------------------

def _rgb_to_hsv_sv(arr_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes HSV Saturation S [0..1] and Value V [0..1] from float32 RGB array."""
    max_c = np.max(arr_rgb, axis=2)
    min_c = np.min(arr_rgb, axis=2)
    hsv_v = max_c / 255.0
    hsv_s = np.where(max_c > 1e-3, (max_c - min_c) / np.maximum(max_c, 1e-3), 0.0)
    return hsv_s.astype(np.float32), hsv_v.astype(np.float32)


def _compute_edt_2d(fg_mask: np.ndarray) -> np.ndarray:
    """Computes exact Euclidean Distance Transform on boolean foreground mask."""
    if HAS_OPENCV:
        return cv2.distanceTransform(fg_mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)

    h, w = fg_mask.shape
    inf = float(h + w)
    g = np.where(fg_mask, inf, 0.0).astype(np.float32)
    for y in range(1, h):
        g[y] = np.where(fg_mask[y], np.minimum(g[y], g[y - 1] + 1.0), 0.0)
    for y in range(h - 2, -1, -1):
        g[y] = np.where(fg_mask[y], np.minimum(g[y], g[y + 1] + 1.0), 0.0)

    dt = np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    for y in range(h):
        if not np.any(fg_mask[y]):
            continue
        gy = g[y]
        for x in range(w):
            if fg_mask[y, x]:
                d2 = (xs - x) ** 2 + gy ** 2
                dt[y, x] = math.sqrt(float(np.min(d2)))
    return dt


def _gate1_and_gate2_vegetation_mask(
    arr_rgb: np.ndarray,
    lum_arr: np.ndarray,
    hsv_s: np.ndarray,
    hsv_v: np.ndarray,
    lap_mag: np.ndarray,
    strict_mask: np.ndarray,
    px_per_m: float,
) -> Tuple[np.ndarray, int, int, float]:
    """
    Gate 1: Spectral Vegetation + Non-Building/Non-Water Mask (ExG + HSV S/V).
    Gate 2: Connected-Component Crop Sheet Rejection (>85 m^2 lacking >=35% internal shadow valleys).
    """
    h, w = lum_arr.shape
    mpp = 1.0 / max(px_per_m, 0.5)
    px_area_m2 = mpp * mpp

    r, g, b = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]
    exg = (2.0 * g - r - b) / (r + g + b + 1e-6)

    veg_mask = (
        strict_mask
        & (exg > 0.045)
        & (g > r * 0.94)
        & (hsv_s >= 0.14)
        & (hsv_v >= 0.16)
        & (hsv_v <= 0.88)
        & (b <= g * 1.05)
    )

    comps_total = 0
    comps_kept = 0
    if HAS_OPENCV:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(veg_mask.astype(np.uint8), connectivity=4)
        for lbl in range(1, num_labels):
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < 10:
                veg_mask[labels == lbl] = False
                continue
            comps_total += 1
            comp_area_m2 = area_px * px_area_m2
            if comp_area_m2 > CROP_SHEET_AREA_M2:
                bx = int(stats[lbl, cv2.CC_STAT_LEFT])
                by = int(stats[lbl, cv2.CC_STAT_TOP])
                bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                comp_pixels = labels == lbl
                mean_lum = float(np.mean(lum_arr[comp_pixels]))
                shadow_cutoff = mean_lum * (1.0 - MIN_CAST_SHADOW_DROP)

                box_strict = strict_mask[by:by + bh, bx:bx + bw]
                box_lum = lum_arr[by:by + bh, bx:bx + bw]
                box_v = hsv_v[by:by + bh, bx:bx + bw]
                box_lap = lap_mag[by:by + bh, bx:bx + bw]
                valid_cnt = max(1, int(np.sum(box_strict)))
                shadow_cnt = int(np.sum(box_strict & (box_lum <= shadow_cutoff) & (box_v < 0.25)))
                shadow_ratio = shadow_cnt / float(valid_cnt)
                lap_var = float(np.var(box_lap[box_strict]))

                if shadow_ratio < 0.025 or lap_var < 36.0:
                    veg_mask[comp_pixels] = False
                    continue
            comps_kept += 1
    else:
        visited = np.zeros((h, w), dtype=bool)
        ys_fg, xs_fg = np.where(veg_mask)
        for sy, sx in zip(ys_fg.tolist(), xs_fg.tolist()):
            if visited[sy, sx]:
                continue
            q = deque([(sx, sy)])
            visited[sy, sx] = True
            pts: List[Tuple[int, int]] = []
            min_x, max_x, min_y, max_y = sx, sx, sy, sy
            while q:
                px, py = q.popleft()
                pts.append((px, py))
                min_x, max_x = min(min_x, px), max(max_x, px)
                min_y, max_y = min(min_y, py), max(max_y, py)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h and veg_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
            if len(pts) < 10:
                for px, py in pts:
                    veg_mask[py, px] = False
                continue
            comps_total += 1
            if len(pts) * px_area_m2 > CROP_SHEET_AREA_M2:
                comp_lums = [lum_arr[py, px] for px, py in pts]
                shadow_cutoff = float(np.mean(comp_lums)) * (1.0 - MIN_CAST_SHADOW_DROP)
                box_strict = strict_mask[min_y:max_y + 1, min_x:max_x + 1]
                box_lum = lum_arr[min_y:max_y + 1, min_x:max_x + 1]
                box_v = hsv_v[min_y:max_y + 1, min_x:max_x + 1]
                box_lap = lap_mag[min_y:max_y + 1, min_x:max_x + 1]
                valid_cnt = max(1, int(np.sum(box_strict)))
                shadow_cnt = int(np.sum(box_strict & (box_lum <= shadow_cutoff) & (box_v < 0.25)))
                lap_var = float(np.var(box_lap[box_strict]))
                if shadow_cnt / float(valid_cnt) < 0.025 or lap_var < 36.0:
                    for px, py in pts:
                        veg_mask[py, px] = False
                    continue
            comps_kept += 1

    total_strict = int(np.sum(strict_mask))
    active_pct = round(float(np.sum(veg_mask) / max(1, total_strict) * 100.0), 1)
    return veg_mask, comps_kept, comps_total, active_pct


def _gate3_watershed_crown_candidates(
    veg_mask: np.ndarray,
    px_per_m: float,
    tree_type_str: str,
    corridor_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Gate 3: Euclidean Distance Transform + Local Maxima (min spacing 2.5m) + Watershed Crown Separation.
    Filters basins to diameter 1.8m..7.0m, circularity >= 0.68, aspect_ratio <= 2.2.
    """
    if not np.any(veg_mask):
        return []

    h, w = veg_mask.shape
    mpp = 1.0 / max(px_per_m, 0.5)
    dt = _compute_edt_2d(veg_mask)

    min_peak_dt = max(1.8, 0.72 / max(mpp, 0.12))
    min_spacing_px = max(4.0, MIN_CROWN_SPACING_M / max(mpp, 0.12))

    # Local 3x3 maxima above min_peak_dt
    padded_dt = np.pad(dt, 1, mode="constant", constant_values=0.0)
    is_local_max = veg_mask & (dt >= min_peak_dt)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            is_local_max &= dt >= padded_dt[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]

    py_idx, px_idx = np.where(is_local_max)
    raw_peaks = sorted(
        zip(px_idx.tolist(), py_idx.tolist(), dt[py_idx, px_idx].tolist()),
        key=lambda t: t[2],
        reverse=True,
    )

    seeds: List[Tuple[int, int, int]] = []
    for px, py, _ in raw_peaks:
        if all(math.hypot(px - sx, py - sy) >= min_spacing_px for sx, sy, _ in seeds):
            seeds.append((px, py, len(seeds) + 1))
            if len(seeds) >= 1200:
                break

    if not seeds:
        return []

    # Descending distance watershed basin assignment
    basin_labels = np.zeros((h, w), dtype=np.int32)
    for sx, sy, sid in seeds:
        basin_labels[sy, sx] = sid

    fg_y, fg_x = np.where(veg_mask)
    fg_dt = dt[fg_y, fg_x]
    order = np.argsort(-fg_dt)
    sorted_yx = list(zip(fg_y[order].tolist(), fg_x[order].tolist()))

    for _ in range(3):
        for y, x in sorted_yx:
            if basin_labels[y, x] != 0:
                continue
            best_lbl = 0
            best_d = -1.0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nl = int(basin_labels[ny, nx])
                    if nl > 0 and float(dt[ny, nx]) > best_d:
                        best_d = float(dt[ny, nx])
                        best_lbl = nl
            if best_lbl > 0:
                basin_labels[y, x] = best_lbl

    t_low = (tree_type_str or "").lower()
    is_palm = any(k in t_low for k in ["نخيل", "نخل", "palm", "date", "بلح"])
    is_windbreak = any(k in t_low for k in ["جازورين", "كازورينا", "كافور", "سرو", "casuarina", "eucalyptus", "مصدات"])

    cands: List[Dict[str, Any]] = []
    perim_lut = [0.0, 0.7071, 0.7071, 1.0, 0.7071, 1.0, 1.4142, 0.7071,
                 0.7071, 1.4142, 1.0, 0.7071, 1.0, 0.7071, 0.7071, 0.0]

    for _, _, sid in seeds:
        by_arr, bx_arr = np.where(basin_labels == sid)
        area_px = int(bx_arr.size)
        if area_px < 10:
            continue

        eq_diam_m = 2.0 * math.sqrt(area_px / math.pi) * mpp
        if not (MIN_CROWN_DIAM_M <= eq_diam_m <= MAX_CROWN_DIAM_M):
            continue

        min_x, max_x = int(np.min(bx_arr)), int(np.max(bx_arr))
        min_y, max_y = int(np.min(by_arr)), int(np.max(by_arr))
        bw = max_x - min_x + 3
        bh = max_y - min_y + 3
        local_bin = np.zeros((bh, bw), dtype=np.uint8)
        local_bin[by_arr - min_y + 1, bx_arr - min_x + 1] = 255

        if HAS_OPENCV:
            contours, _ = cv2.findContours(local_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            perim = float(cv2.arcLength(cnt, True))
        else:
            b0 = (local_bin > 0).astype(np.uint8)
            perim = 0.0
            for ly in range(bh - 1):
                for lx in range(bw - 1):
                    st = (int(b0[ly, lx]) << 3) | (int(b0[ly, lx + 1]) << 2) | (int(b0[ly + 1, lx + 1]) << 1) | int(b0[ly + 1, lx])
                    perim += perim_lut[st]

        if perim <= 1.0:
            continue
        circularity = (4.0 * math.pi * float(area_px)) / (perim * perim)
        if circularity < MIN_BLOB_CIRCULARITY:
            continue

        ax = bx_arr.astype(np.float32)
        ay = by_arr.astype(np.float32)
        cxx = float(np.var(ax))
        cyy = float(np.var(ay))
        cxy = float(np.mean((ax - np.mean(ax)) * (ay - np.mean(ay))))
        trace = cxx + cyy
        det = cxx * cyy - cxy * cxy
        disc = math.sqrt(max(0.0, (trace * trace) / 4.0 - det))
        lam1 = trace / 2.0 + disc
        lam2 = max(0.15, trace / 2.0 - disc)
        aspect_ratio = math.sqrt(lam1 / lam2)
        if aspect_ratio > 2.2:
            continue

        cx = int(round(float(np.mean(ax))))
        cy = int(round(float(np.mean(ay))))
        if not (2 <= cx < w - 2 and 2 <= cy < h - 2):
            continue

        box_r = max(4, int(round((eq_diam_m / 2.0) / max(mpp, 0.12))))
        on_bound = bool(corridor_mask[cy, cx])
        if is_palm or (not on_bound and eq_diam_m >= 5.2):
            cls_id, cls_name = 1, "palm_tree"
        elif on_bound or is_windbreak:
            cls_id, cls_name = 2, "windbreak_tree"
        else:
            cls_id, cls_name = 0, "orchard_tree"

        cands.append({
            "bbox": (max(1, min_x), max(1, min_y), min(w - 2, max_x), min(h - 2, max_y)),
            "center": (cx, cy),
            "conf": min(0.95, 0.52 + circularity * 0.30),
            "crown_radius": box_r,
            "class_id": cls_id,
            "class_name": cls_name,
            "circularity": round(circularity, 2),
            "crown_diameter_m": round(eq_diam_m, 2),
        })

    return cands


def _verify_cast_shadow_and_telemetry(
    candidates: List[Dict[str, Any]],
    arr_rgb: np.ndarray,
    lum_arr: np.ndarray,
    hsv_v: np.ndarray,
    lap_mag: np.ndarray,
    strict_mask: np.ndarray,
    stage1_mask: np.ndarray,
    px_per_m: float,
    require_strict_cast_shadow: bool = True,
    bypass_heuristic_rejection: bool = False,
) -> List[Dict[str, Any]]:
    """
    Computes crown shadow-drop & Laplacian texture telemetry.
    When bypass_heuristic_rejection=True (AI/YOLO path), never drops AI detections via shadow/ExG heuristics;
    strict filtering is reserved exclusively for the 4-Gate morphological fallback.
    """
    h, w = lum_arr.shape
    mpp = 1.0 / max(px_per_m, 0.5)
    r_ch, g_ch, b_ch = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]

    verified: List[Dict[str, Any]] = []
    for cand in candidates:
        cx, cy = cand["center"]
        if not (2 <= cx < w - 2 and 2 <= cy < h - 2):
            continue
        if not strict_mask[cy, cx]:
            continue
        if not bypass_heuristic_rejection and not stage1_mask[cy, cx]:
            continue

        bx1, by1, bx2, by2 = cand["bbox"]
        x1, y1 = max(1, bx1), max(1, by1)
        x2, y2 = min(w - 2, bx2), min(h - 2, by2)
        if x2 <= x1 or y2 <= y1:
            continue

        radius_px = max(3.5, float(cand.get("crown_radius", (x2 - x1) / 2.0)))
        diam_m = float(cand.get("crown_diameter_m") or round(radius_px * 2.0 * mpp, 2))

        c_rad = max(2, int(round(radius_px * 0.45)))
        cy0, cy1 = max(0, cy - c_rad), min(h, cy + c_rad + 1)
        cx0, cx1 = max(0, cx - c_rad), min(w, cx + c_rad + 1)
        core_lum = float(np.mean(lum_arr[cy0:cy1, cx0:cx1]))
        if not bypass_heuristic_rejection and core_lum < 42.0:
            continue

        cr_p = r_ch[cy0:cy1, cx0:cx1]
        cg_p = g_ch[cy0:cy1, cx0:cx1]
        cb_p = b_ch[cy0:cy1, cx0:cx1]
        core_exg = float(np.mean((2.0 * cg_p - cr_p - cb_p) / (cr_p + cg_p + cb_p + 1e-6)))
        if not bypass_heuristic_rejection and core_exg < 0.01 and float(np.mean(cg_p >= cr_p * 0.92)) < 0.5:
            continue

        lap_var = float(np.var(lap_mag[y1:y2 + 1, x1:x2 + 1]))

        # Scan 8 directional angular wedges adjacent to crown edge (0.85*R .. 2.65*R)
        r_in = max(2.0, radius_px * 0.85)
        r_out = min(34.0, max(r_in + 3.0, radius_px * 2.65))
        i_rout = int(math.ceil(r_out))

        sy0, sy1 = max(0, cy - i_rout), min(h, cy + i_rout + 1)
        sx0, sx1 = max(0, cx - i_rout), min(w, cx + i_rout + 1)
        yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
        dx = xx - cx
        dy = yy - cy
        dist = np.hypot(dx, dy)
        ring_sel = (dist >= r_in) & (dist <= r_out) & strict_mask[sy0:sy1, sx0:sx1]
        if not np.any(ring_sel):
            continue

        angles = np.mod(np.arctan2(dy[ring_sel], dx[ring_sel]), 2.0 * math.pi)
        sectors = (np.floor(np.mod(angles + math.pi / 8.0, 2.0 * math.pi) / (math.pi / 4.0))).astype(np.int32)
        ring_lums = lum_arr[sy0:sy1, sx0:sx1][ring_sel]
        ring_vs = hsv_v[sy0:sy1, sx0:sx1][ring_sel]

        best_sector = -1
        best_shadow_lum = core_lum
        best_shadow_v = 1.0
        sector_medians = np.full(8, core_lum, dtype=np.float32)

        for s in range(8):
            s_mask = sectors == s
            cnt_s = int(np.sum(s_mask))
            if cnt_s < 4:
                continue
            s_l = np.sort(ring_lums[s_mask])
            s_v = np.sort(ring_vs[s_mask])
            sector_medians[s] = float(s_l[cnt_s // 2])
            k_dark = max(3, int(cnt_s * 0.25))
            pocket_l = float(np.mean(s_l[:k_dark]))
            pocket_v = float(np.mean(s_v[:k_dark]))
            if pocket_l < best_shadow_lum:
                best_shadow_lum = pocket_l
                best_shadow_v = pocket_v
                best_sector = s

        rel_shadow_drop = (core_lum - best_shadow_lum) / core_lum if best_sector >= 0 else 0.0
        opp_lum = float(sector_medians[(best_sector + 4) % 8]) if best_sector >= 0 else best_shadow_lum
        is_directional = opp_lum >= best_shadow_lum * 1.28

        has_paired_shadow = (
            best_sector >= 0
            and rel_shadow_drop >= MIN_CAST_SHADOW_DROP
            and best_shadow_v <= 0.28
            and is_directional
        )

        if not bypass_heuristic_rejection:
            if require_strict_cast_shadow and not has_paired_shadow:
                continue
            if not require_strict_cast_shadow and rel_shadow_drop < 0.22 and lap_var < 28.0:
                continue

        out_cand = dict(cand)
        out_cand["shadow_drop"] = round(rel_shadow_drop, 3)
        out_cand["laplacian_var"] = round(lap_var, 1)
        out_cand["crown_diameter_m"] = round(diam_m, 2)
        verified.append(out_cand)

    return verified


def detect_and_classify_trees(
    satellite_image: Image.Image,
    parcel_poly_px: Optional[List[Tuple[int, int]]] = None,
    pixel_spacing_px: float = 12.0,
    yolo_model_path: Optional[str] = None,
    healthy_ndvi_threshold: float = 0.16,
    stressed_ndvi_threshold: float = 0.07,
    detection_mode: str = "high_precision",
    tree_type_str: str = "",
    sensitivity: float = 0.50,
    generate_heatmap: bool = False,
    tile_overlap: float = SAHI_OVERLAP,
    px_per_m: float = 3.8,
    slice_size: int = SAHI_TILE_SIZE,
    inward_buffer_m: float = 0.0,
    show_compact_ids: bool = False,
) -> Dict[str, Any]:
    """
    Two-Stage Structural Tree Crown Detection Pipeline:
    - Stage 1: Flat-Crop Texture & Shadow Rejection (4m x 4m pre-filter grid mask).
    - Stage 2:
        * Primary AI Path: YOLOv8 SAHI 512x512 (20% overlap, NMS IoU=0.45, conf >= 0.55).
        * Algorithmic Fallback: Multi-Scale DoG blobs (2.0..6.5m, circularity > 0.65)
          with paired adjacent cast shadows (>= 35% luminance drop).
    """
    orig_img = satellite_image.convert("RGB")
    width_px, height_px = orig_img.size

    ndvi_matrix = compute_ndvi_or_vari(orig_img)
    arr = np.array(orig_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum_arr = 0.299 * r + 0.587 * g + 0.114 * b
    hsv_s, hsv_v = _rgb_to_hsv_sv(arr)

    if HAS_OPENCV:
        lap_mag = np.abs(cv2.Laplacian(lum_arr.astype(np.float32), cv2.CV_32F, ksize=3))
    else:
        padded = np.pad(lum_arr.astype(np.float32), 1, mode="edge")
        lap_mag = np.abs(
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * lum_arr
        )

    # Strict GeoJSON Polygon Mask + optional inward buffer
    buffer_3m_px = max(3, int(round(3.0 * max(px_per_m, 1.0))))
    parcel_mask_img = Image.new("L", (width_px, height_px), 0)
    draw_mask = ImageDraw.Draw(parcel_mask_img)
    if parcel_poly_px and len(parcel_poly_px) >= 3:
        draw_mask.polygon(parcel_poly_px, outline=255, fill=255)
    else:
        draw_mask.rectangle([0, 0, width_px, height_px], fill=255)

    parcel_mask = np.array(parcel_mask_img) > 0
    if inward_buffer_m < -0.05 and parcel_poly_px and len(parcel_poly_px) >= 3:
        erode_px = max(1, int(round(abs(inward_buffer_m) * max(px_per_m, 1.0))))
        if HAS_OPENCV:
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
            parcel_mask = cv2.erode(parcel_mask.astype(np.uint8) * 255, k_erode) > 0
        else:
            parcel_mask = np.array(parcel_mask_img.filter(ImageFilter.MinFilter(erode_px * 2 + 1))) > 0

    total_parcel_pixels = int(np.sum(parcel_mask))

    corridor_mask = np.zeros((height_px, width_px), dtype=bool)
    boundary_buf_mask = parcel_mask.copy()
    if parcel_poly_px and len(parcel_poly_px) >= 3 and inward_buffer_m >= -0.05:
        buf_img = Image.new("L", (width_px, height_px), 0)
        draw_buf = ImageDraw.Draw(buf_img)
        draw_buf.polygon(parcel_poly_px, outline=255, fill=255)
        draw_buf.line(parcel_poly_px + [parcel_poly_px[0]], fill=255, width=max(4, buffer_3m_px * 2))
        boundary_buf_mask = np.array(buf_img) > 0

        corridor_px = max(float(buffer_3m_px), pixel_spacing_px * 1.10)
        corr_img = Image.new("L", (width_px, height_px), 0)
        draw_corr = ImageDraw.Draw(corr_img)
        draw_corr.line(parcel_poly_px + [parcel_poly_px[0]], fill=255, width=int(corridor_px * 2.0))
        corridor_mask = (np.array(corr_img) > 0) & boundary_buf_mask
    elif parcel_poly_px and len(parcel_poly_px) >= 3:
        corridor_px = max(float(buffer_3m_px), pixel_spacing_px * 1.10)
        corr_img = Image.new("L", (width_px, height_px), 0)
        draw_corr = ImageDraw.Draw(corr_img)
        draw_corr.line(parcel_poly_px + [parcel_poly_px[0]], fill=255, width=int(corridor_px * 2.0))
        corridor_mask = (np.array(corr_img) > 0) & parcel_mask

    target_mask = corridor_mask if (detection_mode == "boundary_only" and np.any(corridor_mask)) else boundary_buf_mask

    # Gates 1 & 2: Spectral Vegetation + Connected-Component Crop Sheet (>85 m^2) Rejection
    stage1_mask, cells_kept, cells_total, stage1_active_pct = _gate1_and_gate2_vegetation_mask(
        arr, lum_arr, hsv_s, hsv_v, lap_mag, target_mask, px_per_m
    )

    if parcel_poly_px:
        min_x = max(0, min(p[0] for p in parcel_poly_px) - buffer_3m_px)
        max_x = min(width_px - 1, max(p[0] for p in parcel_poly_px) + buffer_3m_px)
        min_y = max(0, min(p[1] for p in parcel_poly_px) - buffer_3m_px)
        max_y = min(height_px - 1, max(p[1] for p in parcel_poly_px) + buffer_3m_px)
    else:
        min_x, max_x = 0, width_px - 1
        min_y, max_y = 0, height_px - 1

    sahi_slices = _generate_sahi_slices(
        min_x, min_y, max_x, max_y, width_px, height_px,
        slice_size=slice_size, overlap=tile_overlap,
    )
    min_sep_px = max(6.0, 2.0 * px_per_m)

    # Pass RAW unmasked RGB tiles to YOLO (matches training domain); mask strictly post-inference
    raw_rgb_u8 = arr.astype(np.uint8)

    detected_boxes: List[Dict[str, Any]] = []
    used_yolo = False
    stage1_audit_dropped_count = 0
    active_yolo_path = yolo_model_path
    if not active_yolo_path:
        for cand in ["best.pt", "models/best.pt", "/mount/src/farm-qa/best.pt", "/mount/src/farm-qa/models/best.pt"]:
            if os.path.exists(cand):
                active_yolo_path = cand
                break

    yolo_skipped_reason = ""
    raw_yolo_cands: List[Dict[str, Any]] = []
    if active_yolo_path and np.any(target_mask):
        yolo_model, skip_reason = _get_validated_tree_yolo(active_yolo_path)
        if yolo_model is not None:
            try:
                for sx1, sy1, sx2, sy2 in sahi_slices:
                    tile_mask = target_mask[sy1:sy2, sx1:sx2]
                    if tile_mask.size == 0 or float(np.mean(tile_mask)) < 0.05:
                        continue
                    crop_arr = raw_rgb_u8[sy1:sy2, sx1:sx2]
                    yolo_res = yolo_model.predict(
                        source=crop_arr,
                        conf=AI_MODEL_CONF_THRESH,
                        iou=NMS_IOU_THRESH,
                        verbose=False,
                    )
                    if yolo_res and len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                        for b_box in yolo_res[0].boxes:
                            conf_val = float(b_box.conf[0].cpu().numpy())
                            if conf_val < AI_MODEL_CONF_THRESH:
                                continue
                            xyxy = b_box.xyxy[0].cpu().numpy()
                            bx1 = sx1 + int(xyxy[0])
                            by1 = sy1 + int(xyxy[1])
                            bx2 = sx1 + int(xyxy[2])
                            by2 = sy1 + int(xyxy[3])
                            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                            if not (0 <= cy < height_px and 0 <= cx < width_px and target_mask[cy, cx]):
                                continue
                            cls_id = int(b_box.cls[0].cpu().numpy()) if b_box.cls is not None else 0
                            cls_name = TREE_CLASS_NAMES.get(cls_id, "orchard_tree")
                            crown_rad = max(4, int(max(bx2 - bx1, by2 - by1) // 2))
                            raw_yolo_cands.append({
                                "bbox": (bx1, by1, bx2, by2),
                                "center": (cx, cy),
                                "conf": conf_val,
                                "crown_radius": crown_rad,
                                "class_id": cls_id,
                                "class_name": cls_name,
                            })
                used_yolo = True
            except Exception:
                used_yolo = False
                raw_yolo_cands = []
        elif skip_reason:
            yolo_skipped_reason = skip_reason

    if used_yolo:
        fused = _apply_sahi_wbf_nms(
            raw_yolo_cands,
            iou_thresh=NMS_IOU_THRESH,
            min_dist_px=min_sep_px,
            class_aware=CLASS_AWARE_NMS,
            cross_class_iou=CROSS_CLASS_SUPPRESSION_IOU,
        )
        # Non-blocking Stage-1 diagnostic audit (log mismatches without dropping AI boxes)
        for cand in fused:
            cx, cy = cand["center"]
            if 0 <= cy < height_px and 0 <= cx < width_px and not stage1_mask[cy, cx]:
                stage1_audit_dropped_count += 1

        detected_boxes = _verify_cast_shadow_and_telemetry(
            fused, arr, lum_arr, hsv_v, lap_mag, target_mask, target_mask,
            px_per_m, require_strict_cast_shadow=False, bypass_heuristic_rejection=True,
        )
    elif np.any(stage1_mask):
        # Gate 3: EDT + Watershed Crown Separation
        raw_ws = _gate3_watershed_crown_candidates(
            stage1_mask, px_per_m, tree_type_str, corridor_mask
        )
        fused = _apply_sahi_wbf_nms(
            raw_ws,
            iou_thresh=NMS_IOU_THRESH,
            min_dist_px=min_sep_px,
            class_aware=CLASS_AWARE_NMS,
            cross_class_iou=CROSS_CLASS_SUPPRESSION_IOU,
        )
        # Gate 4: Paired Directional Cast-Shadow Verification (>=35% drop)
        detected_boxes = _verify_cast_shadow_and_telemetry(
            fused, arr, lum_arr, hsv_v, lap_mag, target_mask, stage1_mask,
            px_per_m, require_strict_cast_shadow=True, bypass_heuristic_rejection=False,
        )

    del arr, raw_rgb_u8, lum_arr, hsv_s, hsv_v, lap_mag, r, g, b

    # Health Classification on verified detections only
    trees_classified: List[Dict[str, Any]] = []
    healthy_count = 0
    stressed_count = 0
    damaged_count = 0

    for idx, item in enumerate(detected_boxes):
        bx1, by1, bx2, by2 = item["bbox"]
        cx, cy = item["center"]
        crown_rad = item.get("crown_radius", max(4, int((bx2 - bx1) // 2)))
        cx1 = max(0, min(width_px - 1, bx1))
        cy1 = max(0, min(height_px - 1, by1))
        cx2 = max(0, min(width_px - 1, bx2))
        cy2 = max(0, min(height_px - 1, by2))

        crown_ndvi_crop = ndvi_matrix[cy1:cy2 + 1, cx1:cx2 + 1]
        if crown_ndvi_crop.size > 0:
            mean_ndvi = float(np.mean(crown_ndvi_crop))
            max_ndvi = float(np.max(crown_ndvi_crop))
        else:
            mean_ndvi = float(ndvi_matrix[cy, cx])
            max_ndvi = mean_ndvi

        if mean_ndvi >= healthy_ndvi_threshold or max_ndvi >= (healthy_ndvi_threshold + 0.05):
            status = "healthy"
            status_ar = "سليمة / حيوية عالية"
            color_rgb = (34, 197, 94)
            healthy_count += 1
        elif mean_ndvi >= stressed_ndvi_threshold:
            status = "stressed"
            status_ar = "مجهدة / حيوية متوسطة"
            color_rgb = (234, 179, 8)
            stressed_count += 1
        else:
            status = "damaged"
            status_ar = "ميتة أو تالفة / جافة"
            color_rgb = (239, 68, 68)
            damaged_count += 1

        trees_classified.append({
            "tree_id": idx + 1,
            "center": (cx, cy),
            "bbox": (cx1, cy1, cx2, cy2),
            "crown_radius": crown_rad,
            "class_id": int(item.get("class_id", 0)),
            "class_name": str(item.get("class_name", "orchard_tree")),
            "conf": float(item.get("conf", 0.55)),
            "shadow_drop": float(item.get("shadow_drop", 0.0)),
            "laplacian_var": float(item.get("laplacian_var", 0.0)),
            "circularity": item.get("circularity"),
            "crown_diameter_m": item.get("crown_diameter_m"),
            "mean_pseudo_ndvi": round(mean_ndvi, 3),
            "max_pseudo_ndvi": round(max_ndvi, 3),
            "status": status,
            "status_ar": status_ar,
            "color_rgb": color_rgb,
        })

    total_trees = len(trees_classified)
    healthy_pct = round((healthy_count / max(1, total_trees)) * 100.0, 1) if total_trees > 0 else 0.0
    stressed_pct = round((stressed_count / max(1, total_trees)) * 100.0, 1) if total_trees > 0 else 0.0
    damaged_pct = round((damaged_count / max(1, total_trees)) * 100.0, 1) if total_trees > 0 else 0.0

    # Render Clean Canopy Overlay + Dual-Font Reshaped BiDi Arabic Banner (Zero [▯] glyph boxes)
    annotated_img = orig_img.copy()
    draw_ann = ImageDraw.Draw(annotated_img)

    if parcel_poly_px and len(parcel_poly_px) >= 3:
        draw_ann.line(parcel_poly_px + [parcel_poly_px[0]], fill=(0, 255, 200), width=3)

    for t in trees_classified:
        cx, cy = t["center"]
        col = t["color_rgb"]
        cr = t.get("crown_radius", 5)

        draw_ann.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=col, width=2)
        draw_ann.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 0, 0))
        draw_ann.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)

        if show_compact_ids:
            _draw_mixed_bidi_text(draw_ann, (cx + cr + 2, cy - 6), str(t["tree_id"]), fill=(255, 255, 255), size=11)

    banner_h = 38
    draw_ann.rectangle([0, 0, width_px, banner_h], fill=(15, 23, 42))
    header_txt = (
        f"إجمالي الأشجار: {total_trees} | سليمة: {healthy_count} ({healthy_pct}%) | "
        f"مجهدة: {stressed_count} ({stressed_pct}%) | تالفة: {damaged_count} ({damaged_pct}%)"
    )
    _draw_mixed_bidi_text(draw_ann, (12, 9), header_txt, fill=(0, 255, 200), size=14)

    pseudo_ndvi_heatmap = generate_ndvi_colormap_image(ndvi_matrix, mask=parcel_mask) if generate_heatmap else None

    veg_pixels = (ndvi_matrix > 0.12) & parcel_mask
    parcel_veg_coverage_pct = round(float(np.sum(veg_pixels) / max(1, total_parcel_pixels) * 100.0), 1)

    return {
        "total_trees": total_trees,
        "healthy_count": healthy_count,
        "healthy_pct": healthy_pct,
        "stressed_count": stressed_count,
        "stressed_pct": stressed_pct,
        "damaged_count": damaged_count,
        "damaged_pct": damaged_pct,
        "parcel_veg_coverage_pct": parcel_veg_coverage_pct,
        "stage1_active_pct": stage1_active_pct,
        "stage1_cells_kept": cells_kept,
        "stage1_cells_total": cells_total,
        "stage1_audit_dropped_count": stage1_audit_dropped_count,
        "annotated_image": annotated_img,
        "pseudo_ndvi_heatmap_image": pseudo_ndvi_heatmap,
        "engine_used": (
            f"YOLOv8 SAHI {slice_size}px ({os.path.basename(active_yolo_path)}, Conf>={AI_MODEL_CONF_THRESH:.2f})"
            if used_yolo
            else "4-Gate Canopy Watershed Fallback (ExG+CropSheet+EDT+Shadow)"
        ),
        "yolo_model_loaded": used_yolo,
        "yolo_model_path": active_yolo_path if used_yolo else None,
        "yolo_skipped_reason": yolo_skipped_reason,
        "trees_classified": trees_classified,
    }
