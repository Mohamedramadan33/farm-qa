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

AI_MODEL_CONF_THRESH = 0.55
SAHI_TILE_SIZE = 512
SAHI_OVERLAP = 0.20
NMS_IOU_THRESH = 0.45
MIN_CAST_SHADOW_DROP = 0.35
MIN_BLOB_CIRCULARITY = 0.65
MIN_CROWN_DIAM_M = 2.0
MAX_CROWN_DIAM_M = 6.5

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

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
            if dist < min_dist_px or iou >= iou_thresh:
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
# Stage 1: Flat-Crop Texture & Shadow Rejection (4m x 4m Pre-Filter Grid Mask)
# ---------------------------------------------------------------------------

def _build_stage1_flat_crop_mask(
    arr_rgb: np.ndarray,
    lum_arr: np.ndarray,
    hsv_v: np.ndarray,
    lap_mag: np.ndarray,
    strict_mask: np.ndarray,
    px_per_m: float,
) -> Tuple[np.ndarray, int, int, float]:
    """
    Divides the masked polygon into 4m x 4m physical cells.
    Zeros out smooth flat crop basins (berseem, wheat, vegetables) that lack BOTH:
      1) Local Texture Roughness (Laplacian variance + luminance contrast spread)
      2) Dark Shadow Pixel Ratio (HSV V < 0.22 adjacent to brighter vegetation)
    """
    h, w = lum_arr.shape
    mpp = 1.0 / max(px_per_m, 0.5)

    # Local max HSV Value within ~2.2m neighborhood to identify shadow pockets adjacent to brighter crowns
    nb_rad = max(4, min(11, int(round(2.2 / max(mpp, 0.12)))))
    k_nb = nb_rad * 2 + 1
    if HAS_OPENCV:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_nb, k_nb))
        v_masked = np.where(strict_mask, hsv_v, 0.0).astype(np.float32)
        local_max_v = cv2.dilate(v_masked, kernel)
    else:
        v_u8 = np.where(strict_mask, np.clip(hsv_v * 255.0, 0, 255), 0).astype(np.uint8)
        local_max_v = np.array(Image.fromarray(v_u8, mode="L").filter(ImageFilter.MaxFilter(k_nb)), dtype=np.float32) / 255.0

    shadow_pocket = strict_mask & (hsv_v < 0.22) & (local_max_v >= 0.30) & ((local_max_v - hsv_v) >= 0.11)

    cell_px = max(10, int(round(4.0 / max(mpp, 0.12))))
    halo_px = max(3, int(round(1.5 / max(mpp, 0.12))))

    active_mask = np.zeros((h, w), dtype=bool)
    cells_total = 0
    cells_kept = 0

    for cy0 in range(0, h, cell_px):
        cy1 = min(h, cy0 + cell_px)
        hy0 = max(0, cy0 - halo_px)
        hy1 = min(h, cy1 + halo_px)
        for cx0 in range(0, w, cell_px):
            cx1 = min(w, cx0 + cell_px)
            cell_mask = strict_mask[cy0:cy1, cx0:cx1]
            valid_cnt = int(np.sum(cell_mask))
            if valid_cnt < 12:
                continue
            cells_total += 1

            cell_lap = lap_mag[cy0:cy1, cx0:cx1][cell_mask]
            lap_var = float(np.var(cell_lap))

            cell_lum = lum_arr[cy0:cy1, cx0:cx1][cell_mask]
            p5, p95 = np.percentile(cell_lum, [5, 95])
            contrast_range = float(p95 - p5)

            hx0 = max(0, cx0 - halo_px)
            hx1 = min(w, cx1 + halo_px)
            shadow_cnt = int(np.sum(shadow_pocket[hy0:hy1, hx0:hx1]))
            shadow_ratio = shadow_cnt / float(valid_cnt)

            if lap_var >= 35.0 and contrast_range >= 25.0 and shadow_ratio >= 0.012 and shadow_cnt >= 3:
                cells_kept += 1
                active_mask[cy0:cy1, cx0:cx1] = cell_mask

    total_strict = int(np.sum(strict_mask))
    active_pct = round(float(np.sum(active_mask) / max(1, total_strict) * 100.0), 1)
    return active_mask, cells_kept, cells_total, active_pct


# ---------------------------------------------------------------------------
# Stage 2: Structural Multi-Scale DoG + Circularity (>0.65) + Paired Cast Shadow (>=35%)
# ---------------------------------------------------------------------------

def _measure_blob_shape(
    dog_map: np.ndarray,
    stage1_mask: np.ndarray,
    cx: int,
    cy: int,
    peak_val: float,
    max_radius_px: float,
) -> Optional[Tuple[int, float, float, Tuple[int, int, int, int]]]:
    h, w = dog_map.shape
    rad = max(5, min(20, int(math.ceil(max_radius_px))))
    x0 = max(1, cx - rad)
    y0 = max(1, cy - rad)
    x1 = min(w - 2, cx + rad)
    y1 = min(h - 2, cy + rad)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 3 or bh < 3:
        return None

    thresh = peak_val * 0.42
    blob_bin = np.zeros((bh, bw), dtype=np.uint8)
    queue = deque([(cx, cy)])
    blob_bin[cy - y0, cx - x0] = 255

    pts_x: List[int] = []
    pts_y: List[int] = []
    rad2 = rad * rad

    while queue:
        px, py = queue.popleft()
        pts_x.append(px)
        pts_y.append(py)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy
            if nx < x0 or nx > x1 or ny < y0 or ny > y1:
                continue
            if blob_bin[ny - y0, nx - x0]:
                continue
            if (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy) > rad2:
                continue
            if not stage1_mask[ny, nx] or dog_map[ny, nx] < thresh:
                continue
            blob_bin[ny - y0, nx - x0] = 255
            queue.append((nx, ny))

    area_px = len(pts_x)
    if area_px < 10:
        return None

    # Exact perimeter & circularity
    if HAS_OPENCV:
        contours, _ = cv2.findContours(blob_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        perim = float(cv2.arcLength(cnt, True))
        cv_area = float(cv2.contourArea(cnt))
        eff_area = max(float(area_px), cv_area)
    else:
        # 2x2 Marching Squares fallback
        perim_lut = [0.0, 0.7071, 0.7071, 1.0, 0.7071, 1.0, 1.4142, 0.7071,
                     0.7071, 1.4142, 1.0, 0.7071, 1.0, 0.7071, 0.7071, 0.0]
        b0 = (blob_bin > 0).astype(np.uint8)
        perim = 0.0
        for ly in range(bh - 1):
            for lx in range(bw - 1):
                st = (int(b0[ly, lx]) << 3) | (int(b0[ly, lx + 1]) << 2) | (int(b0[ly + 1, lx + 1]) << 1) | int(b0[ly + 1, lx])
                perim += perim_lut[st]
        eff_area = float(area_px)

    if perim <= 1.0:
        return None

    circularity = (4.0 * math.pi * eff_area) / (perim * perim)

    ax = np.array(pts_x, dtype=np.float32)
    ay = np.array(pts_y, dtype=np.float32)
    cxx = float(np.var(ax))
    cyy = float(np.var(ay))
    cxy = float(np.mean((ax - np.mean(ax)) * (ay - np.mean(ay))))
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, (trace * trace) / 4.0 - det))
    lam1 = trace / 2.0 + disc
    lam2 = max(0.15, trace / 2.0 - disc)
    aspect_ratio = math.sqrt(lam1 / lam2)

    bbox = (int(np.min(ax)), int(np.min(ay)), int(np.max(ax)), int(np.max(ay)))
    return area_px, circularity, aspect_ratio, bbox


def _detect_structural_crowns_dog(
    arr_rgb: np.ndarray,
    lum_arr: np.ndarray,
    stage1_mask: np.ndarray,
    px_per_m: float,
    tree_type_str: str,
    corridor_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    h, w = lum_arr.shape
    mpp = 1.0 / max(px_per_m, 0.5)
    r, g, b = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]

    rg_spread = np.maximum(0.0, g - r * 0.85)
    base_struct = (0.22 * r + 0.62 * g + 0.16 * b) * np.where(g >= r * 0.88, 1.0, 0.45)
    canopy_sig = np.where(stage1_mask, np.clip(base_struct * 0.82 + rg_spread * 0.65, 0.0, 255.0), 0.0).astype(np.float32)

    dog_maps: List[np.ndarray] = []
    scale_radii_px: List[float] = []
    for r_m in (1.25, 2.0, 2.85):
        r_px = max(3.8, r_m / max(mpp, 0.12))
        scale_radii_px.append(r_px)
        sig_f = max(1.0, r_px * 0.52)
        sig_c = max(sig_f + 1.4, r_px * 1.45)
        if HAS_OPENCV:
            kf = max(3, int(round(sig_f * 4)) | 1)
            kc = max(5, int(round(sig_c * 4)) | 1)
            bf = cv2.GaussianBlur(canopy_sig, (kf, kf), sig_f)
            bc = cv2.GaussianBlur(canopy_sig, (kc, kc), sig_c)
        else:
            sig_pil = Image.fromarray(np.clip(canopy_sig, 0, 255).astype(np.uint8), mode="L")
            bf = np.array(sig_pil.filter(ImageFilter.GaussianBlur(radius=sig_f)), dtype=np.float32)
            bc = np.array(sig_pil.filter(ImageFilter.GaussianBlur(radius=sig_c)), dtype=np.float32)
        dog_maps.append(np.where(stage1_mask, bf - bc, -999.0))

    dogs_stacked = np.stack(dog_maps, axis=0)
    best_scale = np.argmax(dogs_stacked, axis=0)
    dog_max = np.max(dogs_stacked, axis=0)

    t_low = (tree_type_str or "").lower()
    is_palm = any(k in t_low for k in ["نخيل", "نخل", "palm", "date", "بلح"])
    is_windbreak = any(k in t_low for k in ["جازورين", "كازورينا", "كافور", "سرو", "casuarina", "eucalyptus", "مصدات"])

    cands: List[Dict[str, Any]] = []
    stride = max(2, int(round(1.0 / max(mpp, 0.12))))

    for y in range(4, h - 4, stride):
        for x in range(4, w - 4, stride):
            if not stage1_mask[y, x]:
                continue
            val = float(dog_max[y, x])
            if val < 4.5:
                continue

            s_idx = int(best_scale[y, x])
            win_dog = dog_maps[s_idx]
            patch = win_dog[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
            if float(np.max(patch)) > val:
                continue

            blob_res = _measure_blob_shape(win_dog, stage1_mask, x, y, val, scale_radii_px[s_idx] * 1.55)
            if blob_res is None:
                continue
            area_px, circularity, aspect_ratio, _ = blob_res

            eq_diam_m = 2.0 * math.sqrt(area_px / math.pi) * mpp
            if not (MIN_CROWN_DIAM_M <= eq_diam_m <= MAX_CROWN_DIAM_M):
                continue
            if circularity <= MIN_BLOB_CIRCULARITY or aspect_ratio > 2.2:
                continue

            pr, pg, pb = float(r[y, x]), float(g[y, x]), float(b[y, x])
            chroma_exg = (2.0 * pg - pr - pb) / (pr + pg + pb + 1e-6)
            if float(lum_arr[y, x]) < 40.0:
                continue
            if chroma_exg < 0.01 and pg < pr * 0.92:
                continue
            if max(pr, pg, pb) - min(pr, pg, pb) < 7.0:
                continue

            box_r = max(4, int(round((eq_diam_m / 2.0) / max(mpp, 0.12))))
            on_bound = bool(corridor_mask[y, x])
            if is_palm or (not on_bound and eq_diam_m >= 5.0):
                cls_id, cls_name = 1, "palm_tree"
            elif on_bound or is_windbreak:
                cls_id, cls_name = 2, "windbreak_tree"
            else:
                cls_id, cls_name = 0, "orchard_tree"

            cands.append({
                "bbox": (max(1, x - box_r), max(1, y - box_r), min(w - 2, x + box_r), min(h - 2, y + box_r)),
                "center": (x, y),
                "conf": min(0.96, 0.48 + circularity * 0.35 + min(0.18, val / 40.0)),
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
) -> List[Dict[str, Any]]:
    """
    Verifies paired directional cast shadow immediately adjacent to the crown edge:
    requires luminance drop >= 35% relative to crown center AND opposite sunlit side check.
    """
    h, w = lum_arr.shape
    mpp = 1.0 / max(px_per_m, 0.5)
    r_ch, g_ch, b_ch = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]

    verified: List[Dict[str, Any]] = []
    for cand in candidates:
        cx, cy = cand["center"]
        if not (2 <= cx < w - 2 and 2 <= cy < h - 2):
            continue
        if not strict_mask[cy, cx] or not stage1_mask[cy, cx]:
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
        if core_lum < 42.0:
            continue

        cr_p = r_ch[cy0:cy1, cx0:cx1]
        cg_p = g_ch[cy0:cy1, cx0:cx1]
        cb_p = b_ch[cy0:cy1, cx0:cx1]
        core_exg = float(np.mean((2.0 * cg_p - cr_p - cb_p) / (cr_p + cg_p + cb_p + 1e-6)))
        if core_exg < 0.01 and float(np.mean(cg_p >= cr_p * 0.92)) < 0.5:
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
    hsv_v = np.max(arr, axis=2) / 255.0

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

    # Stage 1: Flat-Crop Texture & Shadow Rejection Pre-Filter (4m x 4m grid cells)
    stage1_mask, cells_kept, cells_total, stage1_active_pct = _build_stage1_flat_crop_mask(
        arr, lum_arr, hsv_v, lap_mag, target_mask, px_per_m
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
    min_sep_px = max(6.0, 2.4 * px_per_m)

    # Zero out outside polygon AND all flat crop cells before Stage 2 detection
    stage1_masked_u8 = np.where(stage1_mask[:, :, None], arr, 0.0).astype(np.uint8)

    detected_boxes: List[Dict[str, Any]] = []
    used_yolo = False
    active_yolo_path = yolo_model_path
    if not active_yolo_path:
        for cand in ["best.pt", "models/best.pt", "/mount/src/farm-qa/best.pt", "/mount/src/farm-qa/models/best.pt"]:
            if os.path.exists(cand):
                active_yolo_path = cand
                break

    yolo_skipped_reason = ""
    raw_yolo_cands: List[Dict[str, Any]] = []
    if active_yolo_path and np.any(stage1_mask):
        yolo_model, skip_reason = _get_validated_tree_yolo(active_yolo_path)
        if yolo_model is not None:
            try:
                for sx1, sy1, sx2, sy2 in sahi_slices:
                    if not np.any(stage1_mask[sy1:sy2, sx1:sx2]):
                        continue
                    crop_arr = stage1_masked_u8[sy1:sy2, sx1:sx2]
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
                            if not (0 <= cy < height_px and 0 <= cx < width_px and stage1_mask[cy, cx]):
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
        fused = _apply_sahi_wbf_nms(raw_yolo_cands, iou_thresh=NMS_IOU_THRESH, min_dist_px=min_sep_px)
        detected_boxes = _verify_cast_shadow_and_telemetry(
            fused, arr, lum_arr, hsv_v, lap_mag, target_mask, stage1_mask,
            px_per_m, require_strict_cast_shadow=False,
        )
    elif np.any(stage1_mask):
        raw_dog = _detect_structural_crowns_dog(
            arr, lum_arr, stage1_mask, px_per_m, tree_type_str, corridor_mask
        )
        fused = _apply_sahi_wbf_nms(raw_dog, iou_thresh=NMS_IOU_THRESH, min_dist_px=min_sep_px)
        detected_boxes = _verify_cast_shadow_and_telemetry(
            fused, arr, lum_arr, hsv_v, lap_mag, target_mask, stage1_mask,
            px_per_m, require_strict_cast_shadow=True,
        )

    del arr, stage1_masked_u8, lum_arr, hsv_v, lap_mag, r, g, b

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
            "mean_ndvi": round(mean_ndvi, 3),
            "max_ndvi": round(max_ndvi, 3),
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

    ndvi_heatmap = generate_ndvi_colormap_image(ndvi_matrix, mask=parcel_mask) if generate_heatmap else None

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
        "annotated_image": annotated_img,
        "ndvi_heatmap_image": ndvi_heatmap,
        "engine_used": (
            f"YOLOv8 SAHI {slice_size}px ({os.path.basename(active_yolo_path)}, Conf>=0.55)"
            if used_yolo
            else "Two-Stage DoG Structural Crowns + Cast-Shadow (>=35%)"
        ),
        "yolo_model_loaded": used_yolo,
        "yolo_model_path": active_yolo_path if used_yolo else None,
        "yolo_skipped_reason": yolo_skipped_reason,
        "trees_classified": trees_classified,
    }
