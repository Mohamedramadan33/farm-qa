# -*- coding: utf-8 -*-
"""
ndvi_tree_analyzer.py
---------------------
GIS & Computer Vision Module for Agricultural Farm QA Audit:
1. Normalized Difference Vegetation Index (NDVI) & VARI computation for aerial/satellite imagery.
2. Healthy vs Damaged/Dead Tree classification (Healthy 🟢 vs Stressed 🟡 vs Dead/Damaged 🔴).
3. Visual Tree Counting & Localization (OpenCV Blob/Contour Detection + YOLO architecture support).
4. Canopy Health Mapping and Colormap Heatmap Generation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Optional headless OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Optional Arabic text shaping for PIL rendering (prevents [□□□□] glyph boxes)
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_get_display
    HAS_ARABIC_SHAPER = True
except ImportError:
    arabic_reshaper = None
    _bidi_get_display = None
    HAS_ARABIC_SHAPER = False

# Optional Ultralytics YOLO (lazy-imported only when validating a custom tree model)
HAS_YOLO = False

_YOLO_MODEL_CACHE: Dict[str, Any] = {}
_FONT_CACHE: Dict[int, Any] = {}

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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts", "NotoSansArabic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _get_arabic_pil_font(size: int = 14) -> Any:
    """Loads a valid UTF-8 Arabic TrueType font for PIL text rendering."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for fp in _ARABIC_FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                fnt = ImageFont.truetype(fp, size=size)
                _FONT_CACHE[size] = fnt
                return fnt
            except Exception:
                continue
    fnt = ImageFont.load_default()
    _FONT_CACHE[size] = fnt
    return fnt


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


def _get_validated_tree_yolo(model_path: str) -> Tuple[Optional[Any], str]:
    """
    Loads YOLO model (cached & lazy-imported) and verifies it defines tree/canopy classes
    matching either the 3-class schema (orchard_tree, palm_tree, windbreak_tree) or legacy 'tree'.
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
    Computes vegetation health index matrix [-1.0 to 1.0]:
    - If NIR and Red channels are provided: uses standard multispectral NDVI = (NIR - Red) / (NIR + Red)
    - If visible RGB is provided: uses standard satellite Visible Atmospherically Resistant Index (VARI)
      and Green-Leaf Normalized Index (GLI), which are the gold standard for visible drone/satellite imagery.
    """
    # 1. If NIR and Red channels explicitly provided
    if nir_channel is not None and red_channel is not None:
        nir = nir_channel.astype(np.float32)
        red = red_channel.astype(np.float32)
        denom = nir + red + 1e-7
        ndvi = (nir - red) / denom
        return np.clip(ndvi, -1.0, 1.0)

    # 2. Extract RGB channels
    if isinstance(image_input, Image.Image):
        arr = np.array(image_input.convert("RGB"), dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
    else:
        raise ValueError("image_input must be PIL Image or NumPy array")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Visible Atmospherically Resistant Index (VARI) = (G - R) / (G + R - B)
    # Normalized & stabilized for satellite / aerial RGB images:
    num = g - r
    denom = g + r - b
    # Protect against zero/negative denominators
    denom = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0 + 1e-5, denom)
    vari = num / denom

    # Combine with Green Leaf Index (GLI) = (2G - R - B) / (2G + R + B) for robust plant chlorophyll sensitivity
    denom_gli = 2.0 * g + r + b + 1e-5
    gli = (2.0 * g - r - b) / denom_gli

    # Weighted composite vegetation index (proxy NDVI)
    composite_ndvi = 0.5 * np.clip(vari, -1.0, 1.0) + 0.5 * np.clip(gli, -1.0, 1.0)
    return np.clip(composite_ndvi, -1.0, 1.0)


def generate_ndvi_colormap_image(ndvi_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> Image.Image:
    """
    Converts NDVI float matrix into a standard RdYlGn (Red-Yellow-Green) colored heatmap image:
    - Deep Red: Bare soil / rock / dead material (NDVI < 0.05)
    - Yellow/Orange: Stressed / dry / sparse vegetation (0.05 <= NDVI < 0.20)
    - Light Green: Moderate healthy vegetation (0.20 <= NDVI < 0.35)
    - Deep Vibrant Green: Dense healthy tree canopy (NDVI >= 0.35)
    """
    h, w = ndvi_matrix.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Normalized from [-0.2, 0.6] to [0.0, 1.0] for vivid visual contrast
    norm_val = np.clip((ndvi_matrix + 0.1) / 0.6, 0.0, 1.0)

    # Red, Yellow, Green color gradients
    # Low values (0.0 - 0.5): Red (215, 48, 39) to Yellow (254, 224, 139)
    # High values (0.5 - 1.0): Yellow (254, 224, 139) to Green (26, 152, 80)
    low_mask = norm_val < 0.5
    high_mask = ~low_mask

    # Low gradient: Red to Yellow
    t_low = norm_val[low_mask] * 2.0
    r_low = (215 + t_low * (254 - 215)).astype(np.uint8)
    g_low = (48 + t_low * (224 - 48)).astype(np.uint8)
    b_low = (39 + t_low * (139 - 39)).astype(np.uint8)

    # High gradient: Yellow to Deep Green
    t_high = (norm_val[high_mask] - 0.5) * 2.0
    r_high = (254 - t_high * (254 - 26)).astype(np.uint8)
    g_high = (224 - t_high * (224 - 152)).astype(np.uint8)
    b_high = (139 - t_high * (139 - 80)).astype(np.uint8)

    colored[low_mask, 0] = r_low
    colored[low_mask, 1] = g_low
    colored[low_mask, 2] = b_low

    colored[high_mask, 0] = r_high
    colored[high_mask, 1] = g_high
    colored[high_mask, 2] = b_high

    # Dim areas outside parcel boundary if mask is provided
    if mask is not None:
        outside = ~mask
        colored[outside] = (colored[outside] * 0.45).astype(np.uint8)

    return Image.fromarray(colored)


def compute_morphological_tophat(
    ndvi_matrix: np.ndarray,
    kernel_radius_px: int,
) -> np.ndarray:
    """
    Morphological White Top-Hat Transform: WTH(I) = I - Opening(I, B)
    Flat agricultural crops (vegetables, clover, wheat) form expansive continuous plateaus.
    When a morphological opening of radius B is performed, the broad crop plateau remains unchanged,
    so WTH = Plateau - Plateau = 0.
    In contrast, isolated 3D tree crowns are narrower than B, so opening removes them, leaving
    a high positive response in WTH!
    """
    # Normalize NDVI from [-0.2, 0.8] to [0, 255]
    norm_u8 = np.clip((ndvi_matrix + 0.2) / 1.0 * 255.0, 0, 255).astype(np.uint8)
    k_size = max(5, int(kernel_radius_px * 2) | 1)

    if HAS_OPENCV:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        tophat = cv2.morphologyEx(norm_u8, cv2.MORPH_TOPHAT, kernel)
        return tophat.astype(np.float32) / 255.0
    else:
        pil_img = Image.fromarray(norm_u8, mode="L")
        # Opening = MinFilter then MaxFilter
        opened = pil_img.filter(ImageFilter.MinFilter(k_size)).filter(ImageFilter.MaxFilter(k_size))
        diff = np.maximum(0, norm_u8.astype(np.int16) - np.array(opened, dtype=np.int16)).astype(np.uint8)
        return diff.astype(np.float32) / 255.0


def count_directional_drops(
    ndvi_mat: np.ndarray,
    cx: int,
    cy: int,
    r_drop: int,
    w: int,
    h: int,
    center_val: float,
    min_drop: float = 0.065,
) -> int:
    """
    Tests 8 compass rays around (cx, cy) at distance r_drop.
    Returns count of rays that drop by at least min_drop below center_val.
    A discrete tree crown is an isolated island, so at least 4 to 5 of 8 rays drop into soil/shadow/skirt.
    A continuous crop field (clover/vegetables) has green crops in all directions -> count is 0, 1, or 2!
    """
    angles = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
    drop_count = 0
    for a in angles:
        rx = int(round(cx + r_drop * math.cos(a)))
        ry = int(round(cy + r_drop * math.sin(a)))
        if 0 <= rx < w and 0 <= ry < h:
            val_at_ray = float(ndvi_mat[ry, rx])
            if (center_val - val_at_ray) >= min_drop:
                drop_count += 1
        else:
            drop_count += 1
    return drop_count


def _generate_sahi_slices(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    width_px: int,
    height_px: int,
    slice_size: int = 512,
    overlap: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """
    Generates overlapping SAHI windows (e.g. 512x512 with 25% overlap -> stride 384px)
    covering the buffered farm parcel bounding box so no tree on a slice seam is split.
    """
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

    slices: List[Tuple[int, int, int, int]] = []
    for sy in ys:
        sy2 = min(height_px, sy + s_size)
        for sx in xs:
            sx2 = min(width_px, sx + s_size)
            slices.append((sx, sy, sx2, sy2))
    return slices


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
    iou_thresh: float = 0.45,
    min_dist_px: float = 8.0,
) -> List[Dict[str, Any]]:
    """
    Weighted Boxes Fusion (WBF) + IoU NMS (default IoU = 0.45) across overlapping SAHI slices.
    Merges duplicate detections from overlapping tile borders using confidence weighting
    while preserving multi-class labels (orchard_tree, palm_tree, windbreak_tree).
    """
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
            })
        else:
            mc = matched_cluster
            if cconf > mc["max_conf"]:
                mc["class_id"] = c_cls_id
                mc["class_name"] = c_cls_name
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
        fx1 = int(round(cl["wx1"] / w))
        fy1 = int(round(cl["wy1"] / w))
        fx2 = int(round(cl["wx2"] / w))
        fy2 = int(round(cl["wy2"] / w))
        fcx = int(round(cl["wcx"] / w))
        fcy = int(round(cl["wcy"] / w))
        frad = max(3, int(round(cl["wrad"] / w)))
        fused.append({
            "bbox": (fx1, fy1, fx2, fy2),
            "center": (fcx, fcy),
            "conf": round(min(1.0, cl["max_conf"]), 4),
            "crown_radius": frad,
            "class_id": cl["class_id"],
            "class_name": cl["class_name"],
        })
    return fused


def _verify_shadow_and_laplacian_texture(
    candidates: List[Dict[str, Any]],
    arr_rgb: np.ndarray,
    gray_arr: np.ndarray,
    exg: np.ndarray,
    target_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Stage-3 Cast-Shadow (HSV Value drop) + Laplacian Texture Variance Post-Filter.
    Applied across all YOLO and CV candidates to reject flat row crops (alfalfa, wheat, clover).
    """
    h, w = gray_arr.shape
    hsv_v = np.max(arr_rgb, axis=2) / 255.0

    if HAS_OPENCV:
        lap_full = np.abs(cv2.Laplacian(gray_arr.astype(np.float32), cv2.CV_32F, ksize=3))
    else:
        padded = np.pad(gray_arr.astype(np.float32), 1, mode="edge")
        lap_full = np.abs(
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray_arr
        )

    verified: List[Dict[str, Any]] = []
    for cand in candidates:
        cx, cy = cand["center"]
        if not (1 <= cx < w - 1 and 1 <= cy < h - 1 and target_mask[cy, cx]):
            continue

        bx1, by1, bx2, by2 = cand["bbox"]
        x1, y1 = max(1, bx1), max(1, by1)
        x2, y2 = min(w - 2, bx2), min(h - 2, by2)
        if x2 <= x1 or y2 <= y1:
            continue

        crown_exg = float(np.mean(exg[y1:y2 + 1, x1:x2 + 1]))
        if crown_exg < 2.0:
            continue

        lap_var = float(np.var(lap_full[y1:y2 + 1, x1:x2 + 1]))
        lum_std = float(np.std(gray_arr[y1:y2 + 1, x1:x2 + 1]))

        rad = max(3, int(cand.get("crown_radius", 5)))
        sy1 = max(0, cy - int(rad * 2.5))
        sy2 = min(h, cy + int(rad * 1.4) + 1)
        sx1 = max(0, cx - int(rad * 2.4))
        sx2 = min(w, cx + int(rad * 2.4) + 1)
        crown_v = float(np.mean(hsv_v[max(0, cy - 2):min(h, cy + 3), max(0, cx - 2):min(w, cx + 3)]))
        patch_v = hsv_v[sy1:sy2, sx1:sx2]
        min_v = float(np.min(patch_v)) if patch_v.size > 0 else crown_v
        shadow_drop = crown_v - min_v
        dark_px = int(np.sum(patch_v < 0.24)) if patch_v.size > 0 else 0
        has_cast_shadow = (shadow_drop >= 0.08 and min_v <= 0.27) or dark_px >= 3

        if not has_cast_shadow and lap_var < 26.0 and lum_std < 6.5:
            continue
        if lap_var < 12.0 and lum_std < 4.8:
            continue

        out_cand = dict(cand)
        out_cand["shadow_drop"] = round(shadow_drop, 3)
        out_cand["laplacian_var"] = round(lap_var, 1)
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
    tile_overlap: float = 0.20,
    px_per_m: float = 3.5,
    slice_size: int = 512,
    inward_buffer_m: float = 0.0,
    show_compact_ids: bool = False,
) -> Dict[str, Any]:
    """
    High-Precision Visual Tree Detection & Flat Crop Rejection Engine (SAHI + WBF):
    - SAHI 512x512 overlapping window slicing (25% overlap) + Weighted Boxes Fusion (WBF).
    - Exact 3-meter GPS polygon buffer masking for perimeter shelterbelts.
    - Dual spectral check (ExG 2G-R-B + VARI/GLI) + cast-shadow dipole verification.
    - Morphological White Top-Hat & Multi-Scale DoG for Total Crop Carpet Elimination.
    - Classifies health status: Healthy (🟢), Stressed (🟡), Dead/Damaged (🔴).
    """
    orig_img = satellite_image.convert("RGB")
    width_px, height_px = orig_img.size

    # 1. Compute NDVI matrix & spectral channels upfront (shared by YOLO SAHI filter & CV engine)
    ndvi_matrix = compute_ndvi_or_vari(orig_img)
    arr = np.array(orig_img, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    exg = 2.0 * g - r - b
    brightness = (r + g + b) / 3.0

    # 2. Build parcel polygon mask + exact 3-meter boundary buffer for windbreak shelterbelts
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
            eroded_pil = parcel_mask_img.filter(ImageFilter.MinFilter(erode_px * 2 + 1))
            parcel_mask = np.array(eroded_pil) > 0

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
    min_sep_px = max(7.5, pixel_spacing_px * 0.70)

    # 3. Candidate Tree Detection (SAHI + WBF on custom YOLO or Multi-Scale Spectral CV)
    detected_boxes: List[Dict[str, Any]] = []
    used_yolo = False
    active_yolo_path = yolo_model_path

    if not active_yolo_path:
        for cand in ["best.pt", "models/best.pt", "/mount/src/farm-qa/best.pt", "/mount/src/farm-qa/models/best.pt"]:
            if os.path.exists(cand):
                active_yolo_path = cand
                break

    # Strict Masked Inference Array: zero all pixels outside target polygon mask
    masked_arr_u8 = np.where(target_mask[:, :, None], arr, 0.0).astype(np.uint8)
    gray_full = 0.299 * r + 0.587 * g + 0.114 * b

    # Attempt YOLO deep learning detection with 512x512 SAHI slicing ONLY if model defines tree/canopy classes
    yolo_skipped_reason = ""
    if active_yolo_path:
        yolo_model, skip_reason = _get_validated_tree_yolo(active_yolo_path)
        if yolo_model is not None:
            try:
                raw_yolo_cands: List[Dict[str, Any]] = []
                for sx1, sy1, sx2, sy2 in sahi_slices:
                    if not np.any(target_mask[sy1:sy2, sx1:sx2]):
                        continue
                    crop_arr = masked_arr_u8[sy1:sy2, sx1:sx2]
                    yolo_res = yolo_model.predict(source=crop_arr, conf=0.20, iou=0.45, verbose=False)
                    if yolo_res and len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                        for b_box in yolo_res[0].boxes:
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
                                "conf": float(b_box.conf[0].cpu().numpy()),
                                "crown_radius": crown_rad,
                                "class_id": cls_id,
                                "class_name": cls_name,
                            })
                if raw_yolo_cands:
                    fused_yolo = _apply_sahi_wbf_nms(raw_yolo_cands, iou_thresh=0.45, min_dist_px=min_sep_px)
                    detected_boxes = _verify_shadow_and_laplacian_texture(
                        fused_yolo, arr, gray_full, exg, target_mask
                    )
                    used_yolo = len(detected_boxes) > 0
            except Exception:
                used_yolo = False
        elif skip_reason:
            yolo_skipped_reason = skip_reason

    # High-Precision Multi-Scale Computer Vision & SAHI Overlapping Window Engine
    if not detected_boxes:

        # Species detection and scale band weighting
        t_low = (tree_type_str or "").lower()
        is_palm = any(k in t_low for k in ["نخيل", "نخل", "palm", "date", "بلح"])
        is_casuarina = any(k in t_low for k in ["جازورين", "كافور", "سرو", "أكاسيا", "casuarina", "eucalyptus", "cypress", "acacia", "مصدات"])
        is_citrus_olive = any(k in t_low for k in ["موالح", "برتقال", "ليمون", "يوسفي", "زيتون", "رمّان", "رمان", "مانجو", "citrus", "orange", "lemon", "olive", "pomegranate", "mango"])

        # Weights across 3 spatial frequency bands:
        # Band 1: Small crowns (1.8m - 3.2m)
        # Band 2: Medium crowns (3.2m - 5.5m)
        # Band 3: Large crowns (5.5m - 9.0m)
        if is_palm:
            w1, w2, w3 = 0.80, 1.10, 1.45
        elif is_casuarina:
            w1, w2, w3 = 1.45, 1.05, 0.70
        elif is_citrus_olive:
            w1, w2, w3 = 0.95, 1.40, 1.05
        else:
            w1, w2, w3 = 1.05, 1.20, 1.00

        # Multi-scale Gaussian Pyramid of NDVI
        ndvi_u8 = ((np.clip(ndvi_matrix, -0.2, 0.8) + 0.2) / 1.0 * 255.0).astype(np.uint8)
        ndvi_pil = Image.fromarray(ndvi_u8, mode="L")

        # Scale 1: Small crowns
        sig_fine1 = max(0.9, pixel_spacing_px * 0.12)
        sig_coarse1 = max(1.9, pixel_spacing_px * 0.28)
        b_fine1 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_fine1)), dtype=np.float32)
        b_coarse1 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_coarse1)), dtype=np.float32)
        dog1 = (b_fine1 - b_coarse1) * w1

        # Scale 2: Medium crowns
        sig_fine2 = max(1.7, pixel_spacing_px * 0.22)
        sig_coarse2 = max(3.5, pixel_spacing_px * 0.52)
        b_fine2 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_fine2)), dtype=np.float32)
        b_coarse2 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_coarse2)), dtype=np.float32)
        dog2 = (b_fine2 - b_coarse2) * w2

        # Scale 3: Large crowns
        sig_fine3 = max(2.8, pixel_spacing_px * 0.35)
        sig_coarse3 = max(5.5, pixel_spacing_px * 0.78)
        b_fine3 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_fine3)), dtype=np.float32)
        b_coarse3 = np.array(ndvi_pil.filter(ImageFilter.GaussianBlur(radius=sig_coarse3)), dtype=np.float32)
        dog3 = (b_fine3 - b_coarse3) * w3

        # Stack and find peak response scale
        dogs_stacked = np.stack([dog1, dog2, dog3], axis=0)
        best_scale_idx = np.argmax(dogs_stacked, axis=0)  # 0, 1, or 2
        dog_canopy = np.max(dogs_stacked, axis=0)

        # Morphological White Top-Hat Transform for Total Crop Carpet Elimination
        # Removes expansive flat green fields (clover, wheat, alfalfa, vegetables)
        k_tophat = max(5, int(pixel_spacing_px * 0.90))
        tophat_matrix = compute_morphological_tophat(ndvi_matrix, kernel_radius_px=k_tophat)

        # Dynamic Calibration Factors based on Sensitivity Slider (0.15 to 0.90)
        sens_clamp = float(np.clip(sensitivity, 0.15, 0.95))
        # rigor_scale: at 0.50 -> 1.0; at 0.30 -> 1.66 (very strict); at 0.75 -> 0.66 (sensitive)
        rigor_scale = float(0.50 / sens_clamp)

        tophat_th = 0.050 * rigor_scale
        dog_th = 6.0 * rigor_scale
        drop_th = 0.060 * rigor_scale
        min_drop_val = 0.055 * rigor_scale
        ratio_th = min(0.88, 0.81 + (sens_clamp - 0.50) * 0.12)

        # Sky-diffuse Blue Shadow Detection
        blue_ratio = b / (r + g + b + 1e-5)
        shadow_map = (blue_ratio > 0.32) & (brightness < 92.0)

        # Grayscale luminance array for micro-texture roughness
        gray_arr = 0.299 * r + 0.587 * g + 0.114 * b
        r_core = max(2, int(pixel_spacing_px * 0.25))
        r_ring = max(r_core + 2, int(pixel_spacing_px * 0.60))

        # Find true physical tree canopy peaks per overlapping SAHI slice, then fuse with WBF
        step_px = max(2, int(pixel_spacing_px * 0.28))
        dog_th_cand = max(2.5, 3.8 * rigor_scale)
        raw_cv_cands: List[Dict[str, Any]] = []

        seen_interior_peaks = set()

        for sx1, sy1, sx2, sy2 in sahi_slices:
            if not np.any(target_mask[sy1:sy2, sx1:sx2]):
                continue
            slice_seen_peaks = set()
            for gy in range(sy1, sy2, step_px):
                for gx in range(sx1, sx2, step_px):
                    if not target_mask[gy, gx]:
                        continue

                    w_y1 = max(sy1, gy - r_core)
                    w_y2 = min(sy2, gy + r_core + 1)
                    w_x1 = max(sx1, gx - r_core)
                    w_x2 = min(sx2, gx + r_core + 1)

                    sub_dog = dog_canopy[w_y1:w_y2, w_x1:w_x2]
                    sub_mask = target_mask[w_y1:w_y2, w_x1:w_x2]
                    if sub_dog.size == 0 or not np.any(sub_mask):
                        continue

                    masked_dog = np.where(sub_mask, sub_dog, -999.0)
                    max_pos = np.unravel_index(np.argmax(masked_dog), masked_dog.shape)
                    cand_y = w_y1 + int(max_pos[0])
                    cand_x = w_x1 + int(max_pos[1])
                    dog_peak = float(sub_dog[max_pos])

                    if dog_peak < dog_th_cand:
                        continue

                    if (cand_x, cand_y) in slice_seen_peaks or (cand_x, cand_y) in seen_interior_peaks:
                        continue
                    slice_seen_peaks.add((cand_x, cand_y))

                    on_slice_margin = (
                        (sx1 > 0 and cand_x < sx1 + r_ring)
                        or (sx2 < width_px and cand_x >= sx2 - r_ring)
                        or (sy1 > 0 and cand_y < sy1 + r_ring)
                        or (sy2 < height_px and cand_y >= sy2 - r_ring)
                    )
                    if not on_slice_margin:
                        seen_interior_peaks.add((cand_x, cand_y))

                    cand_ndvi = float(ndvi_matrix[cand_y, cand_x])
                    cand_exg = float(exg[cand_y, cand_x])
                    cand_bright = float(brightness[cand_y, cand_x])
                    cand_tophat = float(tophat_matrix[cand_y, cand_x])

                    # Tree crown requires green foliage presence (relaxed for dark casuarinas/palms)
                    if cand_ndvi < 0.08 or cand_exg < 3.0:
                        continue

                    # 1. Physical Tall-Tree Cast Shadow in Sun-Opposing Northern Quadrant (North / North-East in Egypt)
                    # Trees in Egypt cast deep shadows (Brightness < 55) northward/northeastward onto ground/roads
                    sh_y1 = max(0, cand_y - int(pixel_spacing_px * 2.2))
                    sh_y2 = max(0, cand_y - 2)
                    sh_x1 = max(0, cand_x - int(pixel_spacing_px * 1.5))
                    sh_x2 = min(width_px, cand_x + int(pixel_spacing_px * 1.8) + 1)
                    sh_patch = brightness[sh_y1:sh_y2, sh_x1:sh_x2]
                    has_tall_shadow = False
                    if sh_patch.size >= 6:
                        dark_pixels = np.sum(sh_patch < 54.0)
                        min_sh = float(np.min(sh_patch))
                        if min_sh < 50.0 and dark_pixels >= 3:
                            has_tall_shadow = True

                    # 2. Opposing Directional Pairs Drop Test (عزل تام وحاسم لحواف الحقول والخطوط العادية):
                    # A 1D step-edge (field border, ditch, road line) drops in ONLY ONE direction (towards road)
                    # while the opposite ray (towards field) remains high.
                    # A 2D tree dome drops in BOTH opposite directions across its cross-section!
                    angles = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
                    two_sided_drop_pairs = 0
                    step_edge_pairs = 0
                    drop_rays_total = 0

                    for p_idx in range(4):
                        a1 = angles[p_idx]
                        a2 = angles[p_idx + 4]
                        rx1 = int(round(cand_x + r_ring * math.cos(a1)))
                        ry1 = int(round(cand_y + r_ring * math.sin(a1)))
                        rx2 = int(round(cand_x + r_ring * math.cos(a2)))
                        ry2 = int(round(cand_y + r_ring * math.sin(a2)))

                        val1 = float(ndvi_matrix[ry1, rx1]) if (0 <= rx1 < width_px and 0 <= ry1 < height_px) else -0.5
                        val2 = float(ndvi_matrix[ry2, rx2]) if (0 <= rx2 < width_px and 0 <= ry2 < height_px) else -0.5

                        d1 = cand_ndvi - val1
                        d2 = cand_ndvi - val2

                        if d1 >= min_drop_val:
                            drop_rays_total += 1
                        if d2 >= min_drop_val:
                            drop_rays_total += 1

                        if d1 >= 0.045 and d2 >= 0.045:
                            two_sided_drop_pairs += 1
                        elif (d1 >= 0.12 and d2 < 0.02) or (d2 >= 0.12 and d1 < 0.02):
                            step_edge_pairs += 1

                    # If it is purely a 1D step edge with NO opposing drops on any axis: REJECT IT!
                    # (This completely eliminates the necklace of fake points along the boundary vector line)
                    if two_sided_drop_pairs == 0 and not has_tall_shadow:
                        continue
                    if step_edge_pairs >= 2 and two_sided_drop_pairs == 0:
                        continue

                    # 3. Luminance Micro-Texture Roughness (عزل المحاصيل والبرسيم المسطح الأملس):
                    patch_gray = gray_arr[max(0, cand_y - 3):min(height_px, cand_y + 4),
                                          max(0, cand_x - 3):min(width_px, cand_x + 4)]
                    lum_std = float(np.std(patch_gray)) if patch_gray.size >= 9 else 0.0
                    if lum_std < 6.8 and not has_tall_shadow:
                        # Smooth flat agricultural crop carpet -> REJECT!
                        continue

                    # 4. Radial Annular Drop (Tree Dome vs Flat Field)
                    ring_y1 = max(0, cand_y - r_ring)
                    ring_y2 = min(height_px, cand_y + r_ring + 1)
                    ring_x1 = max(0, cand_x - r_ring)
                    ring_x2 = min(width_px, cand_x + r_ring + 1)

                    ring_ndvi_crop = ndvi_matrix[ring_y1:ring_y2, ring_x1:ring_x2]
                    core_ndvi_crop = ndvi_matrix[max(0, cand_y - r_core):min(height_px, cand_y + r_core + 1),
                                                 max(0, cand_x - r_core):min(width_px, cand_x + r_core + 1)]

                    mean_core = float(np.mean(core_ndvi_crop)) if core_ndvi_crop.size > 0 else cand_ndvi
                    mean_ring = float(np.mean(ring_ndvi_crop)) if ring_ndvi_crop.size > 0 else mean_core
                    annular_drop = mean_core - mean_ring
                    ring_ratio = mean_ring / (mean_core + 1e-5)

                    # Acceptance Criteria:
                    # Type A: Upright tree with tall cast shadow (e.g. palms, casuarinas, boundary trees)
                    # Type B: Isolated standalone 2D tree dome with two-sided drops on multiple axes
                    is_real_tree = False
                    if has_tall_shadow and (two_sided_drop_pairs >= 1 or drop_rays_total >= 3):
                        is_real_tree = True
                    elif two_sided_drop_pairs >= 2 and annular_drop >= drop_th and ring_ratio <= ratio_th and drop_rays_total >= 4:
                        ring_bright = brightness[ring_y1:ring_y2, ring_x1:ring_x2]
                        min_ring_b = float(np.min(ring_bright)) if ring_bright.size > 0 else 100.0
                        if min_ring_b < 75.0 or cand_tophat >= tophat_th:
                            is_real_tree = True

                    if not is_real_tree:
                        continue

                    # Sub-pixel Quadratic Apex Interpolation
                    dx_sub, dy_sub = 0.0, 0.0
                    if 1 <= cand_x < width_px - 1 and 1 <= cand_y < height_px - 1:
                        dx1 = float(dog_canopy[cand_y, cand_x - 1])
                        dx2 = float(dog_canopy[cand_y, cand_x])
                        dx3 = float(dog_canopy[cand_y, cand_x + 1])
                        denom_x = 2.0 * (2.0 * dx2 - dx1 - dx3)
                        if abs(denom_x) > 1e-4:
                            dx_sub = float(np.clip((dx3 - dx1) / denom_x, -0.45, 0.45))

                        dy1 = float(dog_canopy[cand_y - 1, cand_x])
                        dy2 = float(dog_canopy[cand_y, cand_x])
                        dy3 = float(dog_canopy[cand_y + 1, cand_x])
                        denom_y = 2.0 * (2.0 * dy2 - dy1 - dy3)
                        if abs(denom_y) > 1e-4:
                            dy_sub = float(np.clip((dy3 - dy1) / denom_y, -0.45, 0.45))

                    sub_cx = int(round(cand_x + dx_sub))
                    sub_cy = int(round(cand_y + dy_sub))

                    if not (0 <= sub_cx < width_px and 0 <= sub_cy < height_px and target_mask[sub_cy, sub_cx]):
                        continue

                    win_scale = int(best_scale_idx[cand_y, cand_x])
                    if win_scale == 0:
                        crown_radius = max(3, int(pixel_spacing_px * 0.24))
                    elif win_scale == 1:
                        crown_radius = max(5, int(pixel_spacing_px * 0.38))
                    else:
                        crown_radius = max(7, int(pixel_spacing_px * 0.54))

                    if is_palm or win_scale == 2:
                        cls_id, cls_name = 1, "palm_tree"
                    elif is_casuarina or bool(corridor_mask[sub_cy, sub_cx]):
                        cls_id, cls_name = 2, "windbreak_tree"
                    else:
                        cls_id, cls_name = 0, "orchard_tree"

                    raw_cv_cands.append({
                        "bbox": (sub_cx - crown_radius, sub_cy - crown_radius, sub_cx + crown_radius, sub_cy + crown_radius),
                        "center": (sub_cx, sub_cy),
                        "conf": min(1.0, float(max(annular_drop * 6.0, cand_tophat * 8.0, dog_peak / 20.0))),
                        "crown_radius": crown_radius,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

        # Fuse detections across overlapping SAHI windows with Weighted Boxes Fusion (IoU = 0.45)
        fused_cv = _apply_sahi_wbf_nms(raw_cv_cands, iou_thresh=0.45, min_dist_px=min_sep_px)
        detected_boxes = _verify_shadow_and_laplacian_texture(
            fused_cv, arr, gray_full, exg, target_mask
        )

        # Explicitly release large intermediate arrays before image rendering
        del dogs_stacked, dog1, dog2, dog3, dog_canopy, best_scale_idx, tophat_matrix, shadow_map, gray_arr

    del arr, masked_arr_u8, gray_full, r, g, b, exg, brightness

    # 4. Classify Health (Healthy vs Stressed vs Dead/Damaged) for each detected tree
    trees_classified: List[Dict[str, Any]] = []
    healthy_count = 0
    stressed_count = 0
    damaged_count = 0

    for idx, item in enumerate(detected_boxes):
        bx1, by1, bx2, by2 = item["bbox"]
        cx, cy = item["center"]
        crown_rad = item.get("crown_radius", max(4, int((bx2 - bx1) // 2)))

        # Clamp box to image limits
        cx1 = max(0, min(width_px - 1, bx1))
        cy1 = max(0, min(height_px - 1, by1))
        cx2 = max(0, min(width_px - 1, bx2))
        cy2 = max(0, min(height_px - 1, by2))

        # Sample NDVI inside tree crown box
        crown_ndvi_crop = ndvi_matrix[cy1:cy2 + 1, cx1:cx2 + 1]
        if crown_ndvi_crop.size > 0:
            mean_ndvi = float(np.mean(crown_ndvi_crop))
            max_ndvi = float(np.max(crown_ndvi_crop))
        else:
            mean_ndvi = float(ndvi_matrix[cy, cx])
            max_ndvi = mean_ndvi

        # Health state determination
        if mean_ndvi >= healthy_ndvi_threshold or max_ndvi >= (healthy_ndvi_threshold + 0.05):
            status = "healthy"
            status_ar = "سليمة / حيوية عالية"
            color_rgb = (34, 197, 94)    # Green
            healthy_count += 1
        elif mean_ndvi >= stressed_ndvi_threshold:
            status = "stressed"
            status_ar = "مجهدة / حيوية متوسطة"
            color_rgb = (234, 179, 8)    # Yellow
            stressed_count += 1
        else:
            status = "damaged"
            status_ar = "ميتة أو تالفة / جافة"
            color_rgb = (239, 68, 68)    # Red
            damaged_count += 1

        trees_classified.append({
            "tree_id": idx + 1,
            "center": (cx, cy),
            "bbox": (cx1, cy1, cx2, cy2),
            "crown_radius": crown_rad,
            "class_id": int(item.get("class_id", 0)),
            "class_name": str(item.get("class_name", "orchard_tree")),
            "conf": float(item.get("conf", 0.5)),
            "shadow_drop": float(item.get("shadow_drop", 0.0)),
            "laplacian_var": float(item.get("laplacian_var", 0.0)),
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

    # 5. Render Clean Visual Detection Image (Clean Canopy Circles + UTF-8 Arabic Font, zero overlapping dark boxes)
    annotated_img = orig_img.copy()
    draw_ann = ImageDraw.Draw(annotated_img)
    ar_font_sm = _get_arabic_pil_font(12)
    ar_font_hdr = _get_arabic_pil_font(14)

    # Draw farm boundary in bright turquoise
    if parcel_poly_px and len(parcel_poly_px) >= 3:
        draw_ann.line(parcel_poly_px + [parcel_poly_px[0]], fill=(0, 255, 200), width=3)

    for t in trees_classified:
        cx, cy = t["center"]
        col = t["color_rgb"]
        cr = t.get("crown_radius", 5)

        # Clean concentric canopy circle + high-contrast center pinpoint (no cluttered dark rectangles)
        draw_ann.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=col, width=2)
        draw_ann.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 0, 0))
        draw_ann.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)

        if show_compact_ids:
            tid = t["tree_id"]
            id_txt = f"{tid}"
            draw_ann.text((cx + cr + 2, cy - 6), id_txt, fill=(255, 255, 255), font=ar_font_sm)

    # Top Statistical Banner with reshaped Arabic text (prevents [□□□□] glyph boxes)
    banner_h = 38
    draw_ann.rectangle([0, 0, width_px, banner_h], fill=(15, 23, 42))
    header_txt = _format_arabic_text(
        f"إجمالي الأشجار: {total_trees} | سليمة: {healthy_count} ({healthy_pct}%) | "
        f"مجهدة: {stressed_count} ({stressed_pct}%) | تالفة: {damaged_count} ({damaged_pct}%)"
    )
    draw_ann.text((12, 9), header_txt, fill=(0, 255, 200), font=ar_font_hdr)

    # Generate optional NDVI heatmap image only when explicitly requested
    ndvi_heatmap = generate_ndvi_colormap_image(ndvi_matrix, mask=parcel_mask) if generate_heatmap else None

    # Compute overall vegetation coverage percentage
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
        "annotated_image": annotated_img,
        "ndvi_heatmap_image": ndvi_heatmap,
        "engine_used": f"YOLOv8 SAHI {slice_size}px ({os.path.basename(active_yolo_path)})" if used_yolo else f"SAHI {slice_size}px ({int(tile_overlap * 100)}% Overlap) + OpenCV WBF",
        "yolo_model_loaded": used_yolo,
        "yolo_model_path": active_yolo_path if used_yolo else None,
        "yolo_skipped_reason": yolo_skipped_reason,
        "trees_classified": trees_classified,
    }
