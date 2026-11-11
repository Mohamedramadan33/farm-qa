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

# Optional Ultralytics YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


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
) -> Dict[str, Any]:
    """
    High-Precision Visual Tree Detection & Flat Crop Rejection Engine:
    - Morphological White Top-Hat Transform for Total Crop Carpet Elimination (حقول البرسيم والخضار).
    - Multi-Directional Radial Drop (8 compass rays) requiring discrete island isolation.
    - Multi-Scale Gaussian Pyramid across 3 spatial frequency bands (Small 1.8-3.2m, Medium 3.2-5.5m, Large 5.5-9.0m).
    - Species-aware scale weighting (Palms vs Casuarina vs Citrus vs Olive).
    - Sub-Pixel Quadratic Apex Interpolation for finding the exact physical zenith of the tree dome.
    - 3D Shadow / Relief Dipole Contrast Test.
    - Classifies health status:
        🟢 Healthy (أشجار سليمة مورقة): High NDVI / Vigorous green crown.
        🟡 Stressed (أشجار مجهدة / عطش أو نقص تغذية): Moderate NDVI / Pale green crown.
        🔴 Dead / Damaged (أشجار ميتة أو تالفة أو جافة): Low/Negative NDVI / Desiccated crown.
    """
    orig_img = satellite_image.convert("RGB")
    width_px, height_px = orig_img.size

    # 1. Compute NDVI matrix
    ndvi_matrix = compute_ndvi_or_vari(orig_img)

    # 2. Build parcel polygon mask
    parcel_mask_img = Image.new("L", (width_px, height_px), 0)
    draw_mask = ImageDraw.Draw(parcel_mask_img)
    if parcel_poly_px and len(parcel_poly_px) >= 3:
        draw_mask.polygon(parcel_poly_px, outline=255, fill=255)
    else:
        draw_mask.rectangle([0, 0, width_px, height_px], fill=255)

    parcel_mask = np.array(parcel_mask_img) > 0
    total_parcel_pixels = int(np.sum(parcel_mask))

    # 3. Candidate Tree Detection (Auto-detect best.pt in app root or models/)
    detected_boxes: List[Dict[str, Any]] = []
    used_yolo = False
    active_yolo_path = yolo_model_path

    if not active_yolo_path:
        for cand in ["best.pt", "models/best.pt", "/mount/src/farm-qa/best.pt", "/mount/src/farm-qa/models/best.pt"]:
            if os.path.exists(cand):
                active_yolo_path = cand
                break

    # Attempt YOLO deep learning detection if available
    if HAS_YOLO and active_yolo_path:
        try:
            model = YOLO(active_yolo_path)
            yolo_res = model.predict(source=np.array(orig_img), conf=0.18, verbose=False)
            if yolo_res and len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
                for b in yolo_res[0].boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    if 0 <= cy < height_px and 0 <= cx < width_px and parcel_mask[cy, cx]:
                        crown_rad = max(4, int(max(bx2 - bx1, by2 - by1) // 2))
                        detected_boxes.append({
                            "bbox": (bx1, by1, bx2, by2),
                            "center": (cx, cy),
                            "conf": float(b.conf[0].cpu().numpy()),
                            "crown_radius": crown_rad,
                        })
                if detected_boxes:
                    used_yolo = True
        except Exception:
            used_yolo = False

    # High-Precision Multi-Scale Computer Vision & Annular Drop Engine
    if not detected_boxes:
        arr = np.array(orig_img, dtype=np.float32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        exg = 2.0 * g - r - b
        brightness = (r + g + b) / 3.0

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

        # Bounding box of parcel
        if parcel_poly_px:
            min_x = max(0, min(p[0] for p in parcel_poly_px))
            max_x = min(width_px - 1, max(p[0] for p in parcel_poly_px))
            min_y = max(0, min(p[1] for p in parcel_poly_px))
            max_y = min(height_px - 1, max(p[1] for p in parcel_poly_px))
        else:
            min_x, max_x = 0, width_px - 1
            min_y, max_y = 0, height_px - 1

        placed_centers: List[Tuple[int, int]] = []
        r_core = max(2, int(pixel_spacing_px * 0.25))
        r_ring = max(r_core + 2, int(pixel_spacing_px * 0.60))
        min_sep_px = max(8.0, pixel_spacing_px * 0.70)

        # Grayscale luminance array for micro-texture roughness
        gray_arr = 0.299 * r + 0.587 * g + 0.114 * b

        # Boundary corridor width: ~12 meters in pixels
        corridor_px = max(14.0, pixel_spacing_px * 1.35)

        # Construct spatial search mask: always cover the parcel plus immediate boundary buffer
        corridor_mask = np.zeros((height_px, width_px), dtype=bool)
        if parcel_poly_px and len(parcel_poly_px) >= 3:
            corr_img = Image.new("L", (width_px, height_px), 0)
            draw_corr = ImageDraw.Draw(corr_img)
            draw_corr.line(parcel_poly_px + [parcel_poly_px[0]], fill=255, width=int(corridor_px * 2.0))
            corridor_mask = np.array(corr_img) > 0

        # Always search the entire parcel and boundary buffer for true trees
        # (never restrict search to an artificial 1D line that generates fake boundary necklaces)
        target_mask = parcel_mask | corridor_mask

        # Find true physical tree canopy peaks across target_mask
        step_px = max(2, int(pixel_spacing_px * 0.28))
        dog_th_cand = max(2.5, 3.8 * rigor_scale)

        for gy in range(min_y, max_y + 1, step_px):
            for gx in range(min_x, max_x + 1, step_px):
                if not target_mask[gy, gx]:
                    continue

                w_y1 = max(0, gy - r_core)
                w_y2 = min(height_px, gy + r_core + 1)
                w_x1 = max(0, gx - r_core)
                w_x2 = min(width_px, gx + r_core + 1)

                sub_dog = dog_canopy[w_y1:w_y2, w_x1:w_x2]
                sub_mask = target_mask[w_y1:w_y2, w_x1:w_x2]
                if not np.any(sub_mask):
                    continue

                masked_dog = np.where(sub_mask, sub_dog, -999.0)
                max_pos = np.unravel_index(np.argmax(masked_dog), masked_dog.shape)
                cand_y = w_y1 + int(max_pos[0])
                cand_x = w_x1 + int(max_pos[1])
                dog_peak = float(sub_dog[max_pos])

                if dog_peak < dog_th_cand:
                    continue

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

                # Check if this tree is within boundary corridor if boundary_only mode is requested
                if detection_mode == "boundary_only" and not corridor_mask[cand_y, cand_x]:
                    continue

                if not any(math.hypot(sub_cx - px, sub_cy - py) < min_sep_px for px, py in placed_centers):
                    placed_centers.append((sub_cx, sub_cy))
                    win_scale = int(best_scale_idx[cand_y, cand_x])
                    if win_scale == 0:
                        crown_radius = max(3, int(pixel_spacing_px * 0.24))
                    elif win_scale == 1:
                        crown_radius = max(5, int(pixel_spacing_px * 0.38))
                    else:
                        crown_radius = max(7, int(pixel_spacing_px * 0.54))

                    detected_boxes.append({
                        "bbox": (sub_cx - crown_radius, sub_cy - crown_radius, sub_cx + crown_radius, sub_cy + crown_radius),
                        "center": (sub_cx, sub_cy),
                        "conf": min(1.0, float(max(annular_drop * 6.0, cand_tophat * 8.0, dog_peak / 20.0))),
                        "crown_radius": crown_radius,
                    })

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

    # 5. Render High-Definition Visual Detection Image with True-Radius Halos & Reticles
    annotated_img = orig_img.copy()
    draw_ann = ImageDraw.Draw(annotated_img)

    # Draw farm boundary in bright turquoise
    if parcel_poly_px and len(parcel_poly_px) >= 3:
        draw_ann.line(parcel_poly_px + [parcel_poly_px[0]], fill=(0, 255, 200), width=3)

    # Draw bounding boxes and crown halos
    badge_interval = max(6, total_trees // 14) if total_trees > 18 else 1

    for t in trees_classified:
        bx1, by1, bx2, by2 = t["bbox"]
        cx, cy = t["center"]
        col = t["color_rgb"]
        cr = t.get("crown_radius", max(4, int((bx2 - bx1) // 2)))

        # 1. Subtle glowing outer halo matching physical tree canopy
        draw_ann.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=col, width=2)
        # Inner accent ring at 0.5x radius
        r_mid = max(2, cr // 2)
        draw_ann.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid], outline=(255, 255, 255), width=1)

        # 2. Precision Center Pinpoint with high-contrast dual outline
        draw_ann.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 0, 0))
        draw_ann.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)

        # 3. Corner reticle brackets for high-precision GIS inspection
        bracket_len = max(2, cr // 3)
        # Top-left
        draw_ann.line([(bx1, by1 + bracket_len), (bx1, by1), (bx1 + bracket_len, by1)], fill=col, width=1)
        # Top-right
        draw_ann.line([(bx2 - bracket_len, by1), (bx2, by1), (bx2, by1 + bracket_len)], fill=col, width=1)
        # Bottom-left
        draw_ann.line([(bx1, by2 - bracket_len), (bx1, by2), (bx1 + bracket_len, by2)], fill=col, width=1)
        # Bottom-right
        draw_ann.line([(bx2 - bracket_len, by2), (bx2, by2), (bx2, by2 - bracket_len)], fill=col, width=1)

        # 4. Serial ID and health indicator badge
        tid = t["tree_id"]
        if tid == 1 or tid % badge_interval == 0 or t["status"] == "damaged":
            if t["status"] == "healthy":
                label_text = f"#{tid}"
            elif t["status"] == "stressed":
                label_text = f"#{tid} 🟡"
            else:
                label_text = f"#{tid} [تالف] 🔴"

            tw = len(label_text) * 8
            draw_ann.rectangle([cx + 4, cy - 10, cx + 4 + tw, cy + 8], fill=(15, 23, 42), outline=col)
            draw_ann.text((cx + 6, cy - 8), label_text, fill=(255, 255, 255))

    # Top Statistical Banner
    banner_h = 38
    draw_ann.rectangle([0, 0, width_px, banner_h], fill=(15, 23, 42))
    header_txt = (
        f"TOTAL TREES: {total_trees} | "
        f"Healthy: {healthy_count} ({healthy_pct}%) [GREEN] | "
        f"Stressed: {stressed_count} ({stressed_pct}%) [YELLOW] | "
        f"Damaged: {damaged_count} ({damaged_pct}%) [RED]"
    )
    draw_ann.text((12, 11), header_txt, fill=(0, 255, 200))

    # Generate NDVI heatmap image
    ndvi_heatmap = generate_ndvi_colormap_image(ndvi_matrix, mask=parcel_mask)

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
        "engine_used": f"YOLOv8 Nano AI ({os.path.basename(active_yolo_path)})" if used_yolo else "OpenCV Precision Spectral Morphology",
        "yolo_model_loaded": used_yolo,
        "yolo_model_path": active_yolo_path if used_yolo else None,
        "trees_classified": trees_classified,
    }
