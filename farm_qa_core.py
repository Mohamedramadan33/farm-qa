#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import sys
import types
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ── RUNTIME PATCH FOR ALBUMENTATIONS COMPATIBILITY ──
def _force_patch_albumentations():
    try:
        import albumentations
        if not hasattr(albumentations, "functional"):
            f_mod = None
            try:
                from albumentations.augmentations import functional as f_mod
            except Exception:
                f_mod = types.ModuleType("albumentations.functional")
            albumentations.functional = f_mod
            sys.modules["albumentations.functional"] = f_mod
    except Exception:
        pass


_force_patch_albumentations()

KOBO_SERVER = "https://kf.kobotoolbox.org"
ASSET_UID = "aX4NJWgge6tooXjfSYXhrq"
KOBO_TOKEN = os.getenv("KOBO_TOKEN", "")

ZOOM = 19
TOL = 0.20
AREA_TOL = 0.10
FEDDAN_M2 = 4200.83

# ── PERIMETER / BOUNDARY AUDIT PARAMETERS ──
# Width of the boundary corridor (in meters) around the polygon edge to count boundary trees
BOUNDARY_BUFFER_M = 8.0
DEFAULT_BOUNDARY_SPACING_M = 3.0  # Default: 1 tree every 3 meters along the farm perimeter

# Standard spacing (in meters) between trees when planted along the farm boundary/perimeter
BOUNDARY_SPACING_M = {
    "جازورين": 2.5, "كازورينا": 2.5, "casuarina": 2.5,
    "كافور": 2.5, "ماهوجني": 3.0, "mahogany": 3.0,
    "فيكس": 3.0, "ficus": 3.0, "توت": 3.5,
    "نخيل": 6.0, "نخل": 6.0, "palm": 6.0,
    "زيتون": 5.0, "olive": 5.0,
    "مانجو": 5.0, "مانجا": 5.0, "mango": 5.0,
    "موالح": 4.0, "citrus": 4.0,
    "موز": 3.0, "banana": 3.0,
}

ABS_TOL = 5
AI_BAND = (0.38, 0.20, 0.12)
VEG_EXG = 0.04
VEG_REVIEW = 25.0
IMAGERY_MAX_AGE_YEARS = 2.0
YOUNG_CROWN_M = 2.0
DUP_IOU = 0.80
CALIBRATION_FILE = "ai_calibration.json"

NDVI_DAYS = 120
NDVI_CLOUD = 20
NDVI_GREEN = 0.30
NDVI_BARE = 0.15
NDVI_OLD_IMG_YEARS = 1.0

FIELD_HINTS = {
    "time":       {"exact": ["_submission_time", "start"], "kw": []},
    "farm_id":    {"exact": ["farm_id"], "kw": ["كود المزرعه", "رقم المزرعه", "farm id", "farm code"]},
    "farm_name":  {"exact": ["farm_name_auto", "farm_name"], "kw": ["اسم المزرعة", "farm name"]},
    "engineer":   {"exact": ["engineer_name"], "kw": ["المهندس", "engineer"]},
    "visit_date": {"exact": ["visit_date"], "kw": ["تاريخ الزيارة"]},
    "audit_type": {"exact": ["audit_type"], "kw": ["نوع الزيارة"]},
    "season":     {"exact": ["__010", "agri_season"], "kw": ["الموسم", "season"]},
    "farm_area":  {"exact": ["farm_area"], "kw": ["مساحه المزرعه", "farm area"]},
    "plot_area":  {"exact": ["plot_area", "calculated_area", "__015"], "kw": ["مساحه القطعه", "مساحه", "area"]},
    "gps":        {"exact": ["gps_outline_of_farm"], "kw": ["gps", "حدود", "مضلع", "احداثيات", "polygon", "boundary"]},
    "tree_count": {"exact": ["tree_number", "العدد", "__091"], "kw": ["عدد الاشجار", "عدد الشجر", "tree count", "العدد", "عدد"]},
    "tree_type":  {"exact": ["tree_type", "النوع", "__016"], "kw": ["نوع الشجر", "نوع الاشجار", "tree type", "crop", "محصول", "النوع", "نوع"]},
    "plot_owner": {"exact": ["plot_owner", "__011"], "kw": ["صاحب الحوض", "المالك"]},
    "owner_phone": {"exact": ["plot_owner_number", "__012"], "kw": ["رقم تليفون", "تليفون", "phone"]},
    "ownership":  {"exact": ["__009"], "kw": ["ملكية", "مملوكة"]},
    "soil_type":  {"exact": ["__047", "soil_type"], "kw": ["نوع التربة", "soil"]},
    "irrig_type": {"exact": ["__048", "irrigation_type"], "kw": ["نوع الري", "طريقة الري"]},
    "irrig_amnt": {"exact": ["irrigation_amount_feddan_year"], "kw": ["كمية الاستهلاك"]},
    "crop_type":  {"exact": ["__030", "crop_type_irrigation"], "kw": ["المحصول", "نوع المحصول"]},
    "animal_type": {"exact": ["_Type", "farm_animals_type"], "kw": ["نوع الحيوان", "حيوانات"]},
    "animal_count": {"exact": ["__077", "cow_numbers"], "kw": ["عدد الحيوانات"]},
    "non_compliance": {"exact": ["non_compliance"], "kw": ["عدم المطابقة", "مخالفة"]},
    "corrective_act": {"exact": ["corrective_measures"], "kw": ["إجراءات تصحيحية", "تصحيح"]},
    "auditor_recom":  {"exact": ["auditors_recommendations"], "kw": ["توصيات لجنة المراجعة", "توصيات"]},
}

log = logging.getLogger("farm_qa")


def normalize_digits(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
    return s.translate(_AR_DIGITS)


def norm_ar(text) -> str:
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("ـ", "")
    for a in "أإآ":
        s = s.replace(a, "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", s).strip()


def present(v) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip() not in ("", "[]", "nan", "None")


def to_number(x) -> float:
    if not present(x):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = normalize_digits(str(x)).replace(",", "").replace(" ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else float("nan")


def flatten(obj, parent: str = "", sep: str = "/") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{parent}{sep}{k}" if parent else str(k), sep))
    elif isinstance(obj, list):
        if obj and all(isinstance(i, (dict, list)) for i in obj):
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{parent}[{i}]", sep))
        else:
            out[parent] = ";".join(map(str, obj))
    else:
        out[parent] = obj
    return out


def build_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([flatten(r) for r in rows])
    conv = lambda v: normalize_digits(v) if isinstance(v, str) else v

    def _fix(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return s.map(conv)
        return s
    return df.apply(_fix)


def fetch_kobo(limit: int = 3000) -> list:
    import requests
    if not KOBO_TOKEN:
        raise RuntimeError("KOBO_TOKEN is not set (add it to Streamlit Secrets or environment variables)")
    headers = {"Authorization": f"Token {KOBO_TOKEN}"}
    url = f"{KOBO_SERVER}/api/v2/assets/{ASSET_UID}/data.json?limit={limit}"
    rows = []
    while url:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 401:
            raise RuntimeError("Invalid or expired Kobo Token (401)")
        r.raise_for_status()
        d = r.json()
        rows += d.get("results", [])
        url = d.get("next")
    return rows


def _leaf(col) -> str:
    return re.sub(r"\[\d+\]", "", str(col)).split("/")[-1].lower()


def _share(series: pd.Series, pred) -> float:
    vals = [v for v in series if present(v)]
    return sum(1 for v in vals if pred(v)) / len(vals) if vals else 0.0


def _kw_score(col, kws) -> int:
    name = norm_ar(_leaf(col).replace("_", " ").replace("-", " "))
    return max((len(norm_ar(k)) for k in kws if norm_ar(k) in name), default=0)


def find_cols(df: pd.DataFrame, field: str) -> list:
    h = FIELD_HINTS[field]
    # FIX: _leaf() بترجع lowercase، فلازم الـ exact يتقارن lowercase برضو (كان "_Type" عمره ما بيطابق)
    exact_names = {x.lower() for x in h["exact"]}
    exact = [c for c in df.columns if _leaf(c) in exact_names]
    if exact:
        return exact
    cols = [c for c in df.columns if not str(c).startswith("_")]
    is_num = lambda v: not math.isnan(to_number(v))
    checks = {
        "gps": lambda s: _share(s, lambda v: parse_points(v) is not None) >= 0.5,
        "tree_count": lambda s: _share(s, is_num) >= 0.5,
        "farm_area": lambda s: _share(s, is_num) >= 0.5,
        "plot_area": lambda s: _share(s, is_num) >= 0.5,
        "tree_type": lambda s: _share(s, lambda v: not is_num(v)) >= 0.5,
    }
    chk = checks.get(field)
    cand = [c for c in cols if chk is None or chk(df[c])]
    if field == "gps":
        cand.sort(key=lambda c: -_kw_score(c, h["kw"]))
        return cand[:1]
    scored = [(c, _kw_score(c, h["kw"])) for c in cand]
    top = max((s for _, s in scored), default=0)
    return [c for c, s in scored if s == top] if top > 0 else []


def map_all(df: pd.DataFrame) -> dict:
    return {f: find_cols(df, f) for f in FIELD_HINTS}


def first_value(row, cols):
    for c in cols:
        if c in row.index and present(row[c]):
            return row[c]
    return None


def engineer_entries(row, count_cols, type_cols) -> list:
    out = []

    def get_prefix(col):
        return str(col).rsplit('/', 1)[0] if '/' in str(col) else 'root_flat'

    types_by_prefix = {get_prefix(c): row[c] for c in type_cols if present(row.get(c))}
    counts_by_prefix = {get_prefix(c): row[c] for c in count_cols if present(row.get(c))}

    for prefix, t_val in types_by_prefix.items():
        if prefix in counts_by_prefix:
            c_val = counts_by_prefix[prefix]
            n = to_number(c_val)
            if not math.isnan(n):
                out.append((str(t_val).strip(), n))

    unmatched_counts = set(counts_by_prefix.keys()) - set(types_by_prefix.keys())
    for prefix in unmatched_counts:
        c_val = counts_by_prefix[prefix]
        n = to_number(c_val)
        if not math.isnan(n):
            out.append(("Unspecified", n))
    return out


_DEC = re.compile(r"-?\d+\.\d+")


def parse_points(raw):
    """Returns list of (lon, lat)."""
    if not present(raw):
        return None
    s = normalize_digits(str(raw))
    pts = []
    for chunk in re.split(r"[;|\n]|P", s):
        nums = _DEC.findall(chunk)
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            if abs(a) > 90 and abs(b) <= 90:
                a, b = b, a
            pts.append((b, a))
    if len(pts) < 3:
        nums = [float(x) for x in _DEC.findall(s)]
        step = 4 if len(nums) % 4 == 0 else 2
        pts = [(nums[i + 1], nums[i]) for i in range(0, len(nums) - 1, step)]
    clean = []
    for p in pts:
        if -180 <= p[0] <= 180 and -90 <= p[1] <= 90 and (not clean or clean[-1] != p):
            clean.append(p)
    return clean if len(clean) >= 3 else None


def points_to_polygon(pts):
    """Automatically untangles self-intersecting ('bowtie/X-shaped') point orders to recover full farm rectangle."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    poly = Polygon(pts)
    if not poly.is_valid:
        unique_pts = list(dict.fromkeys(pts[:-1]))
        if 3 <= len(unique_pts) <= 12:
            cx = sum(p[0] for p in unique_pts) / len(unique_pts)
            cy = sum(p[1] for p in unique_pts) / len(unique_pts)
            sorted_pts = sorted(unique_pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            poly_fixed = Polygon(sorted_pts + [sorted_pts[0]])
            if poly_fixed.is_valid and poly_fixed.area > 0:
                return poly_fixed
        poly = make_valid(poly)
    if poly.geom_type != "Polygon":
        polys = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
        if not polys:
            return None
        poly = max(polys, key=lambda g: g.area)
    return None if poly.is_empty or poly.area == 0 else poly


def parse_polygon(raw):
    pts = parse_points(raw)
    return points_to_polygon(pts) if pts else None


def geom_metrics_utm(poly) -> tuple[float, float]:
    """Returns (area_in_feddan, perimeter_in_meters) using accurate UTM projection."""
    import geopandas as gpd
    gs = gpd.GeoSeries([poly], crs=4326)
    utm_geom = gs.to_crs(gs.estimate_utm_crs()).iloc[0]
    area_fed = float(utm_geom.area) / FEDDAN_M2
    perim_m = float(utm_geom.length)
    return area_fed, perim_m


def area_feddan(poly) -> float:
    return geom_metrics_utm(poly)[0]


def perimeter_meters(poly) -> float:
    return geom_metrics_utm(poly)[1]


def gps_accuracy(raw):
    accs = []
    for chunk in re.split(r"[;|\n]|P", normalize_digits(str(raw))):
        nums = re.findall(r"-?\d+(?:\.\d+)?", chunk)
        idx = next((i for i, n in enumerate(nums) if "." in n), None)
        if idx is not None and len(nums) > idx + 3:
            accs.append(float(nums[idx + 3]))
    return float(np.mean(accs)) if accs else None


def gps_quality(raw, pts) -> list:
    from shapely.geometry import Polygon
    warns = []
    if len(pts) < 4:
        warns.append("❌ Too few points (<4)")

    ring = pts if pts[0] == pts[-1] else pts + [pts[0]]
    p = Polygon(ring)

    if not p.is_valid:
        warns.append("⚠️ Points were recorded in crossed order (Auto-corrected to full boundary)")
    elif p.area > 0:
        compactness = (4 * math.pi * p.area) / (p.length ** 2)
        if compactness < 0.15:
            warns.append(f"❌ Extremely thin shape (Compactness {compactness:.2f}) - Likely a road/path, not a farm")
        elif compactness < 0.30:
            warns.append(f"⚠️ Irregular shape (Compactness {compactness:.2f})")

    acc = gps_accuracy(raw)
    if acc is not None and acc > 15:
        warns.append(f"⚠️ Poor GPS accuracy during capture (Avg {acc:.0f}m)")
    return warns


TILE_PROVIDERS = [
    ("Google Earth", "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"),
    ("ESRI World Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
]


def _px(lon, lat, z):
    n = 256 * (2 ** z)
    return (
        (lon + 180.0) / 360.0 * n,
        (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    )


def _is_valid_satellite_image(img) -> bool:
    arr = np.asarray(img, dtype="float32")
    if arr.size == 0:
        return False
    rg_diff = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
    gb_diff = np.abs(arr[:, :, 1] - arr[:, :, 2]).mean()
    color_spread = rg_diff + gb_diff
    std_val = arr.std()
    if color_spread < 3.0 and std_val < 25.0:
        return False
    return True


def _build_boundary_ring_mask(size: tuple[int, int], ring: list, lat: float, zoom: int, buffer_m: float = BOUNDARY_BUFFER_M):
    """Creates a binary mask covering ONLY the farm boundary corridor (e.g. +-8m around the perimeter line)."""
    from PIL import Image, ImageDraw
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    stroke_px = max(8, int(round((buffer_m * 2.0) / max(mpp, 0.05))))
    b_mask = Image.new("L", size, 0)
    pts = [(float(x), float(y)) for x, y in ring]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    ImageDraw.Draw(b_mask).line(pts, fill=255, width=stroke_px, joint="curve")
    return b_mask, stroke_px


def fetch_farm_image(poly, zoom: int = ZOOM, max_tiles: int = 100):
    """Fetches full farm satellite image + creates both full-farm mask and boundary-corridor mask."""
    import requests
    from PIL import Image, ImageDraw
    minx, miny, maxx, maxy = poly.bounds
    dx = max((maxx - minx) * 0.20, 0.00022)
    dy = max((maxy - miny) * 0.20, 0.00022)
    b_minx, b_miny, b_maxx, b_maxy = minx - dx, miny - dy, maxx + dx, maxy + dy

    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    last_img, last_mask, last_z = None, None, 17
    c_lat = poly.centroid.y

    for z in range(min(zoom, 19), 15, -1):
        x0, y1 = _px(b_minx, b_miny, z)
        x1, y0 = _px(b_maxx, b_maxy, z)
        tx0, tx1, ty0, ty1 = int(x0 // 256), int(x1 // 256), int(y0 // 256), int(y1 // 256)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > max_tiles:
            continue

        coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

        for prov_name, url_tpl in TILE_PROVIDERS:
            def get_tile(xy, url_tpl=url_tpl, z=z):
                tx, ty = xy
                for _ in range(2):
                    try:
                        r = sess.get(url_tpl.format(z=z, x=tx, y=ty), timeout=20)
                        if r.ok and len(r.content) > 800:
                            return xy, Image.open(io.BytesIO(r.content)).convert("RGB")
                    except Exception:
                        pass
                return xy, None

            mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
            ok_all = True
            with ThreadPoolExecutor(8) as ex:
                for (tx, ty), tile in ex.map(get_tile, coords):
                    if tile is None:
                        ok_all = False
                        break
                    mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))

            if not ok_all:
                continue

            ox, oy = tx0 * 256, ty0 * 256
            box = (int(x0 - ox), int(y0 - oy), math.ceil(x1 - ox), math.ceil(y1 - oy))
            img = mosaic.crop(box)
            ring = [(_px(lo, la, z)[0] - ox - box[0], _px(lo, la, z)[1] - oy - box[1]) for lo, la in poly.exterior.coords]
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(ring, fill=255)
            b_mask, stroke_px = _build_boundary_ring_mask(img.size, ring, c_lat, z, BOUNDARY_BUFFER_M)

            img.info["ring"] = ring
            img.info["provider"] = prov_name
            img.info["boundary_mask"] = b_mask
            img.info["boundary_stroke_px"] = stroke_px
            last_img, last_mask, last_z = img, mask, z

            if _is_valid_satellite_image(img):
                return img, mask, z

    if last_img is not None:
        return last_img, last_mask, last_z
    raise RuntimeError("Could not fetch satellite imagery from Google Earth or ESRI")


def render_annotated_satellite(img, mask, boxes=None, upscale: float = 1.0, ring=None):
    """Shows the WHOLE farm clearly in satellite view, highlights the boundary belt, and marks boundary trees."""
    from PIL import Image, ImageDraw
    orig_w, orig_h = img.size
    vis_scale = 1.0
    if min(orig_w, orig_h) < 480:
        vis_scale = 480.0 / max(1, min(orig_w, orig_h))
        new_sz = (int(orig_w * vis_scale), int(orig_h * vis_scale))
        base_rgb = img.resize(new_sz, Image.LANCZOS)
        mask_resized = mask.resize(new_sz, Image.NEAREST)
    else:
        base_rgb = img
        mask_resized = mask

    base = base_rgb.convert("RGBA")
    w, h = base.size

    # Keep the whole farm 100% bright and clear; slightly dim outside background
    arr = np.array(base)
    m = np.array(mask_resized) > 0
    if m.shape == arr.shape[:2]:
        arr[~m, :3] = (arr[~m, :3] * 0.60).astype(np.uint8)
    annotated = Image.fromarray(arr).convert("RGBA")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    poly_ring = ring or img.info.get("ring")
    stroke_px = int(img.info.get("boundary_stroke_px", 16) * vis_scale)
    if poly_ring and len(poly_ring) >= 3:
        pts = [(float(x) * vis_scale, float(y) * vis_scale) for x, y in poly_ring]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        # Subtle translucent yellow buffer zone showing the boundary tree corridor
        draw.line(pts, fill=(255, 235, 0, 45), width=max(12, stroke_px), joint="curve")
        # Crisp farm perimeter line
        line_w = max(3, int(min(w, h) / 150))
        draw.line(pts, fill=(0, 255, 255, 255), width=line_w + 2, joint="curve")
        draw.line(pts, fill=(255, 230, 0, 255), width=line_w, joint="curve")
        r_pt = max(3, line_w)
        for px, py in pts:
            draw.ellipse([px - r_pt, py - r_pt, px + r_pt, py + r_pt], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))

    boundary_trees_drawn = 0
    interior_trees_drawn = 0
    if boxes is not None and not boxes.empty:
        factor = vis_scale / upscale
        for b in boxes.itertuples():
            x1, y1, x2, y2 = float(b.xmin) * factor, float(b.ymin) * factor, float(b.xmax) * factor, float(b.ymax) * factor
            score = getattr(b, "score", 0.5)
            on_b = getattr(b, "on_boundary", True)
            if score >= AI_BAND[1]:
                if on_b:
                    boundary_trees_drawn += 1
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 30, 30, 255), width=2)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    draw.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=(0, 255, 0, 255))
                else:
                    interior_trees_drawn += 1
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 255, 150), width=1)

    prov = img.info.get("provider", "Google Earth")
    banner_h = 28
    draw.rectangle([0, 0, w, banner_h], fill=(15, 20, 30, 220))
    legend_txt = f"{prov} | Boundary Trees (Red): {boundary_trees_drawn}"
    if interior_trees_drawn > 0:
        legend_txt += f" | Interior (Cyan): {interior_trees_drawn}"
    draw.text((10, 7), legend_txt, fill=(255, 255, 255, 255))

    return Image.alpha_composite(annotated, overlay).convert("RGB")


_EMPTY_COUNTS = {"lo": 0, "mid": 0, "hi": 0, "interior_mid": 0}


class TreeCounter:
    def __init__(self, patch_size: int = 400, upscale: float = 1.5):
        self.patch_size, self.upscale = patch_size, upscale
        self._model = None

    def _load(self):
        _force_patch_albumentations()
        from deepforest import main
        m = main.deepforest()
        try:
            m.load_model("weecology/deepforest-tree")
        except Exception:
            m.use_release()
        self._model = m

    def predict(self, img, mask, boundary_only: bool = True):
        """Detects trees across the farm image, focusing counts on the farm boundary corridor."""
        from PIL import Image, ImageEnhance, ImageFilter
        if self._model is None:
            self._load()

        b_mask_orig = img.info.get("boundary_mask")

        if self.upscale != 1.0:
            size = (int(img.width * self.upscale), int(img.height * self.upscale))
            img_work = img.resize(size, Image.LANCZOS)
            mask_work = mask.resize(size, Image.NEAREST)
            b_mask_work = b_mask_orig.resize(size, Image.NEAREST) if b_mask_orig is not None else None
        else:
            img_work, mask_work, b_mask_work = img, mask, b_mask_orig

        img_work = ImageEnhance.Contrast(img_work).enhance(1.15)
        arr = np.array(img_work)
        arr_input = arr.astype("float32")

        try:
            boxes = self._model.predict_tile(image=arr_input, patch_size=self.patch_size,
                                             patch_overlap=0.35, iou_threshold=0.15)
        except Exception:
            boxes = self._model.predict_image(image=arr_input)

        if boxes is None or len(boxes) == 0:
            return dict(_EMPTY_COUNTS), pd.DataFrame()

        boxes = boxes[boxes["score"] >= min(AI_BAND)].copy()
        if boxes.empty:
            return dict(_EMPTY_COUNTS), pd.DataFrame()

        # Dilate farm mask so boundary trees slightly outside the GPS line are captured
        farm_dilated = np.array(mask_work.filter(ImageFilter.MaxFilter(15))) > 0
        if b_mask_work is not None:
            b_zone = np.array(b_mask_work) > 0
        else:
            b_zone = farm_dilated

        cx = ((boxes.xmin + boxes.xmax) / 2).astype(int).clip(0, farm_dilated.shape[1] - 1)
        cy = ((boxes.ymin + boxes.ymax) / 2).astype(int).clip(0, farm_dilated.shape[0] - 1)

        in_farm_or_edge = farm_dilated[cy.values, cx.values] | b_zone[cy.values, cx.values]
        boxes = boxes[in_farm_or_edge].copy()
        if boxes.empty:
            return dict(_EMPTY_COUNTS), pd.DataFrame()

        cx = ((boxes.xmin + boxes.xmax) / 2).astype(int).clip(0, b_zone.shape[1] - 1)
        cy = ((boxes.ymin + boxes.ymax) / 2).astype(int).clip(0, b_zone.shape[0] - 1)
        boxes["on_boundary"] = b_zone[cy.values, cx.values].astype(bool)

        # FIX: بنبني mask boolean بدل ما نعمل DataFrame من list of Series (كان بيبوّظ الـ dtypes)
        # وبنعمل clip للإحداثيات عشان القيم السالبة متعملش slicing غلط.
        H, W = arr.shape[:2]
        keep = []
        for box in boxes.itertuples():
            x1, y1 = max(0, int(box.xmin)), max(0, int(box.ymin))
            x2, y2 = min(W, int(box.xmax)), min(H, int(box.ymax))
            box_pixels = arr[y1:y2, x1:x2].astype("float32")
            if box_pixels.size == 0:
                keep.append(False)
                continue
            brightness = box_pixels.mean()
            s = box_pixels.sum(axis=2) + 1e-6
            exg = (2 * box_pixels[:, :, 1] - box_pixels[:, :, 0] - box_pixels[:, :, 2]) / s
            green_ratio = float((exg > VEG_EXG).mean())
            keep.append(bool(green_ratio > 0.05 or (brightness < 145 and box.score >= 0.25)))

        boxes_df = boxes[np.array(keep, dtype=bool)].copy()
        if boxes_df.empty:
            return dict(_EMPTY_COUNTS), pd.DataFrame()

        target_df = boxes_df[boxes_df["on_boundary"]] if boundary_only else boxes_df
        interior_df = boxes_df[~boxes_df["on_boundary"]]

        counts = {k: int((target_df["score"] >= t).sum()) for k, t in zip(("lo", "mid", "hi"), AI_BAND)}
        counts["interior_mid"] = int((interior_df["score"] >= AI_BAND[1]).sum())
        return counts, boxes_df


_CALIB: dict = {}
_USE_CALIB = True
_DATE_CACHE: dict = {}


def load_calibration(path: str = CALIBRATION_FILE) -> dict:
    _CALIB.clear()
    try:
        with open(path, encoding="utf-8") as f:
            _CALIB.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Could not read calibration file: %s", e)
    return _CALIB


def calib_factor(species) -> float:
    s = norm_ar(species) if present(species) else ""
    for k, v in _CALIB.get("species", {}).items():
        if s and norm_ar(k) in s:
            return float(v)
    return float(_CALIB.get("default", 1.0))


def vegetation_pct(img, mask) -> float:
    a = np.asarray(img, dtype="float32")
    m = np.asarray(mask) > 0
    if m.sum() == 0:
        return float("nan")
    s = a.sum(axis=2) + 1e-6
    exg = (2 * a[:, :, 1] - a[:, :, 0] - a[:, :, 2]) / s
    return float(((exg > VEG_EXG) & m).sum() / m.sum() * 100)


def crown_diameter_m(boxes, lat: float, zoom: int, upscale: float = 1.0) -> float:
    if boxes is None or len(boxes) == 0:
        return float("nan")
    b = boxes[boxes["score"] >= AI_BAND[1]]
    if len(b) == 0:
        return float("nan")
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    side_px = np.sqrt((b.xmax - b.xmin).values * (b.ymax - b.ymin).values)
    return float(np.median(side_px) / upscale * mpp)


def imagery_date(lon: float, lat: float):
    key = (round(lon, 3), round(lat, 3))
    if key in _DATE_CACHE:
        return _DATE_CACHE[key]
    out = None
    try:
        import requests
        r = requests.get(
            "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/0/query",
            params={"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects", "outFields": "SRC_DATE2",
                    "returnGeometry": "false", "f": "json"}, timeout=20)
        feats = r.json().get("features", [])
        dates = [f["attributes"].get("SRC_DATE2") for f in feats if f["attributes"].get("SRC_DATE2")]
        if dates:
            out = pd.to_datetime(max(dates), unit="ms", utc=True)
    except Exception as e:
        log.debug("imagery_date failed: %s", e)
    _DATE_CACHE[key] = out
    return out


def recent_ndvi(poly, days: int = NDVI_DAYS) -> float:
    try:
        import planetary_computer as pc
        from pystac_client import Client
        from odc.stac import load
        import rioxarray  # noqa: F401
        from shapely.geometry import mapping

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
        items = cat.search(collections=["sentinel-2-l2a"], bbox=poly.bounds,
                           datetime=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                           query={"eo:cloud_cover": {"lt": NDVI_CLOUD}}).item_collection()
        if len(items) == 0:
            return float("nan")

        ds = load(items, bands=["B04", "B08"], bbox=poly.bounds,
                  crs="EPSG:4326", resolution=0.0001,
                  groupby="solar_day").astype("float32")
        valid = (ds.B04 > 0) & (ds.B08 > 0)
        red = ((ds.B04 - 1000) / 10000).where(valid)
        nir = ((ds.B08 - 1000) / 10000).where(valid)
        ndvi = ((nir - red) / (nir + red)).median("time")

        ndvi = ndvi.rio.write_crs("EPSG:4326").rio.clip(
            [mapping(poly)], crs="EPSG:4326", drop=True, all_touched=True)
        v = float(ndvi.mean(skipna=True))
        return v if not math.isnan(v) else float("nan")
    except Exception as e:
        log.warning("recent_ndvi failed: %s", e)
        return float("nan")


def _close(a, b, tol):
    return abs(a - b) <= max(tol * max(a, b), ABS_TOL)


def spacing_for_boundary(species) -> tuple[float, bool]:
    """Returns standard tree-to-tree spacing (in meters) along the farm perimeter."""
    s = norm_ar(species) if present(species) else ""
    for k, v in BOUNDARY_SPACING_M.items():
        if s and norm_ar(k) in s:
            return float(v), True
    return float(DEFAULT_BOUNDARY_SPACING_M), False


def verdict(E, A, F, tol: float = TOL, band=None, veg_pct=None, img_age=None, crown_m=None) -> str:
    if math.isnan(E) or E <= 0:
        return "⚠️ No valid boundary tree count recorded by the engineer."
    if math.isnan(F):
        return "⚠️ [Manual Review]: Cannot calculate perimeter tree forecast (missing or invalid GPS)."
    if math.isnan(A):
        if E > F * (1 + tol) + ABS_TOL:
            return "❌ [Over Perimeter Capacity]: Reported count exceeds boundary capacity (Unverified by satellite)."
        return "⚠️ [Unverified by Satellite]: Count fits farm perimeter, click 'Run satellite AI' to verify."

    lo, hi = band if band else (A, A)
    in_band = lambda x: lo * (1 - tol) - ABS_TOL <= x <= hi * (1 + tol) + ABS_TOL
    over_capacity = E > F * (1 + tol) + ABS_TOL
    ai_misses_E = E > hi * (1 + tol) + ABS_TOL
    ai_sees_more = lo > E * (1 + tol) + ABS_TOL
    old = img_age is not None and img_age > IMAGERY_MAX_AGE_YEARS

    if in_band(E) and (in_band(F) or _close(E, F, tol) or E <= F * (1 + tol) + ABS_TOL):
        return "✅ [Excellent Boundary Match]: Engineer count aligns with Satellite AI on the farm perimeter."
    if over_capacity and ai_misses_E:
        if old:
            return f"⚠️ [Review - Outdated Imagery]: Count exceeds perimeter capacity & not seen by AI, but imagery is old (~{img_age:.1f} yrs)."
        if veg_pct is not None and veg_pct >= VEG_REVIEW:
            return f"⚠️ [Review]: Count exceeds perimeter capacity, but boundary has dense vegetation ({veg_pct:.0f}%)."
        return "❌ [Major Boundary Discrepancy]: Reported count exceeds perimeter capacity with no satellite evidence."
    if ai_sees_more:
        return "❌ [Boundary Under-counting]: Satellite AI detected more trees along the farm boundary than reported."
    if not over_capacity and ai_misses_E:
        why = ""
        if crown_m is not None and not math.isnan(crown_m) and crown_m < YOUNG_CROWN_M:
            why = f" Avg crown diameter ~{crown_m:.1f}m (young saplings)."
        return "⚠️ [Perimeter Capacity Match]: Farm perimeter fits the reported count, but AI detects fewer mature crowns (likely young saplings or partial border planting)." + why
    return "⚠️ [Moderate Variance]: Noticeable difference between Engineer, Perimeter Forecast, and AI; manual review recommended."


def apply_ndvi(verdict_text: str, ndvi, veg, img_age) -> str:
    if ndvi is None or (isinstance(ndvi, float) and math.isnan(ndvi)):
        return verdict_text
    age_ok = img_age is not None and not (isinstance(img_age, float) and math.isnan(img_age))
    old_img = age_ok and img_age > NDVI_OLD_IMG_YEARS
    weak_veg = veg is None or (isinstance(veg, float) and math.isnan(veg)) or veg < 5
    if old_img and weak_veg and ndvi >= NDVI_GREEN:
        return (f"⚠️ [Review - Recent Greening]: Recent Sentinel-2 NDVI is {ndvi:.2f} "
                f"while older satellite imagery shows low vegetation. Previous verdict may be outdated.\n   "
                + verdict_text)
    if ndvi < NDVI_BARE:
        return verdict_text + f"  (Recent NDVI {ndvi:.2f}: Bare soil currently, supporting imagery result)"
    return verdict_text


def confidence(rec: dict) -> str:
    s = 100
    age = rec.get("img_age")
    if age is not None and not (isinstance(age, float) and math.isnan(age)) and age > IMAGERY_MAX_AGE_YEARS:
        s -= 30
    if rec.get("gps_warnings"):
        s -= 15
    if str(rec.get("area_check", "")).startswith("❌"):
        s -= 20
    if rec.get("density_note"):
        s -= 10
    if rec.get("ai_error"):
        s -= 40
    return "🟢 High" if s >= 80 else "🟡 Medium" if s >= 55 else "🔴 Low"


def _shapely_major() -> int:
    try:
        import shapely
        return int(str(shapely.__version__).split(".")[0])
    except Exception:
        return 1


def cross_checks(recs: list) -> None:
    """Flags near-duplicate polygons. Uses an STRtree (fast) instead of an O(n²) loop."""
    for r in recs:
        r.setdefault("flags", [])
    items = [r for r in recs if r.get("_poly") is not None]
    if len(items) < 2:
        return
    polys = [r["_poly"] for r in items]

    if _shapely_major() >= 2:
        from shapely.strtree import STRtree
        tree = STRtree(polys)

        def candidates(i):
            return sorted(int(j) for j in tree.query(polys[i]) if int(j) > i)
    else:
        def candidates(i):
            return range(i + 1, len(items))

    for i in range(len(items)):
        for j in candidates(i):
            ra, rb = items[i], items[j]
            if str(ra["farm_id"]) == str(rb["farm_id"]):
                continue
            try:
                inter = polys[i].intersection(polys[j]).area
                uni = polys[i].union(polys[j]).area
            except Exception:
                continue
            if uni > 0 and inter / uni >= DUP_IOU:
                ra["flags"].append(f"🚩 Polygon overlaps with Farm {rb['farm_id']} ({inter / uni:.0%})")
                rb["flags"].append(f"🚩 Polygon overlaps with Farm {ra['farm_id']} ({inter / uni:.0%})")


def audit_row(row, cols, counter=None, use_ai=True, debug_dir=None, use_ndvi=True) -> dict:
    nan = float("nan")
    rec = {
        "farm_id":          first_value(row, cols["farm_id"]) or "N/A",
        "farm_name":        first_value(row, cols["farm_name"]) or "—",
        "engineer":         first_value(row, cols["engineer"]) or "N/A",
        "submission_time":  first_value(row, cols["time"]) or "—",
        "visit_date":       first_value(row, cols["visit_date"]) or "—",
        "audit_type":       first_value(row, cols["audit_type"]) or "—",
        "season":           first_value(row, cols["season"]) or "—",
        "plot_owner":       first_value(row, cols["plot_owner"]) or "—",
        "owner_phone":      first_value(row, cols["owner_phone"]) or "—",
        "ownership_type":   first_value(row, cols["ownership"]) or "—",
        "soil_type":        first_value(row, cols["soil_type"]) or "—",
        "irrig_type":       first_value(row, cols["irrig_type"]) or "—",
        "irrig_amnt":       first_value(row, cols["irrig_amnt"]) or "—",
        "crop_type":        first_value(row, cols["crop_type"]) or "—",
        "animal_type":      first_value(row, cols["animal_type"]) or "—",
        "animal_count":     first_value(row, cols["animal_count"]) or "—",
        "non_compliance":   first_value(row, cols["non_compliance"]) or "—",
        "corrective_act":   first_value(row, cols["corrective_act"]) or "—",
        "auditor_recom":    first_value(row, cols["auditor_recom"]) or "—",
        "tree_details":     ""
    }

    gps_raw = first_value(row, cols["gps"])
    entered = to_number(first_value(row, cols["farm_area"]))
    if math.isnan(entered) or entered == 0:
        entered = to_number(first_value(row, cols["plot_area"]))

    missing = []
    if rec["engineer"] == "N/A":
        missing.append("Engineer Name")
    if math.isnan(entered) or entered == 0:
        missing.append("Area")
    if gps_raw is None:
        missing.append("GPS Coordinates")
    rec["missing"] = ", ".join(missing)

    poly = parse_polygon(gps_raw) if gps_raw is not None else None
    rec["_poly"] = poly
    rec["gps_warnings"] = ""
    if poly is not None:
        try:
            rec["gps_warnings"] = "; ".join(gps_quality(gps_raw, parse_points(gps_raw)))
        except Exception as e:
            log.debug("gps_quality failed: %s", e)

    gps_area, perim_m = nan, nan
    if poly is not None:
        try:
            raw_area, raw_perim = geom_metrics_utm(poly)
            gps_area = round(raw_area, 3)
            perim_m = round(raw_perim, 1)
        except Exception as e:
            log.warning("geom_metrics_utm failed for %s: %s", rec["farm_id"], e)

    rec["area_entered"], rec["area_gps"], rec["perimeter_m"] = entered, gps_area, perim_m
    if math.isnan(gps_area) or gps_area <= 0:
        rec["area_check"] = "❌ Calculation Failed (Invalid/Missing GPS)"
    elif math.isnan(entered) or entered <= 0:
        rec["area_check"] = "⚠️ No entered area to compare"
    else:
        ok = abs(entered - gps_area) / gps_area <= AREA_TOL
        rec["area_check"] = "✅ Matched" if ok else "❌ High Discrepancy"

    entries = engineer_entries(row, cols["tree_count"], cols["tree_type"])
    E = sum(n for _, n in entries) if entries else nan
    rec["engineer_count"] = E

    # ── PERIMETER-BASED TREE FORECAST CALCULATION ──
    notes, weighted_spacing = [], 0.0
    for sp, n in entries:
        sp_m, known = spacing_for_boundary(sp)
        weighted_spacing += sp_m * n
        if not known:
            notes.append(f"Default boundary spacing {DEFAULT_BOUNDARY_SPACING_M}m used for '{sp}'")

    tot = sum(n for _, n in entries)
    avg_spacing_m = (weighted_spacing / tot) if tot > 0 else DEFAULT_BOUNDARY_SPACING_M
    F = (perim_m / avg_spacing_m) if (not math.isnan(perim_m) and perim_m > 0 and avg_spacing_m > 0) else nan
    rec["forecast"] = round(F) if not math.isnan(F) else nan
    rec["boundary_spacing_m"] = round(avg_spacing_m, 2)

    by_type = {}
    for sp, n in entries:
        sp_name = str(sp).strip()
        by_type[sp_name] = by_type.get(sp_name, 0) + n

    rec["species"] = max(by_type, key=by_type.get) if by_type else "Unspecified"
    rec["tree_details"] = " | ".join([f"{k} ({int(v)})" for k, v in by_type.items()]) if by_type else "None"
    rec["density_note"] = "; ".join(sorted(set(notes)))

    A = A_lo = A_hi = veg = crown = img_age = ndvi = nan
    ai_err = ""
    if use_ai and poly is not None:
        img = mask = None
        boxes = pd.DataFrame()
        active_counter = counter or TreeCounter()
        try:
            img, mask, z = fetch_farm_image(poly)
            rec["zoom_used"] = z
            b_mask = img.info.get("boundary_mask", mask)
            # Calculate vegetation cover along the boundary corridor
            veg = round(vegetation_pct(img, b_mask), 1)
            c = poly.centroid
            idate = imagery_date(c.x, c.y)
            if idate is not None:
                rec["imagery_date"] = idate.strftime("%Y-%m")
                ref = pd.to_datetime(rec["submission_time"], errors="coerce", utc=True)
                if pd.isna(ref):
                    ref = pd.Timestamp.now(tz="UTC")
                img_age = (ref - idate).days / 365.25

            counts, boxes = active_counter.predict(img, mask, boundary_only=True)
            f = calib_factor(rec["species"]) if _USE_CALIB else 1.0
            A_lo, A, A_hi = (round(counts[k] * f) for k in ("lo", "mid", "hi"))
            rec["ai_raw"], rec["ai_factor"] = counts["mid"], f
            rec["ai_interior_trees"] = counts.get("interior_mid", 0)
            crown = crown_diameter_m(boxes[boxes["on_boundary"]] if not boxes.empty and "on_boundary" in boxes else boxes,
                                     c.y, z, getattr(active_counter, "upscale", 1.5))
        except Exception as e:
            ai_err = str(e)
            log.warning("AI step failed for %s: %s", rec["farm_id"], e)
        finally:
            if debug_dir and img is not None and mask is not None:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    s = getattr(active_counter, "upscale", 1.5)
                    vis_img = render_annotated_satellite(img, mask, boxes=boxes, upscale=s, ring=img.info.get("ring"))
                    vis_img.save(os.path.join(debug_dir, f"{str(rec['farm_id']).replace('/', '_')}.jpg"), quality=92)
                except Exception as save_err:
                    log.warning("Failed saving annotated satellite image: %s", save_err)

    if use_ai and use_ndvi and poly is not None:
        ndvi = recent_ndvi(poly)

    rec.update(ai_count=A, ai_low=A_lo, ai_high=A_hi, veg_pct=veg, crown_m=crown,
               img_age=img_age, ndvi_recent=ndvi, ai_error=ai_err)
    band = (A_lo, A_hi) if not math.isnan(A) else None
    rec["verdict"] = verdict(E, A, F, band=band, veg_pct=veg if not math.isnan(veg) else None,
                             img_age=img_age if not math.isnan(img_age) else None,
                             crown_m=crown if not math.isnan(crown) else None)
    rec["verdict"] = apply_ndvi(rec["verdict"], ndvi if not math.isnan(ndvi) else None,
                                veg if not math.isnan(veg) else None,
                                img_age if not math.isnan(img_age) else None)
    if ai_err:
        rec["verdict"] += f"  (Satellite AI Error: {ai_err})"
    rec["confidence"] = confidence(rec)
    return rec


def audit_df(df: pd.DataFrame, use_ai=True, debug_dir=None, verbose=False, calibrated=True, use_ndvi=True) -> pd.DataFrame:
    global _USE_CALIB
    _USE_CALIB = calibrated
    if calibrated and not _CALIB:
        load_calibration()
    cols = map_all(df)
    counter = TreeCounter() if use_ai else None
    recs = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        try:
            rec = audit_row(row, cols, counter, use_ai, debug_dir, use_ndvi)
        except Exception as e:
            rec = {"farm_id": first_value(row, cols["farm_id"]) or i, "engineer": "—",
                   "verdict": f"❌ Processing Error: {e}"}
        recs.append(rec)
    cross_checks(recs)
    for rec in recs:
        rec["flags"] = "; ".join(rec.get("flags", []))
    return pd.DataFrame(recs)


def export_report(res: pd.DataFrame, path: str = "farm_qa_report.xlsx") -> str:
    display_cols = [c for c in res.columns if not str(c).startswith("_")]
    res[display_cols].to_excel(path, index=False)
    return path


def filter_df(df, mode="all", date=None, farm=None, cols=None) -> pd.DataFrame:
    """`cols` اختياري: لو اتبعت (من الكاش) مش بنعمل map_all تاني (غالي على آلاف الصفوف)."""
    if df is None or df.empty:
        return df
    cols = cols or map_all(df)
    if mode == "date" and date:
        tc = [c for c in cols["time"] if _leaf(c) == "_submission_time"] or cols["time"]
        if tc:
            df = df[df[tc[0]].astype(str).str.startswith(date.strip())]
    elif mode == "farm" and farm:
        mask = pd.Series(False, index=df.index)
        for c in cols["farm_id"]:
            mask |= df[c].astype(str).str.strip().str.lower() == farm.strip().lower()
        df = df[mask]
    return df


def load_data(mode="all", date=None, farm=None) -> pd.DataFrame:
    df = build_dataframe(fetch_kobo())
    if df.empty:
        return df
    return filter_df(df, mode, date, farm)
