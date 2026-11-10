# -*- coding: utf-8 -*-
"""
pgs_rules.py — PGS (Participatory Guarantee System) Rules Engine for Egyptian Organic/Regenerative Farms.
Validates administrative compliance, field measurements, required attachments, and agronomic consistency.
"""

import re
import datetime
from typing import Dict, List, Any, Optional, Tuple
from shapely.geometry import Point, Polygon

AR_DIGITS_MAP = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

STRIP_RE = re.compile(r'[\u0640\u064B-\u065F\u200B-\u200F\u202A-\u202E\u2066-\u2069\ufeff]')


def norm_pgs(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    for ar_d, en_d in AR_DIGITS_MAP.items():
        s = s.replace(ar_d, en_d)
    s = STRIP_RE.sub('', s)
    s = s.replace('\u00a0', ' ')
    s = re.sub(r'[أإآ]', 'ا', s)
    s = s.replace('ى', 'ي').replace('ة', 'ه')
    return " ".join(s.split()).strip()


def to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    clean = norm_pgs(val).replace('٫', '.').replace(',', '.')
    m = re.search(r'-?\d+(?:\.\d+)?', clean)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def is_valid_egypt_phone(phone_str: Any) -> bool:
    """
    Validates Egyptian phone numbers:
    - Mobile: 11 digits starting with 010, 011, 012, 015
    - Landline: 9-10 digits starting with governorate prefix (02, 03, 040, 045, etc.)
    """
    clean = re.sub(r'\D', '', norm_pgs(phone_str))
    if clean.startswith('0020'):
        clean = '0' + clean[4:]
    elif clean.startswith('20') and len(clean) in (11, 12):
        clean = '0' + clean[2:]

    if len(clean) == 10 and any(clean.startswith(p) for p in ['10', '11', '12', '15']):
        clean = '0' + clean

    if len(clean) == 11 and any(clean.startswith(p) for p in ['010', '011', '012', '015']):
        return True

    landline_prefixes = [
        '02', '03', '040', '045', '047', '048', '050', '055', '057',
        '062', '064', '065', '066', '068', '069', '082', '084', '086',
        '088', '092', '093', '095', '096', '097'
    ]
    if len(clean) in (9, 10) and any(clean.startswith(p) for p in landline_prefixes):
        return True

    return False


def parse_kobo_date(date_str: Any) -> Optional[datetime.date]:
    if not date_str:
        return None
    s = str(date_str).strip()
    m_iso = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m_iso:
        try:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            return None
    m_dmy = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m_dmy:
        try:
            return datetime.date(int(m_dmy.group(3)), int(m_dmy.group(2)), int(m_dmy.group(1)))
        except ValueError:
            return None
    return None


def evaluate_pgs_rules(
    farm_record: Dict[str, Any],
    flat_row: Dict[str, Any],
    polygon_coords: Optional[List[Tuple[float, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates PGS rules and returns a list of rule findings:
    { "rule": str, "passed": bool, "severity": "error"|"warning"|"info", "message": str }
    """
    findings: List[Dict[str, Any]] = []

    # 1. Phone number check
    phone = farm_record.get("owner_phone") or flat_row.get("farm_details/rep_phone") or flat_row.get("rep_phone")
    if phone:
        if is_valid_egypt_phone(phone):
            findings.append({
                "rule": "رقم الهاتف",
                "passed": True,
                "severity": "info",
                "message": f"رقم الهاتف ({phone}) مطابق لمعايير الأرقام المصرية."
            })
        else:
            findings.append({
                "rule": "رقم الهاتف",
                "passed": False,
                "severity": "warning",
                "message": f"رقم الهاتف المسجل ({phone}) غير مطابق لأرقام المحمول (010/011/012/015) أو الأرضي في مصر."
            })
    else:
        findings.append({
            "rule": "رقم الهاتف",
            "passed": False,
            "severity": "warning",
            "message": "لم يتم تسجيل رقم هاتف لممثل المزرعة أو صاحب الحوض."
        })

    # 2. Visit Date check
    visit_d = parse_kobo_date(farm_record.get("visit_date"))
    sub_d = parse_kobo_date(farm_record.get("submission_time"))
    today = datetime.date.today()

    if visit_d:
        if visit_d > today:
            findings.append({
                "rule": "تاريخ الزيارة",
                "passed": False,
                "severity": "error",
                "message": f"تاريخ الزيارة ({visit_d}) لا يمكن أن يكون في المستقبل."
            })
        elif sub_d and visit_d > sub_d:
            findings.append({
                "rule": "تاريخ الزيارة",
                "passed": False,
                "severity": "warning",
                "message": f"تاريخ الزيارة ({visit_d}) مسجل بعد تاريخ إرسال الاستمارة ({sub_d})."
            })
        else:
            findings.append({
                "rule": "تاريخ الزيارة",
                "passed": True,
                "severity": "info",
                "message": f"تاريخ الزيارة سليم ومنطقي ({visit_d})."
            })
    else:
        findings.append({
            "rule": "تاريخ الزيارة",
            "passed": False,
            "severity": "warning",
            "message": "تاريخ الزيارة غير محدد أو غير قابل للقراءة."
        })

    # 3. Soil Samples Count & Polygon Containment
    soil_pts: List[Tuple[float, float]] = []
    for k, v in flat_row.items():
        if "sample_point" in k.lower() or "soil_samples" in k.lower():
            if isinstance(v, str) and len(v.split()) >= 2:
                parts = v.split()
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                    soil_pts.append((lat, lon))
                except ValueError:
                    pass

    poly_geom = None
    if polygon_coords and len(polygon_coords) >= 3:
        try:
            poly_geom = Polygon([(p[0], p[1]) for p in polygon_coords])  # (lon, lat)
            if not poly_geom.is_valid:
                poly_geom = poly_geom.buffer(0)
        except Exception:
            poly_geom = None

    if len(soil_pts) >= 4:
        inside_count = 0
        if poly_geom:
            buf_poly = poly_geom.buffer(0.0001)  # ~10 meters buffer tolerance
            for lat, lon in soil_pts:
                pt = Point(lon, lat)
                if buf_poly.contains(pt):
                    inside_count += 1
            if inside_count == len(soil_pts):
                findings.append({
                    "rule": "عينات التربة",
                    "passed": True,
                    "severity": "info",
                    "message": f"تم أخذ {len(soil_pts)} عينات تربة (مستوفي الحد الأدنى 4) وجميعها تقع داخل حدود المزرعة."
                })
            else:
                findings.append({
                    "rule": "عينات التربة",
                    "passed": False,
                    "severity": "error",
                    "message": f"يوجد {len(soil_pts) - inside_count} من أصل {len(soil_pts)} عينات تربة تقع خارج حدود مضلع المزرعة!"
                })
        else:
            findings.append({
                "rule": "عينات التربة",
                "passed": True,
                "severity": "info",
                "message": f"تم تسجيل {len(soil_pts)} نقاط عينات تربة (مستوفي الحد الأدنى 4)."
            })
    elif len(soil_pts) > 0:
        findings.append({
            "rule": "عينات التربة",
            "passed": False,
            "severity": "error",
            "message": f"تم تسجيل {len(soil_pts)} عينات تربة فقط، بينما يشترط نظام PGS أخذ 4 عينات على الأقل لتغطية القطعة."
        })
    else:
        findings.append({
            "rule": "عينات التربة",
            "passed": False,
            "severity": "warning",
            "message": "لم يتم رصد إحداثيات نقاط أخذ عينات التربة في الاستمارة."
        })

    # 4. Mandatory Documents Attachments
    doc_keys = [
        ("صورة الحيازة / عقد الملكية", ["tenure", "حيازة", "حيازه", "ملكية"]),
        ("بيان التنوع الحيوي", ["biodiversity", "تنوع حيوي", "التنوع الحيوي"]),
        ("إقرار عدم استخدام كائنات معدلة وراثياً (Non-GMO)", ["gmo", "معدل وراثيا", "تقاوي"]),
        ("سجل المزرعة وعمليات الري والتسميد", ["ledger", "سجل المزرعة", "سجل"]),
    ]

    for doc_name, keywords in doc_keys:
        found = False
        for k, v in flat_row.items():
            k_low = norm_pgs(k).lower()
            if any(kw in k_low for kw in keywords):
                if v and str(v).strip():
                    found = True
                    break
        if found:
            findings.append({
                "rule": f"مستند: {doc_name}",
                "passed": True,
                "severity": "info",
                "message": f"مرفق وثيقة {doc_name} بنجاح."
            })
        else:
            findings.append({
                "rule": f"مستند: {doc_name}",
                "passed": False,
                "severity": "warning",
                "message": f"مستند {doc_name} غير مرفق بالاستمارة."
            })

    # 5. Farm Area Consistency (Entered Area vs GPS Polygon Area)
    area_ent = farm_record.get("area_entered")
    area_gps = farm_record.get("area_gps")
    if area_ent and area_gps:
        diff_pct = abs(area_ent - area_gps) / max(area_ent, 0.01) * 100
        if diff_pct <= 15:
            findings.append({
                "rule": "مطابقة المساحة",
                "passed": True,
                "severity": "info",
                "message": f"المساحة المسجلة ({area_ent:.2f} فدان) متطابقة مع مساحة مضلع GPS ({area_gps:.2f} فدان) بنسبة تفاوت {diff_pct:.1f}%."
            })
        else:
            findings.append({
                "rule": "مطابقة المساحة",
                "passed": False,
                "severity": "warning",
                "message": f"تفاوت بين المساحة المسجلة ({area_ent:.2f} فدان) ومساحة GPS ({area_gps:.2f} فدان) بنسبة {diff_pct:.1f}%."
            })

    # 6. Tree Health & Windbreak Condition (NDVI Healthy vs Damaged Ratio)
    tree_damaged_pct = farm_record.get("tree_damaged_pct")
    tree_healthy_pct = farm_record.get("tree_healthy_pct", 100.0 - (tree_damaged_pct or 0.0))
    if tree_damaged_pct is not None:
        if tree_damaged_pct > 15.0:
            findings.append({
                "rule": "الحالة الصحية للأشجار والمصدات (NDVI)",
                "passed": False,
                "severity": "warning",
                "message": f"نسبة الأشجار التالفة أو الميتة مرتفعة ({tree_damaged_pct:.1f}%) وتتجاوز الحد المسموح به في PGS (15%). يلزم الترقيع لضمان عزل المزرعة ومصدات الرياح.",
            })
        else:
            findings.append({
                "rule": "الحالة الصحية للأشجار والمصدات (NDVI)",
                "passed": True,
                "severity": "info",
                "message": f"الحالة الصحية لأشجار المزرعة ممتازة ({tree_healthy_pct:.1f}% سليمة خضراء) ونسبة التلف منخفضة ({tree_damaged_pct:.1f}%).",
            })

    # 7. Vegetative Canopy Green Coverage (نسبة الغطاء النباتي الفضائي)
    veg_cov_pct = farm_record.get("parcel_veg_coverage_pct") or farm_record.get("veg_pct")
    if veg_cov_pct is not None:
        if float(veg_cov_pct) < 25.0:
            findings.append({
                "rule": "كثافة الغطاء النباتي للمزرعة",
                "passed": False,
                "severity": "warning",
                "message": f"نسبة الغطاء النباتي الأخضر المرصودة بالقمر الصناعي منخفضة ({veg_cov_pct:.1f}%) — يلزم فحص الحزام العازل ومصدات الحماية.",
            })
        else:
            findings.append({
                "rule": "كثافة الغطاء النباتي للمزرعة",
                "passed": True,
                "severity": "info",
                "message": f"كثافة الغطاء النباتي الأخضر للمزرعة مستوفية لمعايير الحماية والتنوع الحيوي ({veg_cov_pct:.1f}%).",
            })

    return findings
