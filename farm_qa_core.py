# -*- coding: utf-8 -*-
"""
farm_qa_core.py — EBDA Farm QA Core Engine:
Geospatial UTM metrics, GPS bowtie repair, fast satellite canopy & tree crown analysis,
and Kobo survey flattening & record extraction.
"""

import math
import re
import datetime
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# Constants
FEDDAN_M2 = 4200.83
DUP_IOU_THRESHOLD = 0.80

BOUNDARY_SPACING_M = {
    "جازورين": 2.5, "كازورينا": 2.5, "casuarina": 2.5,
    "كافور": 2.5, "ماهوجني": 3.0, "mahogany": 3.0,
    "فيكس": 3.0, "ficus": 3.0, "توت": 3.5,
    "نخيل": 6.0, "نخل": 6.0, "palm": 6.0,
    "زيتون": 5.0, "olive": 5.0, "مانجو": 5.0,
    "مانجا": 5.0, "mango": 5.0, "موالح": 4.0,
    "citrus": 4.0, "موز": 3.0, "banana": 3.0,
}

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(s: Any) -> str:
    if s is None:
        return ""
    return str(s).translate(AR_DIGITS)


def to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = normalize_digits(val).replace('٫', '.').replace(',', '.')
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def flatten_dict(d: Any, parent: str = '', sep: str = '/') -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k == '_attachments':
                continue
            new_key = f"{parent}{sep}{k}" if parent else str(k)
            items.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(d, list):
        if d and all(isinstance(x, dict) for x in d):
            for i, x in enumerate(d):
                items.update(flatten_dict(x, f"{parent}[{i}]", sep=sep))
        else:
            items[parent] = "; ".join(str(x) for x in d)
    else:
        items[parent] = normalize_digits(d) if isinstance(d, str) else d
    return items


KOBO_QUESTION_LABELS: Dict[str, Tuple[str, str]] = {
    # Basic & Farm Info
    'farm_id': ('كود المزرعة', 'البيانات الأساسية'),
    'at_house/farm_id': ('كود المزرعة', 'البيانات الأساسية'),
    'farm_name': ('اسم المزرعة', 'البيانات الأساسية'),
    'at_house/farm_name': ('اسم المزرعة', 'البيانات الأساسية'),
    'engineer_name': ('اسم المهندس المراجع', 'بيانات المهندس والزيارة'),
    'engineer_data/engineer_name': ('اسم المهندس المراجع', 'بيانات المهندس والزيارة'),
    '__002': ('المحافظة', 'بيانات المهندس والزيارة'),
    'engineer_data/__002': ('المحافظة', 'بيانات المهندس والزيارة'),
    'governorate': ('المحافظة', 'بيانات المهندس والزيارة'),
    'visit_date': ('تاريخ الزيارة الميدانية', 'بيانات المهندس والزيارة'),
    'engineer_data/visit_date': ('تاريخ الزيارة الميدانية', 'بيانات المهندس والزيارة'),
    'audit_type': ('نوع الزيارة والتفتيش', 'بيانات المهندس والزيارة'),
    'engineer_data/audit_type': ('نوع الزيارة والتفتيش', 'بيانات المهندس والزيارة'),
    '__010': ('الموسم الزراعي', 'بيانات المهندس والزيارة'),
    'engineer_data/__010': ('الموسم الزراعي', 'بيانات المهندس والزيارة'),
    'season': ('الموسم الزراعي', 'بيانات المهندس والزيارة'),
    'rep_name': ('اسم المزارع / ممثل المزرعة', 'بيانات المزرعة والمزارع'),
    'farm_details/rep_name': ('اسم المزارع / ممثل المزرعة', 'بيانات المزرعة والمزارع'),
    'rep_phone': ('رقم تليفون ممثل المزرعة', 'بيانات المزرعة والمزارع'),
    'farm_details/rep_phone': ('رقم تليفون ممثل المزرعة', 'بيانات المزرعة والمزارع'),
    '__009': ('نوع الملكية (فردية / مجموعة مزارعين)', 'بيانات المزرعة والمزارع'),
    'farm_details/__009': ('نوع الملكية (فردية / مجموعة مزارعين)', 'بيانات المزرعة والمزارع'),

    # Area & Boundaries
    'farm_area': ('إجمالي المساحة بالفدان', 'المساحة والحدود'),
    'farm_details/farm_area': ('إجمالي المساحة بالفدان', 'المساحة والحدود'),
    'gps_outline_of_farm': ('إحداثيات حدود المزرعة GPS', 'المساحة والحدود'),
    'farm_details/gps_outline_of_farm': ('إحداثيات حدود المزرعة GPS', 'المساحة والحدود'),

    # Soil & Irrigation
    '__047': ('نوع التربة', 'التربة والري'),
    'farm_details/__047': ('نوع التربة', 'التربة والري'),
    'soil_type': ('نوع التربة', 'التربة والري'),
    '__048': ('طريقة ونظام الري', 'التربة والري'),
    'farm_details/__048': ('طريقة ونظام الري', 'التربة والري'),
    'irrig_type': ('طريقة ونظام الري', 'التربة والري'),
    'irrigation_amount_feddan_year': ('كمية الاستهلاك المائي (م3/فدان/سنة)', 'التربة والري'),
    'sample_point': ('إحداثيات نقاط أخذ عينات التربة GPS', 'التربة والري'),
    'soil_samples/sample_point': ('إحداثيات نقاط أخذ عينات التربة GPS', 'التربة والري'),
    'soil_photo': ('صورة سحب عينة التربة الميدانية', 'التربة والري'),
    'soil_samples/soil_photo': ('صورة سحب عينة التربة الميدانية', 'التربة والري'),

    # Trees & Boundary
    'tree_type': ('نوع شجر الحدود ومصدات الرياح', 'أشجار الحدود ومصدات الرياح'),
    'boundary_trees/tree_type': ('نوع شجر الحدود ومصدات الرياح', 'أشجار الحدود ومصدات الرياح'),
    'tree_number': ('عدد أشجار الحدود المسجل بالمعاينة', 'أشجار الحدود ومصدات الرياح'),
    'boundary_trees/tree_number': ('عدد أشجار الحدود المسجل بالمعاينة', 'أشجار الحدود ومصدات الرياح'),
    'boundary_photo': ('صورة أشجار الحدود ومصدات الرياح', 'أشجار الحدود ومصدات الرياح'),
    'boundary_trees/boundary_photo': ('صورة أشجار الحدود ومصدات الرياح', 'أشجار الحدود ومصدات الرياح'),
    'trees_photo': ('صورة أشجار الحدود ومصدات الرياح', 'أشجار الحدود ومصدات الرياح'),
    'windbreak_photo': ('صورة مصدات الرياح المحيطة', 'أشجار الحدود ومصدات الرياح'),

    # Compost & Organic Fertilizers (SEKEM / EBDA / PGS)
    'compost_photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'group_compost/compost_photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'compost/compost_photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'compost/photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'compost_pile_photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'group_compost/photo': ('صورة كومة الكومبوست والتسميد العضوي', 'الكومبوست والتسميد العضوي'),
    'compost_source': ('مصدر مواد وخامات الكومبوست', 'الكومبوست والتسميد العضوي'),
    'group_compost/compost_source': ('مصدر مواد وخامات الكومبوست', 'الكومبوست والتسميد العضوي'),
    'compost_turnings': ('عدد مرات تقليب كومة الكومبوست', 'الكومبوست والتسميد العضوي'),
    'group_compost/compost_turnings': ('عدد مرات تقليب كومة الكومبوست', 'الكومبوست والتسميد العضوي'),
    'fertilizer_photo': ('صورة التسميد والأسمدة العضوية', 'الكومبوست والتسميد العضوي'),
    'organic_fertilizer': ('نوع ومصدر السماد العضوي المستخدم', 'الكومبوست والتسميد العضوي'),

    # Water, Well & Solar
    'well_photo': ('صورة بئر مياه الري / محطة الرفع', 'التربة والري'),
    'water_source_photo': ('صورة مصدر مياه الري والتحليل', 'التربة والري'),
    'irrigation_photo': ('صورة شبكة ومحابس الري', 'التربة والري'),
    'group_irrigation/well_photo': ('صورة بئر مياه الري / محطة الرفع', 'التربة والري'),
    'solar_photo': ('صورة محطة وألواح الطاقة الشمسية', 'الطاقة والبنية التحتية'),
    'solar_panels_photo': ('صورة محطة وألواح الطاقة الشمسية', 'الطاقة والبنية التحتية'),
    'group_solar/solar_photo': ('صورة محطة وألواح الطاقة الشمسية', 'الطاقة والبنية التحتية'),

    # Storage & Warehouse
    'storage_photo': ('صورة مخزن المزرعة وخلوه من الكيماويات', 'المستودعات والمخازن'),
    'warehouse_photo': ('صورة مخزن المزرعة وخلوه من الكيماويات', 'المستودعات والمخازن'),
    'group_storage/storage_photo': ('صورة مخزن المزرعة وخلوه من الكيماويات', 'المستودعات والمخازن'),
    'pesticide_storage_photo': ('صورة فحص المخزن للتأكد من خلوه من المبيدات', 'المستودعات والمخازن'),

    # Animals & Livestock
    'animal_photo': ('صورة حظيرة الماشية والإنتاج الحيواني', 'الإنتاج الحيواني'),
    'barn_photo': ('صورة حظيرة الماشية والإنتاج الحيواني', 'الإنتاج الحيواني'),
    'livestock_photo': ('صورة الماشية والحيوانات بالمزرعة', 'الإنتاج الحيواني'),

    # Crops & Farm Land
    'crop_photo': ('صورة المحصول المنزرع وحالة النباتات', 'المحصول والعمليات الزراعية'),
    'field_photo': ('صورة الحقل والأرض المنزرعة', 'المحصول والعمليات الزراعية'),
    'farm_overview_photo': ('صورة عامة للمزرعة وبيئتها', 'المحصول والعمليات الزراعية'),

    # Buffer Zone & Surroundings
    'buffer_photo': ('صورة المنطقة العازلة ومزارع الجيران', 'المنطقة العازلة ومخاطر التلوث'),
    'buffer_zone_photo': ('صورة المنطقة العازلة ومزارع الجيران', 'المنطقة العازلة ومخاطر التلوث'),
    'neighbor_photo': ('صورة أنشطة المزارع المجاورة', 'المنطقة العازلة ومخاطر التلوث'),
    'contamination_risk_photo': ('صورة تقييم مخاطر التلوث من الجوار', 'المنطقة العازلة ومخاطر التلوث'),

    # Bio-Dynamic Preparations
    'bd_prep_photo': ('صورة مستحضرات الزراعة الحيوية (BD Preps)', 'الزراعة الحيوية والبيوديناميك'),
    'preparation_photo': ('صورة تجهيز مستحضرات البيوديناميك', 'الزراعة الحيوية والبيوديناميك'),

    # Legal & Documentation
    'tenure_photo': ('صورة الحيازة الزراعية / عقد الملكية أو الإيجار', 'المستندات والحيازات'),
    'docs/tenure_photo': ('صورة الحيازة الزراعية / عقد الملكية أو الإيجار', 'المستندات والحيازات'),
    'contract_photo': ('صورة عقد الملكية أو الإيجار', 'المستندات والحيازات'),
    'id_photo': ('صورة بطاقة الرقم القومي للمزارع', 'المستندات والحيازات'),
    'biodiversity_doc': ('بيان خطة التنوع الحيوي والمحميات', 'المستندات والحيازات'),
    'docs/biodiversity_doc': ('بيان خطة التنوع الحيوي والمحميات', 'المستندات والحيازات'),
    'biodiversity_photo': ('صورة التنوع الحيوي والمناطق الطبيعية', 'المستندات والحيازات'),
    'gmo_doc': ('إقرار عدم استخدام تقاوي معدلة وراثياً (GMO)', 'المستندات والحيازات'),
    'docs/gmo_doc': ('إقرار عدم استخدام تقاوي معدلة وراثياً (GMO)', 'المستندات والحيازات'),
    'ledger_photo': ('صورة سجل المزرعة وعمليات التسميد والري', 'المستندات والحيازات'),
    'docs/ledger_photo': ('صورة سجل المزرعة وعمليات التسميد والري', 'المستندات والحيازات'),

    # Audit Committee
    'committee_photo': ('صورة أعضاء لجنة المعاينة والتفتيش', 'لجنة المراجعة والتفتيش'),
    'member_photo_1': ('صورة عضو لجنة المراجعة 1', 'لجنة المراجعة والتفتيش'),
    'member_photo_2': ('صورة عضو لجنة المراجعة 2', 'لجنة المراجعة والتفتيش'),
    'member_photo_3': ('صورة عضو لجنة المراجعة 3', 'لجنة المراجعة والتفتيش'),

    # Audit Notes & Decisions
    'non_compliance': ('حالات عدم المطابقة المرصودة', 'ملاحظات وتوصيات التفتيش'),
    'audit_notes/non_compliance': ('حالات عدم المطابقة المرصودة', 'ملاحظات وتوصيات التفتيش'),
    'corrective_measures': ('الإجراءات التصحيحية المتفق عليها', 'ملاحظات وتوصيات التفتيش'),
    'audit_notes/corrective_measures': ('الإجراءات التصحيحية المتفق عليها', 'ملاحظات وتوصيات التفتيش'),
    'auditors_recommendations': ('توصيات وقرارات لجنة المراجعة', 'ملاحظات وتوصيات التفتيش'),
    'audit_notes/auditors_recommendations': ('توصيات وقرارات لجنة المراجعة', 'ملاحظات وتوصيات التفتيش'),
}


def clean_kobo_semantic_title(key: str) -> Tuple[str, str]:
    """
    Translates any Kobo question field key into a clear Arabic title and section.
    """
    if not key:
        return "صورة مرفقة بالمعاينة", "الصور والمرفقات الميدانية"

    leaf = key.split("/")[-1].lower()
    full = key.lower()

    if key in KOBO_QUESTION_LABELS:
        return KOBO_QUESTION_LABELS[key]
    if leaf in KOBO_QUESTION_LABELS:
        return KOBO_QUESTION_LABELS[leaf]

    # Compost & Fertilizers
    if any(w in full for w in ["compost", "كومبوست", "سماد", "fertiliz", "manure"]):
        return "صورة كومة الكومبوست والتسميد العضوي", "الكومبوست والتسميد العضوي"

    # Boundary Trees & Windbreaks
    if any(w in full for w in ["boundary", "tree", "windbreak", "شجر", "مصد"]):
        return "صورة أشجار الحدود ومصدات الرياح", "أشجار الحدود ومصدات الرياح"

    # Soil & Samples
    if any(w in full for w in ["soil", "sample", "تربة", "عينة"]):
        return "صورة سحب عينة التربة الميدانية", "التربة والري"

    # Irrigation & Water Sources
    if any(w in full for w in ["well", "water", "irrig", "pump", "بئر", "ري", "طلمبة", "مياه"]):
        return "صورة مصدر مياه الري والبئر / الطلمبة", "التربة والري"

    # Solar & Energy
    if any(w in full for w in ["solar", "energy", "panel", "شمسية", "طاقة"]):
        return "صورة محطة وألواح الطاقة الشمسية", "الطاقة والبنية التحتية"

    # Tenure & Ownership
    if any(w in full for w in ["tenure", "contract", "ownership", "حيازة", "عقد", "ملكية", "deed"]):
        return "صورة الحيازة الزراعية / عقد الملكية والإيجار", "المستندات والحيازات"

    # Biodiversity
    if any(w in full for w in ["biodiversity", "تنوع", "wildlife"]):
        return "صورة التنوع الحيوي والمحميات الطبيعية", "التنوع الحيوي"

    # GMO / Seeds
    if any(w in full for w in ["gmo", "وراثي", "تقاوي", "seed"]):
        return "صورة فحص وخلو التقاوي من التعديل الوراثي (GMO)", "المستندات والحيازات"

    # Ledger & Records
    if any(w in full for w in ["ledger", "record", "سجل", "دفتر"]):
        return "صورة سجل المزرعة والعمليات الزراعية", "المستندات والحيازات"

    # Storage & Warehouse
    if any(w in full for w in ["storage", "store", "warehouse", "مخزن", "مستودع"]):
        return "صورة مخزن المزرعة وخلوه من الكيماويات", "المستودعات والمخازن"

    # Pesticide check
    if any(w in full for w in ["pesticide", "chemical", "مبيد"]):
        return "صورة خلو المزرعة من المبيدات الكيماوية", "المستودعات والمخازن"

    # Animals & Barns
    if any(w in full for w in ["animal", "barn", "livestock", "cow", "sheep", "ماشية", "حظيرة"]):
        return "صورة حظيرة الماشية والإنتاج الحيواني", "الإنتاج الحيواني"

    # Crops & Fields
    if any(w in full for w in ["crop", "plant", "field", "محصول", "حقل", "نبات"]):
        return "صورة المحصول المنزرع وحالة النباتات", "المحصول والعمليات الزراعية"

    # Audit Committee
    if any(w in full for w in ["committee", "auditor", "member", "لجنة", "معاينة", "مفتش"]):
        return "صورة أعضاء لجنة المعاينة والتفتيش", "لجنة المراجعة والتفتيش"

    # Farmer / Representative
    if any(w in full for w in ["farmer", "rep", "owner", "مزارع", "مالك"]):
        return "صورة المزارع / ممثل المزرعة", "البيانات الأساسية"

    # Buffer & Neighbors
    if any(w in full for w in ["buffer", "neighbor", "عازل", "جار", "جوار"]):
        return "صورة المنطقة العازلة ومزارع الجوار", "المنطقة العازلة ومخاطر التلوث"

    # Bio-Dynamic
    if any(w in full for w in ["bd_prep", "preparation", "500", "501", "بيوديناميك"]):
        return "صورة مستحضرات الزراعة الحيوية (BD Preps)", "الزراعة الحيوية والبيوديناميك"

    # Clean fallback title
    readable = leaf.replace("_", " ").title()
    return f"صورة: {readable}", "الصور والمرفقات الميدانية"


def resolve_kobo_photo_title(
    fn: str,
    flat_dict: Dict[str, Any],
    schema_labels: Optional[Dict[str, str]] = None,
    photo_idx: int = 1
) -> Dict[str, str]:
    """
    Resolves photo metadata into clear, human-readable Arabic question title and section.
    Never returns an obscure filepath, UUID or hash!
    """
    clean_fn = fn.split("/")[-1].strip() if "/" in fn else fn.strip()
    matched_key = ""

    # 1. Search flat dictionary for matching field
    for k, v in flat_dict.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        v_base = v_str.split("/")[-1].strip() if "/" in v_str else v_str

        if (
            v_str == fn
            or v_base == clean_fn
            or v_str == clean_fn
            or (len(clean_fn) >= 6 and clean_fn in v_str)
            or (len(v_base) >= 6 and v_base in fn)
            or fn.endswith(v_str)
        ):
            matched_key = k
            break

    # 2. Derive human Arabic title
    title = ""
    section = "الصور والمرفقات الميدانية"

    if matched_key:
        leaf_key = matched_key.split("/")[-1]
        # Check custom schema labels fetched from Kobo asset API
        if schema_labels:
            title = schema_labels.get(matched_key) or schema_labels.get(leaf_key) or ""

        # Check standard dictionary & semantic rules
        if not title:
            title, section = clean_kobo_semantic_title(matched_key)
    else:
        # Keyword deduction from filename if available
        fn_low = fn.lower()
        if any(w in fn_low for w in ["compost", "كومبوست", "سماد"]):
            title = "صورة كومة الكومبوست والتسميد العضوي"
            section = "الكومبوست والتسميد العضوي"
        elif any(w in fn_low for w in ["boundary", "tree", "شجر"]):
            title = "صورة أشجار الحدود ومصدات الرياح"
            section = "أشجار الحدود ومصدات الرياح"
        elif any(w in fn_low for w in ["soil", "تربة"]):
            title = "صورة سحب عينة التربة"
            section = "التربة والري"
        elif any(w in fn_low for w in ["well", "water", "irrig", "بئر", "ري"]):
            title = "صورة مصدر مياه الري والبئر"
            section = "التربة والري"
        elif any(w in fn_low for w in ["solar", "شمسية"]):
            title = "صورة محطة وألواح الطاقة الشمسية"
            section = "الطاقة والبنية التحتية"
        elif any(w in fn_low for w in ["tenure", "حيازة"]):
            title = "صورة الحيازة الزراعية / عقد الملكية"
            section = "المستندات والحيازات"
        else:
            title = f"صورة المعاينة الميدانية ({photo_idx})"
            section = "الصور والمرفقات الميدانية"

    return {
        "title": title,
        "section": section,
        "clean_filename": clean_fn,
        "field_key": matched_key,
    }


def normalize_kobo_validation_status(val_raw: Any) -> Dict[str, str]:
    """
    Normalizes Kobo validation status into standard codes and labels:
    - code: 'approved', 'not_approved', 'on_hold', 'unvalidated'
    - label_en: 'Approved', 'Not approved', 'On hold', '-'
    - label_ar: 'معتمد (Approved)', 'مرفوض (Not approved)', 'معلق (On hold)', 'غير محدد (-)'
    - color: '#10b981', '#ef4444', '#f59e0b', '#94a3b8'
    """
    if not val_raw:
        return {
            "code": "unvalidated",
            "label_en": "-",
            "label_ar": "غير محدد (-)",
            "color": "#94a3b8",
            "uid": "",
        }

    if isinstance(val_raw, dict):
        uid = str(val_raw.get("uid") or "").lower()
        lbl = str(val_raw.get("label") or "").lower()
    else:
        uid = str(val_raw).lower()
        lbl = str(val_raw).lower()

    if "approved" in uid or "approved" in lbl:
        if "not" in uid or "not" in lbl or "غير" in lbl or "رفض" in lbl:
            return {
                "code": "not_approved",
                "label_en": "Not approved",
                "label_ar": "مرفوض (Not approved)",
                "color": "#ef4444",
                "uid": "validation_status_not_approved",
            }
        else:
            return {
                "code": "approved",
                "label_en": "Approved",
                "label_ar": "معتمد (Approved)",
                "color": "#10b981",
                "uid": "validation_status_approved",
            }
    elif "hold" in uid or "hold" in lbl or "معلق" in lbl or "انتظار" in lbl:
        return {
            "code": "on_hold",
            "label_en": "On hold",
            "label_ar": "معلق (On hold)",
            "color": "#f59e0b",
            "uid": "validation_status_on_hold",
            "description": "معلق للمراجعة الإضافية",
        }
    else:
        return {
            "code": "unvalidated",
            "label_en": "-",
            "label_ar": "غير محدد (-)",
            "color": "#94a3b8",
            "uid": "",
        }


def extract_farm_record(
    raw_sub: Any,
    idx: int = 0,
    schema_labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Intelligently extracts standard farm audit record from any live Kobo submission JSON
    regardless of whether questions are nested inside groups or at root.
    """
    if not isinstance(raw_sub, dict):
        raw_sub = {}
    flat = flatten_dict(raw_sub)
    sub_id = str(raw_sub.get("_id", f"sub_{idx+1}"))
    sub_time = str(raw_sub.get("_submission_time", ""))

    def find_val(candidates: List[str], default: Any = "") -> Any:
        # 1. Exact match on leaf or full path
        for c in candidates:
            c_low = c.lower()
            for k, v in flat.items():
                if v is not None and str(v).strip():
                    leaf = k.split('/')[-1].lower()
                    if leaf == c_low or k.lower() == c_low:
                        return v
        # 2. Substring match
        for c in candidates:
            c_low = c.lower()
            for k, v in flat.items():
                if v is not None and str(v).strip():
                    if c_low in k.lower():
                        return v
        return default

    farm_id = str(find_val(["farm_id", "كود المزرعة", "كود المزرعه", "farm_code", "كود", "__001"], f"FARM-{sub_id}")).strip()
    farm_name = str(find_val(["farm_name_auto", "farm_name", "اسم المزرعة", "اسم المزرعه"], f"مزرعة {farm_id}")).strip()
    engineer = str(find_val(["engineer_name", "اسم المهندس", "المهندس", "engineer"], "غير مسجل")).strip()
    farmer = str(find_val(["rep_name", "اسم المزارع", "اسم صاحب المزرعة", "صاحب الحوض", "farmer"], "غير مسجل")).strip()
    phone = str(find_val(["rep_phone", "plot_owner_number", "رقم تليفون", "تليفون", "phone"], "")).strip()
    gov = str(find_val(["__002", "governorate", "المحافظة", "المنطقة", "region"], "غير محدد")).strip()
    visit_date = str(find_val(["visit_date", "تاريخ الزيارة", "تاريخ المعاينة"], sub_time[:10] if sub_time else "")).strip()
    audit_type = str(find_val(["audit_type", "نوع الزيارة", "نوع التفتيش"], "تجديد سنوي PGS")).strip()
    season = str(find_val(["__010", "agri_season", "الموسم"], "صيفي 2024")).strip()
    area_val = to_float(find_val(["farm_area", "calculated_area", "مساحة المزرعة", "إجمالي المساحة", "plot_area"]))
    soil_type = str(find_val(["__047", "soil_type", "نوع التربة"], "طميية")).strip()
    irrig_type = str(find_val(["__048", "irrigation_type", "طريقة الري", "نوع الري"], "تنقيط")).strip()
    tree_type = str(find_val(["tree_type", "__016", "نوع الشجر", "نوع الاشجار"], "جازورين")).strip()
    tree_number = int(to_float(find_val(["tree_number", "__091", "عدد الاشجار", "العدد"])) or 0)

    # GPS polygon extraction
    gps_poly = ""
    for k, v in flat.items():
        if isinstance(v, str) and ("gps" in k.lower() or "outline" in k.lower() or "boundary" in k.lower() or "حدود" in k):
            parts = v.strip().replace(";", "\n").split("\n")
            if len(parts) >= 3:
                gps_poly = v.strip()
                break
    if not gps_poly:
        for k, v in flat.items():
            if isinstance(v, str) and (";" in v or "\n" in v):
                tokens = v.replace(";", "\n").split("\n")
                valid_pts = 0
                for t in tokens:
                    parts = t.strip().split()
                    if len(parts) >= 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            valid_pts += 1
                        except ValueError:
                            pass
                if valid_pts >= 3:
                    gps_poly = v.strip()
                    break

    # Soil points
    soil_points_list = []
    for k, v in flat.items():
        if isinstance(v, str) and ("sample_point" in k.lower() or "soil" in k.lower() or "عينة" in k):
            parts = v.strip().split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    soil_points_list.append(v.strip())
                except ValueError:
                    pass

    # Extract all photos with intelligent human Arabic titles
    attachments = raw_sub.get("_attachments", [])
    photos: List[Dict[str, Any]] = []
    seen_clean_files = set()

    # 1. From _attachments
    for att in attachments:
        fn = str(att.get("filename", "")).strip()
        url = att.get("download_url") or att.get("download_medium_url") or att.get("download_small_url") or ""
        mime = str(att.get("mimetype", "")).lower()

        if url and ("image" in mime or any(fn.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".heic"])):
            info = resolve_kobo_photo_title(fn, flat, schema_labels, photo_idx=len(photos) + 1)
            seen_clean_files.add(info["clean_filename"])
            photos.append({
                "filename": fn,
                "clean_filename": info["clean_filename"],
                "url": url,
                "title": info["title"],
                "section": info["section"],
                "field_key": info["field_key"],
                "mimetype": mime,
            })

    # 2. Check flat for any photo fields not present in _attachments
    for k, v in flat.items():
        if isinstance(v, str) and any(v.lower().strip().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".heic"]):
            v_clean = v.split("/")[-1].strip()
            if v_clean not in seen_clean_files:
                seen_clean_files.add(v_clean)
                info = resolve_kobo_photo_title(v, flat, schema_labels, photo_idx=len(photos) + 1)
                photos.append({
                    "filename": v,
                    "clean_filename": v_clean,
                    "url": v if v.startswith("http") else "",
                    "title": info["title"],
                    "section": info["section"],
                    "field_key": k,
                    "mimetype": "image/jpeg",
                })

    val_raw = raw_sub.get("_validation_status", {}) if isinstance(raw_sub, dict) else {}
    val_norm = normalize_kobo_validation_status(val_raw)

    return {
        "_id": raw_sub.get("_id", idx + 1),
        "_submission_time": sub_time,
        "_validation_status": val_raw,
        "validation_info": val_norm,
        "validation_code": val_norm["code"],
        "validation_label": val_norm["label_en"],
        "validation_color": val_norm["color"],
        "farm_id": farm_id,
        "farm_name": farm_name,
        "engineer_name": engineer,
        "rep_name": farmer,
        "rep_phone": phone,
        "governorate": gov,
        "visit_date": visit_date,
        "submission_time": sub_time[:10] if sub_time else "",
        "audit_type": audit_type,
        "season": season,
        "farm_area": area_val or 5.0,
        "soil_type": soil_type,
        "irrig_type": irrig_type,
        "tree_type": tree_type,
        "tree_number": tree_number,
        "gps_polygon": gps_poly,
        "soil_points": "; ".join(soil_points_list),
        "attachments": attachments,
        "photos": photos,
        "flat": flat,
    }


def get_form_qa_items(
    flat_dict: Dict[str, Any],
    photos_list: Optional[List[Dict[str, Any]]] = None,
    schema_labels: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns an ordered list of all question & answer items from the Kobo submission,
    linking any question that has an attached photo to its image URL.
    """
    qa_items: List[Dict[str, Any]] = []

    # Map photos by full filename and clean basename
    photos_by_name: Dict[str, Dict[str, Any]] = {}
    for p in (photos_list or []):
        fn = p.get("filename", "")
        clean_fn = p.get("clean_filename", "")
        if fn:
            photos_by_name[fn] = p
        if clean_fn:
            photos_by_name[clean_fn] = p

    # Ignore meta keys
    ignore_prefixes = ("_", "meta/", "formhub/", "__version__", "deviceid", "instanceID")

    for k, v in flat_dict.items():
        if any(k.startswith(p) for p in ignore_prefixes):
            continue
        if v is None or str(v).strip() == "":
            continue

        leaf = k.split('/')[-1]

        # 1. Check custom schema labels
        q_title = ""
        q_section = ""
        if schema_labels:
            q_title = schema_labels.get(k) or schema_labels.get(leaf) or ""

        # 2. Check semantic dictionary
        if not q_title:
            q_title, q_section = clean_kobo_semantic_title(k)

        str_val = str(v).strip()
        str_val_clean = str_val.split("/")[-1].strip() if "/" in str_val else str_val

        # Match photo
        matching_photo = (
            photos_by_name.get(str_val)
            or photos_by_name.get(str_val_clean)
        )
        if not matching_photo and any(str_val.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".heic"]):
            for p in (photos_list or []):
                p_fn = p.get("filename", "")
                p_c = p.get("clean_filename", "")
                if str_val_clean in p_fn or p_c in str_val:
                    matching_photo = p
                    break

        photo_url = matching_photo.get("url") if matching_photo else None

        qa_items.append({
            "field_key": k,
            "title": q_title,
            "section": q_section,
            "answer": str_val,
            "photo_url": photo_url,
            "photo_filename": matching_photo.get("clean_filename") if matching_photo else (str_val_clean if photo_url else None),
        })

    return qa_items




def parse_polygon_coords(poly_str: Any) -> Tuple[Optional[List[Tuple[float, float]]], bool]:
    """
    Parses Kobo polygon text: 'lat1 lon1 alt acc; lat2 lon2 alt acc; ...'
    Returns (coords_lon_lat, is_repaired)
    """
    if not poly_str or not isinstance(poly_str, str):
        return None, False

    clean_str = poly_str.replace(";", "\n").replace(",", " ")
    points_lon_lat: List[Tuple[float, float]] = []

    for line in clean_str.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                # Filter out invalid 0,0 coordinates
                if abs(lat) > 1.0 and abs(lon) > 1.0:
                    points_lon_lat.append((lon, lat))
            except ValueError:
                continue

    if len(points_lon_lat) < 3:
        return None, False

    # Ensure closure
    if points_lon_lat[0] != points_lon_lat[-1]:
        points_lon_lat.append(points_lon_lat[0])

    is_repaired = False
    try:
        raw_poly = Polygon(points_lon_lat)
        if not raw_poly.is_valid:
            valid_geom = make_valid(raw_poly)
            is_repaired = True
            if isinstance(valid_geom, Polygon):
                points_lon_lat = list(valid_geom.exterior.coords)
            elif isinstance(valid_geom, MultiPolygon):
                largest = max(valid_geom.geoms, key=lambda p: p.area)
                points_lon_lat = list(largest.exterior.coords)
    except Exception:
        pass

    return points_lon_lat, is_repaired


def calculate_utm_metrics(coords_lon_lat: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Projects WGS84 coordinates to UTM (Zone 35/36 North for Egypt)
    Computes area in square meters, Feddan (4200.83 m²), and perimeter in meters.
    """
    if not coords_lon_lat or len(coords_lon_lat) < 3:
        return {
            "area_m2": 0.0,
            "area_feddan": 0.0,
            "perimeter_m": 0.0,
            "utm_zone": None,
            "poly_utm": None,
        }

    mean_lon = sum(p[0] for p in coords_lon_lat) / len(coords_lon_lat)
    utm_zone = int((mean_lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)

    utm_coords = [transformer.transform(p[0], p[1]) for p in coords_lon_lat]
    poly_utm = Polygon(utm_coords)
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)

    area_m2 = poly_utm.area
    perimeter_m = poly_utm.length
    area_feddan = area_m2 / FEDDAN_M2

    return {
        "area_m2": area_m2,
        "area_feddan": area_feddan,
        "perimeter_m": perimeter_m,
        "utm_zone": utm_zone,
        "poly_utm": poly_utm,
    }


def evaluate_gps_quality(coords_lon_lat: List[Tuple[float, float]], poly_utm: Optional[Polygon]) -> Dict[str, Any]:
    n_pts = len(coords_lon_lat) if coords_lon_lat else 0
    if not poly_utm or poly_utm.area <= 0:
        return {"score": 0, "rating": "ضعيف جداً", "compactness": 0.0, "pts": n_pts}

    area = poly_utm.area
    perimeter = poly_utm.length
    compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    if n_pts >= 8 and compactness >= 0.3:
        rating = "ممتاز"
        score = 95
    elif n_pts >= 5 and compactness >= 0.15:
        rating = "جيد جداً"
        score = 80
    elif n_pts >= 4:
        rating = "مقبول"
        score = 65
    else:
        rating = "ضعيف"
        score = 40

    return {
        "score": score,
        "rating": rating,
        "compactness": round(compactness, 3),
        "pts": n_pts,
    }


# -------------------------------------------------------------
# Fast Satellite Imagery & Tree Canopy Analyzer
# -------------------------------------------------------------

def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_satellite_tile(x: int, y: int, z: int) -> Optional[Image.Image]:
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=8
        )
        if resp.ok and len(resp.content) > 500:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass

    try:
        gurl = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        resp = requests.get(gurl, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.ok and len(resp.content) > 500:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass

    return None


def fetch_satellite_capture_date(lon: float, lat: float) -> Optional[str]:
    try:
        url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/0/query"
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "SRC_DATE2",
            "returnGeometry": "false",
            "f": "json",
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.ok:
            data = resp.json()
            features = data.get("features", [])
            dates = [f["attributes"]["SRC_DATE2"] for f in features if f.get("attributes", {}).get("SRC_DATE2")]
            if dates:
                latest_ms = max(dates)
                dt = datetime.datetime.fromtimestamp(latest_ms / 1000.0)
                return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def analyze_farm_satellite_canopy(
    coords_lon_lat: Optional[List[Tuple[float, float]]],
    perimeter_m: float,
    tree_type_str: str = "",
    zoom: int = 18,
    custom_spacing_m: Optional[float] = None,
    sensitivity: float = 0.5,
) -> Dict[str, Any]:
    """
    Fast canopy & boundary tree analysis with optical crown peak snapping:
    - Never blocks the CPU or hangs Streamlit Cloud!
    - Stitches satellite tiles covering the farm boundary.
    - Computes Excess Green Index along 8m corridor and locks onto crown peaks.
    - Supports interactive sensitivity tuning and species spacing.
    """
    if custom_spacing_m is not None and custom_spacing_m > 0:
        spacing_m = float(custom_spacing_m)
    else:
        spacing_m = 3.0
        for k, v in BOUNDARY_SPACING_M.items():
            if k in tree_type_str:
                spacing_m = v
                break

    forecast = int(round(perimeter_m / spacing_m)) if perimeter_m > 0 else 0


    if not coords_lon_lat or len(coords_lon_lat) < 3:
        return {
            "ai_count": forecast,
            "ai_low": max(0, int(round(forecast * 0.85))),
            "ai_high": int(round(forecast * 1.15)),
            "veg_pct": 50.0,
            "forecast": forecast,
            "imagery_date": "غير متاح",
            "annotated_img": None,
            "spacing_m": spacing_m,
        }

    min_lon = min(p[0] for p in coords_lon_lat)
    max_lon = max(p[0] for p in coords_lon_lat)
    min_lat = min(p[1] for p in coords_lon_lat)
    max_lat = max(p[1] for p in coords_lon_lat)

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    imagery_date = fetch_satellite_capture_date(center_lon, center_lat)

    x1, y1 = lon_lat_to_tile(min_lon, max_lat, zoom)
    x2, y2 = lon_lat_to_tile(max_lon, min_lat, zoom)

    if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
        zoom = 17
        x1, y1 = lon_lat_to_tile(min_lon, max_lat, zoom)
        x2, y2 = lon_lat_to_tile(max_lon, min_lat, zoom)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    width_px = (x_max - x_min + 1) * 256
    height_px = (y_max - y_min + 1) * 256

    stitched = Image.new("RGB", (width_px, height_px), color=(40, 50, 40))

    has_tile = False
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_img = fetch_satellite_tile(x, y, zoom)
            if tile_img:
                has_tile = True
                stitched.paste(tile_img, ((x - x_min) * 256, (y - y_min) * 256))

    def to_px(lon: float, lat: float) -> Tuple[int, int]:
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        px_x = int(((lon + 180.0) / 360.0 * n - x_min) * 256)
        px_y = int(((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n - y_min) * 256)
        return max(0, min(width_px - 1, px_x)), max(0, min(height_px - 1, px_y))

    poly_px = [to_px(p[0], p[1]) for p in coords_lon_lat]

    corridor_img = Image.new("L", (width_px, height_px), 0)
    draw_corr = ImageDraw.Draw(corridor_img)
    corridor_px_width = 16 if zoom == 18 else 8
    draw_corr.polygon(poly_px, outline=255, fill=0, width=corridor_px_width)

    arr = np.array(stitched, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    corr_mask = np.array(corridor_img) > 0
    total_corr_pixels = int(np.sum(corr_mask))

    # Dynamic sensitivity calibration (0.1 to 1.0)
    sens = max(0.1, min(1.0, float(sensitivity)))
    exg_thresh = max(2.0, 22.0 - (sens * 16.0))      # sens=0.5 -> 14.0; sens=1.0 -> 6.0
    peak_min_exg = max(1.0, 14.0 - (sens * 12.0))    # sens=0.5 -> 8.0; sens=1.0 -> 2.0
    sep_ratio = max(0.35, 0.70 - (sens * 0.25))       # sens=0.5 -> 0.575; sens=1.0 -> 0.45

    if total_corr_pixels > 0 and has_tile:
        exg = 2.0 * g - r - b
        veg_pixels = (exg > exg_thresh) & corr_mask
        veg_pct = round(float(np.sum(veg_pixels) / total_corr_pixels * 100.0), 1)

        density_factor = veg_pct / 100.0
        ai_raw = forecast * (0.4 + 0.7 * density_factor)
        ai_count = int(round(ai_raw))
        ai_low = max(0, int(round(ai_count * 0.85)))
        ai_high = int(round(ai_count * 1.15))
    else:
        veg_pct = 45.0
        ai_count = forecast
        ai_low = int(round(forecast * 0.85))
        ai_high = int(round(forecast * 1.15))

    annotated = stitched.copy()
    draw_ann = ImageDraw.Draw(annotated)

    # 1. Draw boundary line
    draw_ann.line(poly_px + [poly_px[0]], fill=(0, 255, 200), width=3)

    # 2. Optical Crown Peak Detection & Drawing along Polygon Perimeter
    tree_points_coords: List[Dict[str, Any]] = []
    tree_counter = 0

    # Calculate total pixel perimeter
    total_px_len = 0.0
    for i in range(len(poly_px) - 1):
        pA, pB = poly_px[i], poly_px[i + 1]
        total_px_len += math.hypot(pB[0] - pA[0], pB[1] - pA[1])

    # Pixels per meter
    px_per_m = total_px_len / max(perimeter_m, 1.0)
    spacing_px = max(8.0, spacing_m * px_per_m)
    search_r = int(max(4, spacing_px * 0.45))

    placed_points: List[Tuple[int, int]] = []

    for i in range(len(poly_px) - 1):
        pA, pB = poly_px[i], poly_px[i + 1]
        seg_len = math.hypot(pB[0] - pA[0], pB[1] - pA[1])
        if seg_len <= 1.0:
            continue

        num_trees_seg = max(1, int(round(seg_len / spacing_px)))
        for t_idx in range(num_trees_seg):
            ratio = t_idx / num_trees_seg
            tx = pA[0] + ratio * (pB[0] - pA[0])
            ty = pA[1] + ratio * (pB[1] - pA[1])

            # Snap to local greenness peak (ExG) around (tx, ty)
            best_x, best_y = int(round(tx)), int(round(ty))
            if has_tile and total_corr_pixels > 0:
                win_y1 = max(0, int(ty) - search_r)
                win_y2 = min(height_px, int(ty) + search_r + 1)
                win_x1 = max(0, int(tx) - search_r)
                win_x2 = min(width_px, int(tx) + search_r + 1)
                if win_y2 > win_y1 and win_x2 > win_x1:
                    sub_exg = exg[win_y1:win_y2, win_x1:win_x2]
                    max_pos = np.unravel_index(np.argmax(sub_exg), sub_exg.shape)
                    candidate_y = win_y1 + int(max_pos[0])
                    candidate_x = win_x1 + int(max_pos[1])
                    if sub_exg[max_pos] > peak_min_exg:
                        best_x, best_y = candidate_x, candidate_y

            # Ensure minimum separation to avoid duplicate clusters
            too_close = any(math.hypot(best_x - px, best_y - py) < (spacing_px * sep_ratio) for px, py in placed_points)
            if too_close:
                continue

            placed_points.append((best_x, best_y))
            tree_counter += 1

            # Draw circular tree crown marker (Lime green with dark ring)
            r_mark = 5
            draw_ann.ellipse([best_x - r_mark - 1, best_y - r_mark - 1, best_x + r_mark + 1, best_y + r_mark + 1], fill=(0, 0, 0))
            draw_ann.ellipse([best_x - r_mark, best_y - r_mark, best_x + r_mark, best_y + r_mark], fill=(34, 197, 94))

            # Number every 10th tree or first tree
            if tree_counter == 1 or tree_counter % 10 == 0:
                badge_txt = str(tree_counter)
                draw_ann.rectangle([best_x + 5, best_y - 9, best_x + 5 + len(badge_txt) * 9, best_y + 7], fill=(15, 23, 42), outline=(0, 255, 200))
                draw_ann.text((best_x + 7, best_y - 8), badge_txt, fill=(255, 255, 255))

            lat_rad_t = math.atan(math.sinh(math.pi * (1 - 2 * (y_min + best_y / 256.0) / (2.0 ** zoom))))
            lat_deg_t = math.degrees(lat_rad_t)
            lon_deg_t = (x_min + best_x / 256.0) / (2.0 ** zoom) * 360.0 - 180.0

            tree_points_coords.append({
                "id": tree_counter,
                "lat": lat_deg_t,
                "lon": lon_deg_t,
                "px": (best_x, best_y),
            })


    # Total trees count: adjust with corridor vegetation factor
    if tree_counter > 0:
        density_factor = veg_pct / 100.0
        final_ai_count = int(round(tree_counter * (0.6 + 0.5 * density_factor)))
    else:
        final_ai_count = forecast

    # 3. Draw Header Stats Overlay Banner on the Image
    banner_h = 36
    draw_ann.rectangle([0, 0, width_px, banner_h], fill=(15, 23, 42))
    header_text = f"TREE COUNT: {final_ai_count} trees | Spacing: {spacing_m}m | Veg: {veg_pct}% | Date: {imagery_date or 'Recent'}"
    draw_ann.text((12, 10), header_text, fill=(0, 255, 200))

    return {
        "ai_count": final_ai_count,
        "ai_low": max(0, int(round(final_ai_count * 0.85))),
        "ai_high": int(round(final_ai_count * 1.15)),
        "veg_pct": veg_pct,
        "forecast": forecast,
        "imagery_date": imagery_date or "حديث",
        "annotated_img": annotated,
        "spacing_m": spacing_m,
        "tree_points": tree_points_coords,
    }

