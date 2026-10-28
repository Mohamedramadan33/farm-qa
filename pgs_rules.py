import math
import re
from datetime import datetime

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_STRIP_RE = re.compile(r"[\u0640\u064B-\u065F\u200B-\u200F\u202A-\u202E\u2066-\u2069\ufeff]")


def norm(s) -> str:
    if s is None:
        return ""
    s = str(s).translate(_AR_DIGITS)
    s = _STRIP_RE.sub("", s)
    s = s.replace("\u00a0", " ")
    for a in "أإآ":
        s = s.replace(a, "ا")
    s = s.replace("ى", "ي").replace("ة", "ه")
    return " ".join(s.split()).strip()


# ملاحظة مهمة: Raw_Label لازم يكون نص السؤال الصافي (من غير "[0] " ولا "(leaf)")
# عشان المقارنات الـ exact (زي "اسم المحصول") تشتغل.
def L(row) -> str: return norm(row.get("Raw_Label"))
def V(row) -> str: return norm(row.get("Submitted_Answer"))
def K(row) -> str: return str(row.get("Attachment_Link") or "").strip()


def is_yes(v: str) -> bool:
    return norm(v).lstrip("-. ").startswith(norm("نعم"))


def to_float(s):
    s = norm(s).replace("٫", ".").replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def first_row(table, text, exact=False):
    t = norm(text)
    for row in table:
        l = L(row)
        if (l == t) if exact else (t in l):
            return row
    return None


def has_link(table, text, exact=False, soft=False) -> bool:
    if soft:
        return True
    t = norm(text)
    for row in table:
        l = L(row)
        if ((l == t) if exact else (t in l)) and (K(row) or V(row)):
            return True
    return False


def section(table, start_text, end_texts):
    st = norm(start_text)
    start = None
    for i, row in enumerate(table):
        if st in L(row):
            start = i
            break
    if start is None:
        return []
    ends = [norm(e) for e in end_texts]
    end = len(table)
    for j in range(start + 1, len(table)):
        if any(e in L(table[j]) for e in ends):
            end = j
            break
    return table[start:end]


def is_valid_egypt_phone(phone_str: str) -> bool:
    clean = "".join(c for c in norm(phone_str) if c.isdigit())
    if clean.startswith("0020"):
        clean = "0" + clean[4:]
    elif clean.startswith("20") and len(clean) in (11, 12):
        clean = "0" + clean[2:]
    if len(clean) == 10 and clean.startswith(("10", "11", "12", "15")):
        clean = "0" + clean
    if len(clean) == 11 and clean.startswith(('010', '011', '012', '015')):
        return True
    landline_prefixes = ('02', '03', '040', '045', '047', '048', '050', '055', '057',
                         '062', '064', '065', '066', '068', '069', '082', '084',
                         '086', '088', '092', '093', '095', '096', '097')
    return len(clean) in (9, 10) and clean.startswith(landline_prefixes)


def parse_kobo_date(date_str: str):
    formats = ["%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            pass
    return None


def point_in_polygon(lat, lng, poly):
    """poly = list of (lat, lng)."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i][0], poly[i][1]
        yj, xj = poly[j][0], poly[j][1]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def meters_to_polygon_edge(lat, lng, poly):
    """poly = list of (lat, lng)."""
    m_lat = 111132.92
    m_lng = 111412.84 * math.cos(math.radians(lat))
    px, py = 0.0, 0.0
    best = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ax, ay = (a[1] - lng) * m_lng, (a[0] - lat) * m_lat
        bx, by = (b[1] - lng) * m_lng, (b[0] - lat) * m_lat
        dx, dy = bx - ax, by - ay
        seg = dx * dx + dy * dy
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(cx - px, cy - py))
    return best


def soil_points_from_table(table):
    """Kobo geopoint = 'lat lon alt acc' → بنرجع (lat, lon)."""
    pts = []
    for row in table:
        if norm("نقاط اخذ عينة للتربة") in L(row):
            nums = re.findall(r"-?\d+\.\d+", str(row.get("Submitted_Answer") or ""))
            if len(nums) >= 2:
                pts.append((float(nums[0]), float(nums[1])))
    return pts


def parse_area_feddan(s):
    s = norm(s).replace("٫", ".").replace(",", ".")
    units = {"فدان": 1.0, "قيراط": 1 / 24, "سهم": 1 / 576}
    total, found = 0.0, False
    # FIX: "فدانين" كانت بترجع 2 على طول وتتجاهل "وقيراطين" مثلا
    if "فدانين" in s:
        total, found = 2.0, True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(فدان|قيراط|سهم)", s):
        total += float(m.group(1)) * units[m.group(2)]
        found = True
    if found:
        return total
    if "فدان" in s:
        return 1.0
    return to_float(s)


SOIL_MIN_SAMPLES = 4
SOIL_EDGE_TOLERANCE_M = 5


def check_rules(table: list, calc_area=None, farm_polygon=None) -> list:
    """
    table: list of dicts فيها Raw_Label / Submitted_Answer / Attachment_Link
    farm_polygon: list of (lat, lon)  ← مهم: lat الأول
    """
    broken_rules = []
    add = broken_rules.append
    today = datetime.now().date()
    n = len(table)
    links_ok = any(K(r) for r in table)

    total_farm_area = 0.0
    r = first_row(table, "إجمالي المساحة بالفدان")
    if r is not None:
        total_farm_area = to_float(r.get("Submitted_Answer")) or 0.0

    calc_ok = calc_area is not None and calc_area == calc_area and calc_area > 0  # NaN-safe
    if calc_ok and total_farm_area > 0:
        if abs(calc_area - total_farm_area) / total_farm_area > 0.10:
            add(f"المزرعة المرسومة {round(calc_area, 2):g} فدان لكن الاستمارة مكتوب فيها {total_farm_area:g} فدان")

    soils = soil_points_from_table(table)
    if len(soils) < SOIL_MIN_SAMPLES:
        add(f"عينات التربة ناقصة ({len(soils)} من {SOIL_MIN_SAMPLES})")

    poly = [(float(p[0]), float(p[1])) for p in (farm_polygon or []) if p and len(p) >= 2]
    if len(poly) >= 3 and soils:
        outside = [i + 1 for i, (la, ln) in enumerate(soils)
                   if not point_in_polygon(la, ln, poly) and meters_to_polygon_edge(la, ln, poly) > SOIL_EDGE_TOLERANCE_M]
        inside_count = len(soils) - len(outside)
        if outside and inside_count < SOIL_MIN_SAMPLES:
            nums = "، ".join(str(x) for x in outside)
            add(f"عينة التربة رقم {nums} برا حدود المزرعة" if len(outside) == 1 else f"عينات التربة رقم {nums} برا حدود المزرعة")

    r = first_row(table, "تاريخ الزيارة")
    if r is not None:
        v_date = parse_kobo_date(r.get("Submitted_Answer", ""))
        if v_date and v_date > today:
            add("تاريخ الزيارة في المستقبل")

    r = first_row(table, "رقم تليفون ممثل")
    if r is not None:
        phone_val = str(r.get("Submitted_Answer") or "").strip()
        if phone_val and not is_valid_egypt_phone(phone_val):
            add("رقم تليفون ممثل المزرعة غلط")

    auditor_photos_count = sum(1 for row in table if norm("صورة لعضو لجنة المراجعة") in L(row) and (K(row) or V(row)))
    if links_ok and auditor_photos_count < 3:
        add(f"صور لجنة المراجعة ناقصة ({auditor_photos_count} من 3)")

    ownership_type = ""
    r = first_row(table, "مملوكة لمالك واحد") or first_row(table, "لمجموعة مزارعين")
    if r is not None:
        ownership_type = V(r)

    if norm("مجموعة مزارعين") in ownership_type:
        farmer_names, basin_areas, farmer_phones = [], [], []
        last_owner = ""
        for row in table:
            lbl, val = L(row), V(row)
            if norm("اسم صاحب الحوض") in lbl and val:
                farmer_names.append(val)
                last_owner = str(row.get("Submitted_Answer")).strip()
            elif norm("مساحة الحوض بالفدان") in lbl and val:
                f = to_float(val)
                if f is not None:
                    basin_areas.append(f)
            elif lbl == norm("التليفون") and val:
                farmer_phones.append((last_owner, str(row.get("Submitted_Answer")).strip()))

        if len(farmer_names) == 1:
            add("قائمة المزارعين فيها مزارع واحد بس")
        for owner, ph in farmer_phones:
            if not is_valid_egypt_phone(ph):
                add(f"رقم تليفون {owner} غلط" if owner else "فيه رقم تليفون غلط في قائمة المزارعين")
        summed_basin_area = sum(basin_areas)
        if abs(summed_basin_area - total_farm_area) > 0.001:
            add(f"مساحات المزارعين ({summed_basin_area:g} فدان) مش زي مساحة المزرعة ({total_farm_area:g} فدان)")

    if not has_link(table, "ارفق صورة الحيازة"):
        add("الحيازة مش مرفقة")
    if not has_link(table, "الرجاء ارفاق بيان التنوع الحيوي"):
        add("بيان التنوع الحيوي مش مرفق")
    if not has_link(table, "الرجاء ارفاق بيان GMO"):
        add("بيان GMO مش مرفق")

    CROP_NAME = norm("اسم المحصول")
    BOUNDARIES = {CROP_NAME, norm("رقم الحوض"), norm("نوع المحصول")}
    CUR, PREV = norm("الموسم الحالي"), norm("الموسم السابق")
    PHOTO_LABELS = (norm("ارفق الصورة"), norm("ارفق صورة"))

    current_crop_areas, previous_crop_areas = [], []
    has_current_crop_photo, has_previous_crop_data = True, False

    for i in range(n):
        if L(table[i]) != CROP_NAME:
            continue
        crop_name = V(table[i])
        end, marker = n, None
        for j in range(i + 1, n):
            lj = L(table[j])
            if lj in BOUNDARIES:
                end = j
                break
            if CUR in lj:
                marker, end = "cur", j + 1
                break
            if PREV in lj:
                marker, end = "prev", j + 1
                break
        group = table[i:end]

        if marker is None:
            labels = [L(g) for g in group]
            marker = "cur" if norm("الانتاجية المتوقعه") in labels else ("prev" if norm("الانتاجية") in labels else "cur")

        area = next((parse_area_feddan(g.get("Submitted_Answer")) for g in group if L(g) == norm("المساحة")), None)

        if marker == "cur":
            if area is not None:
                current_crop_areas.append(area)
            photo_ok = any(L(g) in PHOTO_LABELS and (K(g) or V(g)) for g in group)
            k = end
            while not photo_ok and k < n and k < end + 3 and L(table[k]) not in BOUNDARIES:
                if L(table[k]) in PHOTO_LABELS:
                    photo_ok = bool(K(table[k]) or V(table[k]))
                    break
                k += 1
            if links_ok and not photo_ok:
                has_current_crop_photo = False
        else:
            if crop_name:
                has_previous_crop_data = True
            if area is not None:
                previous_crop_areas.append(area)

    if not has_current_crop_photo:
        add("صورة المحصول الحالي مش مرفقة")
    sum_current_area = sum(current_crop_areas)
    if abs(sum_current_area - total_farm_area) > 0.001:
        add(f"المساحة المزروعة هذا الموسم {total_farm_area:g} فدان حسب الاستمارة، لكن بيانات المحاصيل مجموعها {sum_current_area:g} فدان")

    if not has_previous_crop_data:
        add("بيانات المحاصيل السابقة ناقصة")
    else:
        sum_prev_area = sum(previous_crop_areas)
        if abs(sum_prev_area - total_farm_area) > 0.001:
            add(f"المساحة المزروعة الموسم السابق {total_farm_area:g} فدان، لكن بيانات المحاصيل المدخلة {sum_prev_area:g} فدان")

    if not has_link(table, "ارفاق صورة الدفتر", soft=not links_ok):
        add("دفتر المزرعة مش مرفق")

    return broken_rules
