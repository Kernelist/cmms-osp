"""
Survival probability data for CMMS parts.
Loaded from: data/Part_Survial_Probability_raw.csv
Time unit: MONTHS (converted from days, 1 month = 30.44 days)

Age types:
  OLD   — legacy equipment (pre-2006)
  NEW   — newer equipment (post-2018)
  ALL   — combined pool (전체)
  ACCEL — 5x accelerated degradation scenario (OLD × 1/5 time)
"""
import csv
from pathlib import Path
from collections import defaultdict

_CSV_PATH = Path(__file__).parent / "Part_Survial_Probability_raw.csv"

_LOWER_WORDS = {'for', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'by'}
_UPPER_WORDS = {'PLC', 'S/W', 'SW', 'SDL', 'ABS', 'LED'}


def _normalize_name(raw: str) -> str:
    words = raw.strip().split()
    result = []
    for i, word in enumerate(words):
        prefix = '(' if word.startswith('(') else ''
        suffix = ')' if word.endswith(')') else ''
        core = word.strip('()')
        if core.upper() in _UPPER_WORDS:
            result.append(prefix + core.upper() + suffix)
        elif i > 0 and core.lower() in _LOWER_WORDS:
            result.append(prefix + core.lower() + suffix)
        else:
            result.append(prefix + core.capitalize() + suffix)
    return ' '.join(result)


def _load_csv() -> dict:
    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"Survival data CSV not found: {_CSV_PATH}")

    raw: dict = defaultdict(lambda: defaultdict(list))
    with open(_CSV_PATH, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            age = row['Age'].strip()
            part = _normalize_name(row['PART_NM_EN'].strip())
            day = float(row['duration_censoring(day)'])
            surv = float(row['Survial'])
            raw[age][part].append((day, surv))

    result: dict = {}
    for age, parts in raw.items():
        result[age] = {}
        for pname, points in parts.items():
            points.sort(key=lambda x: x[0])
            steps: dict = {}
            prev_s = None
            for day, s in points:
                if s != prev_s:
                    month = round(day / 30.44, 2)
                    steps[month] = round(s, 4)
                    prev_s = s
            if steps:
                result[age][pname] = steps

    return result


_raw = _load_csv()

SURVIVAL_DATA_OLD = _raw.get('OLD', {})
SURVIVAL_DATA_NEW = _raw.get('NEW', {})
SURVIVAL_DATA_ALL = _raw.get('결합', {})

SURVIVAL_DATA_ACCEL = {
    part: {round(t / 5, 4): s for t, s in curve.items()}
    for part, curve in SURVIVAL_DATA_OLD.items()
}


def interpolate_survival(time_months: float, survival_curve_dict: dict) -> float:
    if not survival_curve_dict:
        return 1.0

    sorted_points = sorted(survival_curve_dict.items(), key=lambda x: x[0])

    if len(sorted_points) == 1:
        t0, p0 = sorted_points[0]
        if time_months <= t0:
            return 1.0
        if t0 == 0:
            return max(0.0, 1.0 - 0.5 * time_months)
        slope = -0.5 / t0
        return max(0.0, 1.0 + slope * (time_months - t0))

    times = [p[0] for p in sorted_points]
    probs = [p[1] for p in sorted_points]

    if time_months <= times[0]:
        return 1.0

    if time_months >= times[-1]:
        t1, p1 = times[-2], probs[-2]
        t2, p2 = times[-1], probs[-1]
        if t2 == t1:
            return max(0.0, p2)
        slope = (p2 - p1) / (t2 - t1)
        return max(0.0, p2 + slope * (time_months - t2))

    for i in range(len(times) - 1):
        if times[i] <= time_months <= times[i + 1]:
            t1, p1 = times[i], probs[i]
            t2, p2 = times[i + 1], probs[i + 1]
            if t2 == t1:
                return p1
            fraction = (time_months - t1) / (t2 - t1)
            return p1 + fraction * (p2 - p1)

    return 1.0


def get_survival_curve_dict(part_name: str, equipment_type: str = "OLD") -> dict:
    et = equipment_type.upper()
    if et == "ACCEL":
        return SURVIVAL_DATA_ACCEL.get(part_name, {})
    if et == "NEW":
        return SURVIVAL_DATA_NEW.get(part_name, {})
    if et == "ALL":
        return SURVIVAL_DATA_ALL.get(part_name, {})
    return SURVIVAL_DATA_OLD.get(part_name, {})


def get_all_part_names(equipment_type: str = "OLD") -> list:
    et = equipment_type.upper()
    if et == "ACCEL":
        data = SURVIVAL_DATA_ACCEL
    elif et == "NEW":
        data = SURVIVAL_DATA_NEW
    elif et == "ALL":
        data = SURVIVAL_DATA_ALL
    else:
        data = SURVIVAL_DATA_OLD
    return sorted(data.keys())
