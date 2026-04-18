"""
Core calculator functions for CMMS survival probability analysis.
"""

from data.survival_data import (
    SURVIVAL_DATA_OLD,
    SURVIVAL_DATA_NEW,
    interpolate_survival,
    get_survival_curve_dict,
)


def get_survival_probability(
    part_name: str, operating_months: float, equipment_type: str = "OLD"
) -> float:
    """
    Return the survival probability (0.0–1.0) for a part at a given operating age.

    Args:
        part_name: Name of the part (must match a key in SURVIVAL_DATA_OLD/NEW)
        operating_months: Months since last replacement
        equipment_type: "OLD" (pre-2006) or "NEW"

    Returns:
        Survival probability between 0.0 and 1.0. Returns 1.0 if part not found.
    """
    curve = get_survival_curve_dict(part_name, equipment_type)
    if not curve:
        return 1.0
    return interpolate_survival(operating_months, curve)


def get_failure_probability(
    part_name: str, operating_months: float, equipment_type: str = "OLD"
) -> float:
    """
    Return the failure probability (0.0–1.0) = 1 - survival probability.
    """
    return 1.0 - get_survival_probability(part_name, operating_months, equipment_type)


def get_risk_level(failure_probability: float) -> str:
    """
    Convert failure probability to a risk level string.

    Thresholds:
        GREEN  : failure_prob < 0.10  (survival > 90%)
        YELLOW : failure_prob < 0.20  (survival > 80%)
        ORANGE : failure_prob < 0.35  (survival > 65%)
        RED    : failure_prob >= 0.35 (survival <= 65%)
    """
    if failure_probability < 0.10:
        return "GREEN"
    elif failure_probability < 0.20:
        return "YELLOW"
    elif failure_probability < 0.35:
        return "ORANGE"
    else:
        return "RED"


def _find_month_for_failure_prob(
    part_name: str, target_failure_prob: float, equipment_type: str, max_months: int = 2000
) -> float | None:
    """
    Binary search to find the month at which failure probability reaches target_failure_prob.
    Returns None if not reached within max_months.
    """
    target_survival = 1.0 - target_failure_prob

    # Check if we ever reach that survival level
    end_survival = get_survival_probability(part_name, float(max_months), equipment_type)
    if end_survival > target_survival:
        return None  # Never reaches this threshold in our range

    # Binary search
    lo, hi = 0.0, float(max_months)
    for _ in range(60):  # ~60 iterations gives sub-millisecond precision
        mid = (lo + hi) / 2
        surv = get_survival_probability(part_name, mid, equipment_type)
        if surv > target_survival:
            lo = mid
        else:
            hi = mid

    return round((lo + hi) / 2, 1)


def get_replacement_recommendation(
    part_name: str, operating_months: float, equipment_type: str = "OLD"
) -> dict:
    """
    Return a comprehensive replacement recommendation dict.

    Returns:
        {
            "urgency": str,               # "OK" / "MONITOR" / "PLAN_REPLACEMENT" / "REPLACE_NOW"
            "survival_prob": float,       # current survival probability
            "failure_prob": float,        # current failure probability
            "risk_level": str,            # GREEN / YELLOW / ORANGE / RED
            "months_to_80pct_failure": float | None,  # months from NOW until 80% failure prob
            "recommended_replacement_month": float | None,  # operating months at which to replace
        }
    """
    survival_prob = get_survival_probability(part_name, operating_months, equipment_type)
    failure_prob = 1.0 - survival_prob
    risk_level = get_risk_level(failure_prob)

    # Urgency mapping
    urgency_map = {
        "GREEN": "OK",
        "YELLOW": "MONITOR",
        "ORANGE": "PLAN_REPLACEMENT",
        "RED": "REPLACE_NOW",
    }
    urgency = urgency_map[risk_level]

    # Find when 80% failure probability is reached (20% survival)
    month_80pct = _find_month_for_failure_prob(part_name, 0.80, equipment_type)

    # Recommended replacement: at 20% failure (80% survival) — proactive threshold
    recommended_month = _find_month_for_failure_prob(part_name, 0.20, equipment_type)

    # months_to_80pct_failure: how many months from NOW until 80% failure
    months_to_80pct = None
    if month_80pct is not None:
        months_to_80pct = max(0.0, month_80pct - operating_months)

    return {
        "urgency": urgency,
        "survival_prob": round(survival_prob, 4),
        "failure_prob": round(failure_prob, 4),
        "risk_level": risk_level,
        "months_to_80pct_failure": round(months_to_80pct, 1) if months_to_80pct is not None else None,
        "recommended_replacement_month": recommended_month,
    }


def get_survival_curve(
    part_name: str, equipment_type: str = "OLD", max_months: int = 1000
) -> list[dict]:
    """
    Generate a survival/failure curve for plotting.

    Returns a list of dicts with:
        {
            "month": int,
            "survival": float,   # 0.0–1.0
            "failure": float,    # 0.0–1.0
        }

    Samples at every 5 months from 0 to max_months.
    Stops early if survival drops to 0.
    """
    curve = []
    step = 1
    for month in range(0, max_months + step, step):
        surv = get_survival_probability(part_name, float(month), equipment_type)
        surv = max(0.0, min(1.0, surv))
        fail = 1.0 - surv
        curve.append({
            "month": month,
            "survival": round(surv, 4),
            "failure": round(fail, 4),
        })
        if surv <= 0.0:
            break

    return curve
