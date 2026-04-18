"""
CMMS Monitoring Engine.
Loads CMMS data, calculates risk for each part, and generates a monitoring report.
"""

import json
import sys
import os
from datetime import datetime, date
from pathlib import Path

# Allow imports from the project root when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.calculator import (
    get_survival_probability,
    get_failure_probability,
    get_risk_level,
    get_replacement_recommendation,
)

# Risk level display config
RISK_CONFIG = {
    "GREEN":  {"symbol": "[G]", "label": "GREEN",  "color_code": "\033[92m"},
    "YELLOW": {"symbol": "[Y]", "label": "YELLOW", "color_code": "\033[93m"},
    "ORANGE": {"symbol": "[O]", "label": "ORANGE", "color_code": "\033[33m"},
    "RED":    {"symbol": "[R]", "label": "RED",    "color_code": "\033[91m"},
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _months_between(start: date, end: date) -> float:
    """Return fractional months between two dates (end - start)."""
    delta_days = (end - start).days
    return delta_days / 30.4375  # average days per month


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def analyze_part(part: dict, equipment_type: str, reference_date: date) -> dict:
    """
    Compute all monitoring metrics for a single part.

    Returns an enriched part dict with:
        operating_months, survival_prob, failure_prob, risk_level,
        urgency, months_to_80pct_failure, recommended_replacement_month
    """
    last_replaced = _parse_date(part["last_replacement_date"])
    operating_months = max(0.0, _months_between(last_replaced, reference_date))

    rec = get_replacement_recommendation(
        part["part_name"], operating_months, equipment_type
    )

    # Estimate recommended replacement date from operating months threshold
    recommended_replacement_date = None
    if rec["recommended_replacement_month"] is not None:
        months_remaining = rec["recommended_replacement_month"] - operating_months
        if months_remaining > 0:
            from datetime import timedelta
            days_remaining = int(months_remaining * 30.4375)
            rec_date = reference_date + timedelta(days=days_remaining)
            recommended_replacement_date = rec_date.isoformat()
        else:
            recommended_replacement_date = "OVERDUE"

    return {
        **part,
        "equipment_type": equipment_type,
        "operating_months": round(operating_months, 1),
        "survival_prob": rec["survival_prob"],
        "failure_prob": rec["failure_prob"],
        "risk_level": rec["risk_level"],
        "urgency": rec["urgency"],
        "months_to_80pct_failure": rec["months_to_80pct_failure"],
        "recommended_replacement_month": rec["recommended_replacement_month"],
        "recommended_replacement_date": recommended_replacement_date,
        "reference_date": reference_date.isoformat(),
    }


def run_monitoring_report(
    cmms_data_path: str, reference_date: date | None = None
) -> dict:
    """
    Load CMMS data and compute full monitoring report.

    Args:
        cmms_data_path: Path to sample_cmms_data.json (or similar)
        reference_date: Date to use as "today" for operating months calculation.
                        Defaults to datetime.today().

    Returns:
        Structured report dict with machines, parts, and summary statistics.
    """
    if reference_date is None:
        reference_date = datetime.today().date()

    with open(cmms_data_path, "r") as f:
        cmms_data = json.load(f)

    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_date": reference_date.isoformat(),
        "machines": [],
        "all_parts": [],
        "summary": {
            "total_parts": 0,
            "GREEN": 0,
            "YELLOW": 0,
            "ORANGE": 0,
            "RED": 0,
        },
    }

    for machine in cmms_data["machines"]:
        machine_report = {
            "machine_id": machine["machine_id"],
            "machine_name": machine["machine_name"],
            "equipment_type": machine["equipment_type"],
            "installation_year": machine["installation_year"],
            "parts": [],
            "risk_summary": {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0},
            "overall_risk": "GREEN",
        }

        for part in machine["parts"]:
            analyzed = analyze_part(part, machine["equipment_type"], reference_date)
            analyzed["machine_id"] = machine["machine_id"]
            analyzed["machine_name"] = machine["machine_name"]
            machine_report["parts"].append(analyzed)
            report["all_parts"].append(analyzed)

            risk = analyzed["risk_level"]
            machine_report["risk_summary"][risk] += 1
            report["summary"][risk] += 1
            report["summary"]["total_parts"] += 1

        # Overall machine risk = worst part risk
        risk_order = ["GREEN", "YELLOW", "ORANGE", "RED"]
        for risk in reversed(risk_order):
            if machine_report["risk_summary"][risk] > 0:
                machine_report["overall_risk"] = risk
                break

        report["machines"].append(machine_report)

    return report


def _color(text: str, risk_level: str) -> str:
    cfg = RISK_CONFIG.get(risk_level, RISK_CONFIG["GREEN"])
    return f"{cfg['color_code']}{text}{RESET}"


def print_monitoring_report(report: dict) -> None:
    """Print a colored console monitoring report."""
    ref = report["reference_date"]
    gen = report["generated_at"]
    s = report["summary"]

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  CMMS REAL-TIME MONITORING REPORT{RESET}")
    print(f"  Reference date : {ref}")
    print(f"  Generated at   : {gen}")
    print(f"{BOLD}{'='*70}{RESET}")

    # Summary bar
    print(f"\n  {BOLD}FLEET SUMMARY{RESET}  (total parts: {s['total_parts']})")
    for level in ["GREEN", "YELLOW", "ORANGE", "RED"]:
        count = s[level]
        bar = "#" * count
        label = _color(f"  {level:<8} {count:>3}  {bar}", level)
        print(label)

    # Per-machine detail
    for machine in report["machines"]:
        overall = machine["overall_risk"]
        print(f"\n  {BOLD}Machine: {machine['machine_name']}  [{machine['machine_id']}]{RESET}")
        print(f"  Type: {machine['equipment_type']}  |  Overall Risk: {_color(overall, overall)}")
        print(f"  {'Part Name':<30} {'Op.Mo':>6}  {'Surv%':>6}  {'Fail%':>6}  {'Risk':<8}  Urgency")
        print(f"  {'-'*80}")

        # Sort parts: RED first, then ORANGE, YELLOW, GREEN
        risk_sort = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
        sorted_parts = sorted(
            machine["parts"], key=lambda p: risk_sort.get(p["risk_level"], 4)
        )

        for part in sorted_parts:
            risk = part["risk_level"]
            cfg = RISK_CONFIG[risk]
            symbol = cfg["symbol"]
            name = part["part_name"][:29]
            op_mo = part["operating_months"]
            surv = f"{part['survival_prob']*100:.1f}%"
            fail = f"{part['failure_prob']*100:.1f}%"
            urgency = part["urgency"]

            line = f"  {symbol} {name:<29} {op_mo:>6.1f}  {surv:>6}  {fail:>6}  {risk:<8}  {urgency}"
            print(_color(line, risk))

    print(f"\n{BOLD}{'='*70}{RESET}\n")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "sample_cmms_data.json"
    ref_date = date(2024, 2, 1)
    report = run_monitoring_report(str(data_path), reference_date=ref_date)
    print_monitoring_report(report)
