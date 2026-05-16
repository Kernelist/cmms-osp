"""
FastAPI application for CMMS monitoring system.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.monitor import run_monitoring_report, analyze_part
from core.calculator import get_survival_curve
from core.pm_optimizer import optimize_pm, compute_cost_curve, optimize_all_parts
from core.pdm_predictor import predict_pdm, predict_all_parts

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "sample_cmms_data.json"
DASHBOARD_PATH = Path(__file__).parent.parent / "web" / "dashboard.html"
REFERENCE_DATE = date.today()

app = FastAPI(
    title="CMMS Monitoring API",
    description="Real-time survival probability monitoring for machine parts",
    version="1.0.0",
)

# ── In-memory state (loaded once on startup, mutations held in memory) ─────────
_cmms_data: dict = {}
_replacement_log: dict = {}  # part_id -> new last_replacement_date


def _load_cmms_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def _parse_ref_date(s: Optional[str]) -> date:
    """Parse YYYY-MM-DD query param; fall back to today."""
    if s:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today()


def _get_live_report(ref: Optional[date] = None) -> dict:
    """Generate a fresh report, applying any in-memory replacement overrides."""
    data = _load_cmms_data()

    # Apply any logged replacements
    for machine in data["machines"]:
        for part in machine["parts"]:
            pid = part["part_id"]
            if pid in _replacement_log:
                part["last_replacement_date"] = _replacement_log[pid]

    # Write patched data to a temp structure and run report
    if ref is None:
        ref = date.today()
    from core.monitor import _parse_date, analyze_part as _analyze_part

    report = {
        "generated_at": datetime.now().isoformat(),
        "reference_date": ref.isoformat(),
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

    for machine in data["machines"]:
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
            analyzed = _analyze_part(part, machine["equipment_type"], ref)
            analyzed["machine_id"] = machine["machine_id"]
            analyzed["machine_name"] = machine["machine_name"]
            machine_report["parts"].append(analyzed)
            report["all_parts"].append(analyzed)

            risk = analyzed["risk_level"]
            machine_report["risk_summary"][risk] += 1
            report["summary"][risk] += 1
            report["summary"]["total_parts"] += 1

        risk_order = ["GREEN", "YELLOW", "ORANGE", "RED"]
        for risk in reversed(risk_order):
            if machine_report["risk_summary"][risk] > 0:
                machine_report["overall_risk"] = risk
                break

        report["machines"].append(machine_report)

    return report


def _find_part(part_id: str, ref: Optional[date] = None) -> tuple[dict | None, dict | None]:
    """Return (part_analyzed, machine) or (None, None) if not found."""
    report = _get_live_report(ref)
    for machine in report["machines"]:
        for part in machine["parts"]:
            if part["part_id"] == part_id:
                return part, machine
    return None, None


# ── Request / Response models ──────────────────────────────────────────────────
class ReplacementRequest(BaseModel):
    replacement_date: str  # "YYYY-MM-DD"


class PMOptimizeRequest(BaseModel):
    part_name: str
    equipment_type: str = "OLD"
    obs_period_years: float = 7.0
    weight: float = 3.0


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Serve dashboard")
async def serve_dashboard():
    """Serve the single-file HTML dashboard."""
    if DASHBOARD_PATH.exists():
        return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found</h1><p>Expected at web/dashboard.html</p>", status_code=404)


@app.get("/api/parts", summary="All parts with current risk status")
async def get_all_parts(reference_date: Optional[str] = None):
    """Return all parts with their current survival/risk metrics."""
    ref = _parse_ref_date(reference_date)
    report = _get_live_report(ref)
    return {
        "reference_date": ref.isoformat(),
        "generated_at": report["generated_at"],
        "summary": report["summary"],
        "parts": report["all_parts"],
    }


@app.get("/api/parts/{part_id}", summary="Single part detail with survival curve")
async def get_part_detail(part_id: str, reference_date: Optional[str] = None):
    """Return full detail for a specific part, including survival curve data for plotting."""
    ref = _parse_ref_date(reference_date)
    part, machine = _find_part(part_id, ref)
    if part is None:
        raise HTTPException(status_code=404, detail=f"Part '{part_id}' not found")

    # Compute max_months: cover data range + current operating time with margin
    from data.survival_data import get_survival_curve_dict
    curve_dict = get_survival_curve_dict(part["part_name"], part["equipment_type"])
    data_max = round(max(curve_dict.keys()), 1) if curve_dict else 40.0
    op_months = float(part.get("operating_months", 0))
    max_months = max(int(data_max * 1.3), int(op_months * 1.3), 50)

    curve = get_survival_curve(
        part["part_name"],
        part["equipment_type"],
        max_months=max_months,
    )

    return {
        "part": part,
        "machine": {
            "machine_id": machine["machine_id"],
            "machine_name": machine["machine_name"],
            "equipment_type": machine["equipment_type"],
        },
        "survival_curve": curve,
        "data_range_months": data_max,
        "reference_date": ref.isoformat(),
    }


@app.get("/api/alerts", summary="Parts with ORANGE or RED risk")
async def get_alerts():
    """Return only parts that require attention (ORANGE or RED risk level)."""
    report = _get_live_report()
    alerts = [
        p for p in report["all_parts"]
        if p["risk_level"] in ("ORANGE", "RED")
    ]
    # Sort: RED first, then ORANGE, then by failure probability descending
    risk_sort = {"RED": 0, "ORANGE": 1}
    alerts.sort(key=lambda p: (risk_sort.get(p["risk_level"], 2), -p["failure_prob"]))

    return {
        "reference_date": report["reference_date"],
        "generated_at": report["generated_at"],
        "alert_count": len(alerts),
        "alerts": alerts,
    }


@app.get("/api/machines", summary="Machine list with overall risk summary")
async def get_machines():
    """Return all machines with part counts and overall risk level."""
    report = _get_live_report()
    machines = []
    for m in report["machines"]:
        machines.append({
            "machine_id": m["machine_id"],
            "machine_name": m["machine_name"],
            "equipment_type": m["equipment_type"],
            "installation_year": m["installation_year"],
            "overall_risk": m["overall_risk"],
            "risk_summary": m["risk_summary"],
            "part_count": sum(m["risk_summary"].values()),
        })
    return {
        "reference_date": report["reference_date"],
        "generated_at": report["generated_at"],
        "machines": machines,
    }


@app.post("/api/parts/{part_id}/replacement", summary="Log a part replacement")
async def log_replacement(part_id: str, body: ReplacementRequest):
    """
    Log a replacement for a part. Updates in-memory state (not persisted to disk).
    The new replacement date is used for all subsequent calculations.
    """
    # Validate date format
    try:
        replacement_date = datetime.strptime(body.replacement_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Verify the part exists
    part, machine = _find_part(part_id)
    if part is None:
        raise HTTPException(status_code=404, detail=f"Part '{part_id}' not found")

    # Store override
    _replacement_log[part_id] = body.replacement_date

    # Recalculate with new date
    updated_part, _ = _find_part(part_id)

    return {
        "message": "Replacement logged successfully",
        "part_id": part_id,
        "part_name": updated_part["part_name"],
        "replacement_date": body.replacement_date,
        "new_risk_level": updated_part["risk_level"],
        "new_survival_prob": updated_part["survival_prob"],
    }


@app.get("/api/health", summary="Health check")
async def health_check():
    return {"status": "ok", "reference_date": REFERENCE_DATE.isoformat()}


# ── PM Optimization Routes ─────────────────────────────────────────────────────

@app.post("/api/optimize/pm-cycle")
def api_optimize_pm(req: PMOptimizeRequest):
    """Find optimal PM interval for a single part."""
    obs_months = req.obs_period_years * 12
    result = optimize_pm(req.part_name, req.equipment_type, obs_months, req.weight)
    return vars(result)


@app.get("/api/optimize/cost-curve/{part_name}")
def api_cost_curve(part_name: str, equipment_type: str = "OLD",
                   obs_period_years: float = 7.0, weight: float = 3.0):
    """Return cost curve data for visualization."""
    obs_months = obs_period_years * 12
    curve = compute_cost_curve(part_name, equipment_type, obs_months, weight)
    result = optimize_pm(part_name, equipment_type, obs_months, weight)
    return {
        "part_name": part_name,
        "equipment_type": equipment_type,
        "obs_period_years": obs_period_years,
        "weight": weight,
        "optimization_result": vars(result),
        "cost_curve": curve
    }


@app.get("/api/optimize/all-parts")
def api_optimize_all(equipment_type: str = "OLD", obs_period_years: float = 7.0,
                     weight: float = 3.0):
    """Run PM optimization for all parts and return summary table."""
    obs_months = obs_period_years * 12
    results = optimize_all_parts(equipment_type, obs_months, weight)

    summary = []
    for r in results:
        item = {
            "part_name": r.part_name,
            "equipment_type": r.equipment_type,
            "strategy": r.strategy,
            "hazard_trend_beta": r.hazard_trend_beta,
            "data_range_months": r.data_range_months,
            "n_data_points": r.n_data_points,
            "optimal_pm_months": r.optimal_pm_months,
            "optimal_pm_days": r.optimal_pm_days,
            "optimal_survival_pct": r.optimal_survival_pct,
            "n_pm_per_period": r.n_pm_per_period,
            "n_rm_per_period": r.n_rm_per_period,
            "min_total_cost": r.min_total_cost,
            "strategy_reason": r.strategy_reason
        }
        summary.append(item)

    # Group by strategy
    strategy_counts = {}
    for r in results:
        strategy_counts[r.strategy] = strategy_counts.get(r.strategy, 0) + 1

    return {
        "obs_period_years": obs_period_years,
        "weight": weight,
        "equipment_type": equipment_type,
        "strategy_summary": strategy_counts,
        "parts": summary
    }


# ── PdM (Predictive Maintenance) Routes ──────────────────────────────────────

@app.get("/api/pdm/predict/{part_id}", summary="PdM prediction for a single part")
async def api_pdm_predict(part_id: str, reference_date: Optional[str] = None):
    """
    Multiple Classifier 앙상블 PdM 예측 (Susto et al. 2015).
    기존 PM 생존모델 + Cox PH + 이력패턴 + 추세분석 4개 분류기의 가중 투표.
    """
    ref = _parse_ref_date(reference_date)
    part, machine = _find_part(part_id, ref)
    if part is None:
        raise HTTPException(status_code=404, detail=f"Part '{part_id}' not found")

    history = [part["last_replacement_date"]]

    pred = predict_pdm(
        part_name=part["part_name"],
        equipment_type=machine["equipment_type"],
        operating_months=part["operating_months"],
        installation_year=machine.get("installation_year", 2005),
        replacement_history=history,
        reference_date=ref,
    )

    return {
        "part_id": part_id,
        "part_name": part["part_name"],
        "machine_id": machine["machine_id"],
        "machine_name": machine["machine_name"],
        "reference_date": ref.isoformat(),
        "operating_months": pred.operating_months,
        "pm_risk_level": pred.pm_risk_level,
        "pdm_risk_level": pred.ensemble_class,
        "pdm_confidence": pred.ensemble_confidence,
        "pdm_vs_pm": pred.pdm_vs_pm,
        "agreement_ratio": pred.agreement_ratio,
        "estimated_rul_months": pred.estimated_rul_months,
        "estimated_rul_days": pred.estimated_rul_days,
        "ensemble_probabilities": pred.ensemble_probabilities,
        "classifiers": pred.classifiers,
        "features": pred.features,
    }


@app.get("/api/pdm/model-info", summary="ML model training info")
async def api_pdm_model_info():
    """ML 분류기 학습 정보 및 정확도."""
    from core.model_trainer import load_models
    trained = load_models()
    metrics = trained.get("metrics", {})
    weights = metrics.get("weights", {})
    return {
        "models": [
            {"name": "Survival Model", "key": "survival_model",
             "accuracy": metrics.get("survival_model", {}).get("accuracy", 0),
             "weight": weights.get("survival_model", 0.15), "type": "rule-based"},
            {"name": "SBM (Sensor)", "key": "sbm",
             "accuracy": metrics.get("sbm", {}).get("accuracy", 0),
             "weight": weights.get("sbm", 0.15), "type": "sensor-based"},
            {"name": "Random Forest", "key": "random_forest",
             "accuracy": metrics.get("random_forest", {}).get("accuracy", 0),
             "weight": weights.get("random_forest", 0.30), "type": "ml-trained"},
            {"name": "SVM (RBF)", "key": "svm",
             "accuracy": metrics.get("svm", {}).get("accuracy", 0),
             "weight": weights.get("svm", 0.25), "type": "ml-trained"},
            {"name": "Gradient Boosting", "key": "gradient_boosting",
             "accuracy": metrics.get("gradient_boosting", {}).get("accuracy", 0),
             "weight": weights.get("gradient_boosting", 0.30), "type": "ml-trained"},
        ],
        "weight_method": "log-odds (Susto et al. 2015)",
        "feature_count": len(trained.get("feature_names", [])),
        "feature_names": trained.get("feature_names", []),
    }


@app.post("/api/pdm/retrain", summary="Retrain ML models")
async def api_pdm_retrain():
    """ML 모델 재학습."""
    from core.model_trainer import train_all_models
    import core.model_trainer as mt
    mt._cached_models = None
    result = train_all_models(verbose=False)
    return {
        "status": "ok",
        "accuracies": {
            k: v.get("accuracy", 0) for k, v in result["metrics"].items()
        },
    }


@app.get("/api/pdm/all", summary="PdM predictions for all parts")
async def api_pdm_all(reference_date: Optional[str] = None):
    """전체 부품 PdM 예측 결과 및 PM 대비 변화 요약."""
    ref = _parse_ref_date(reference_date)
    predictions = predict_all_parts(ref)

    # PM vs PdM 변화 요약
    change_summary = {"SAME": 0, "UPGRADED": 0, "DOWNGRADED": 0}
    risk_summary = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
    for p in predictions:
        change_summary[p["pdm_vs_pm"]] += 1
        risk_summary[p["pdm_risk_level"]] += 1

    return {
        "reference_date": ref.isoformat(),
        "total_parts": len(predictions),
        "pdm_risk_summary": risk_summary,
        "pm_vs_pdm_changes": change_summary,
        "predictions": predictions,
    }
