"""
PM Cycle Optimization Engine — Empirical Survival Data Based
============================================================
Algorithm: Periodic PM with minimal repair on failure (경험적 생존확률 기반)

비용 공식:
  Cost(T) = (P/T) × (1 + w × H(T))
  H(T)    = -ln(S(T))   (경험적 누적 위험함수, VJ OSP 실측 데이터 기반)
  S(T)    = interpolate_survival(T, curve)  (PPTX 실측 생존확률 보간)

전략 판별 (β 추세 분석):
  β > 1.05 → 마모 고장 (Wear-out) → 최적 PM 주기 존재
  β ≈ 1    → 랜덤 고장 (Random)   → PM 효과 없음
  β < 0.95 → 초기 고장 (Infant)   → Run-To-Failure 권장

최적 PM 주기: T* = argmin Cost(T) 수치 탐색 (실측 데이터 범위 내)
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PMOptimizationResult:
    part_name: str
    equipment_type: str
    obs_period_months: float
    weight: float

    # 추세 분석 (전략 판별용)
    hazard_trend_beta: float   # ln(H) vs ln(T) 회귀 기울기
    data_range_months: float   # 실측 데이터 최대 범위 (월)
    n_data_points: int         # 사용 가능한 데이터 포인트 수

    # 전략
    strategy: str              # "OPTIMAL_PM" | "RUN_TO_FAILURE" | "RANDOM_FAILURE" | "INSUFFICIENT_DATA"
    strategy_reason: str

    # 최적 PM 결과 (strategy == OPTIMAL_PM 일 때만 유효)
    optimal_pm_months: Optional[float] = None
    optimal_pm_days: Optional[float] = None
    optimal_survival_pct: Optional[float] = None   # 최적 PM 시점의 생존확률 (%)
    min_total_cost: Optional[float] = None
    n_pm_per_period: Optional[float] = None
    n_rm_per_period: Optional[float] = None


def _estimate_beta(survival_curve: dict) -> Optional[float]:
    """
    ln(H) = β·ln(T) + const 회귀로 위험 증가율 β 추정.
    전략 판별(PM 여부 결정)에만 사용; 비용 계산에는 사용 안 함.
    """
    valid = [(t, s) for t, s in sorted(survival_curve.items())
             if t > 0 and 0 < s < 1.0]
    if len(valid) < 2:
        return None

    ln_T = [math.log(t) for t, _ in valid]
    ln_H = [math.log(-math.log(s)) for _, s in valid]

    n = len(ln_T)
    sum_x, sum_y = sum(ln_T), sum(ln_H)
    sum_xy = sum(x * y for x, y in zip(ln_T, ln_H))
    sum_xx = sum(x * x for x in ln_T)
    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-12:
        return None

    beta = (n * sum_xy - sum_x * sum_y) / denom
    return beta if beta > 0 else None


def _empirical_cost(T: float, curve: dict, obs_period: float, weight: float,
                    interpolate_fn) -> float:
    """실측 생존확률 기반 비용 계산: Cost(T) = (P/T) × (1 + w × H(T))"""
    S = interpolate_fn(T, curve)
    S = max(S, 1e-9)
    H = -math.log(S)          # 경험적 누적 위험함수
    return (obs_period / T) * (1.0 + weight * H)


def _find_optimal_T(curve: dict, obs_period: float, weight: float,
                    interpolate_fn, n_sweep: int = 500) -> tuple:
    """
    실측 데이터 범위 내에서 수치적으로 최소 비용 PM 주기 탐색.
    Returns: (T_opt, min_cost, S_at_opt)
    """
    times = sorted(curve.keys())
    # 첫 번째 고장 발생 시점(S<1) 이후부터 최대 데이터 범위까지 탐색
    valid_times = [t for t, s in curve.items() if s < 1.0]
    if not valid_times:
        return None, None, None

    T_min = max(1.0, min(valid_times) * 0.5)
    T_max = max(times)

    # 로그 균등 분포로 탐색 포인트 생성
    log_min = math.log(T_min)
    log_max = math.log(T_max)
    T_sweep = [math.exp(log_min + (log_max - log_min) * i / (n_sweep - 1))
               for i in range(n_sweep)]

    best_T, best_cost = T_sweep[0], float('inf')
    for T in T_sweep:
        c = _empirical_cost(T, curve, obs_period, weight, interpolate_fn)
        if c < best_cost:
            best_cost, best_T = c, T

    # 황금 분할 탐색으로 정밀화 (±10% 범위 내)
    lo = best_T * 0.9
    hi = best_T * 1.1
    phi = (math.sqrt(5) - 1) / 2
    for _ in range(50):
        m1 = hi - phi * (hi - lo)
        m2 = lo + phi * (hi - lo)
        if _empirical_cost(m1, curve, obs_period, weight, interpolate_fn) < \
           _empirical_cost(m2, curve, obs_period, weight, interpolate_fn):
            hi = m2
        else:
            lo = m1
    T_opt = (lo + hi) / 2
    min_cost = _empirical_cost(T_opt, curve, obs_period, weight, interpolate_fn)
    S_opt = max(interpolate_fn(T_opt, curve), 1e-9)
    return T_opt, min_cost, S_opt


def optimize_pm(part_name: str, equipment_type: str,
                obs_period_months: float, weight: float) -> PMOptimizationResult:
    """
    VJ OSP 실측 생존확률 데이터 기반으로 최적 PM 주기 도출.

    Args:
        part_name: 부품명 (생존확률 데이터 딕셔너리 키와 일치)
        equipment_type: "OLD" or "NEW"
        obs_period_months: 관측/계획 기간 (월)
        weight: RM/PM 비용 가중치 (RM이 PM의 몇 배인지)
    """
    from data.survival_data import SURVIVAL_DATA_OLD, SURVIVAL_DATA_NEW, SURVIVAL_DATA_ACCEL, SURVIVAL_DATA_ALL
    from data.survival_data import interpolate_survival

    et = equipment_type.upper()
    if et == "ACCEL":
        raw = SURVIVAL_DATA_ACCEL
    elif et == "NEW":
        raw = SURVIVAL_DATA_NEW
    elif et == "ALL":
        raw = SURVIVAL_DATA_ALL
    else:
        raw = SURVIVAL_DATA_OLD

    if part_name not in raw:
        return PMOptimizationResult(
            part_name=part_name, equipment_type=equipment_type,
            obs_period_months=obs_period_months, weight=weight,
            hazard_trend_beta=0, data_range_months=0, n_data_points=0,
            strategy="INSUFFICIENT_DATA",
            strategy_reason=f"'{part_name}' ({equipment_type}) 데이터 없음"
        )

    curve = raw[part_name]
    valid_pts = [(t, s) for t, s in curve.items() if s < 1.0]
    n_pts = len(valid_pts)
    data_range = max(curve.keys())

    beta = _estimate_beta(curve)

    if beta is None or n_pts < 2:
        return PMOptimizationResult(
            part_name=part_name, equipment_type=equipment_type,
            obs_period_months=obs_period_months, weight=weight,
            hazard_trend_beta=0, data_range_months=data_range, n_data_points=n_pts,
            strategy="INSUFFICIENT_DATA",
            strategy_reason="고장 데이터 포인트 2개 미만 — β 추세 추정 불가"
        )

    WEAR_THRESHOLD   = 1.05
    RANDOM_THRESHOLD = 0.95

    if beta < RANDOM_THRESHOLD:
        return PMOptimizationResult(
            part_name=part_name, equipment_type=equipment_type,
            obs_period_months=obs_period_months, weight=weight,
            hazard_trend_beta=round(beta, 3),
            data_range_months=data_range, n_data_points=n_pts,
            strategy="RUN_TO_FAILURE",
            strategy_reason=(
                f"초기/감소 고장률 (β={beta:.3f} < 1). "
                "노화할수록 신뢰성 증가 → PM이 오히려 비용 상승. Run-To-Failure 권장."
            )
        )

    if beta < WEAR_THRESHOLD:
        return PMOptimizationResult(
            part_name=part_name, equipment_type=equipment_type,
            obs_period_months=obs_period_months, weight=weight,
            hazard_trend_beta=round(beta, 3),
            data_range_months=data_range, n_data_points=n_pts,
            strategy="RANDOM_FAILURE",
            strategy_reason=(
                f"랜덤/지수 고장률 (β={beta:.3f} ≈ 1). "
                "PM으로 고장률 감소 효과 없음. 상태 기반 점검(CBM) 권장."
            )
        )

    # β > 1: 최적 PM 주기 수치 탐색 (실측 데이터 기반)
    T_opt, min_cost, S_opt = _find_optimal_T(
        curve, obs_period_months, weight, interpolate_survival
    )

    if T_opt is None:
        return PMOptimizationResult(
            part_name=part_name, equipment_type=equipment_type,
            obs_period_months=obs_period_months, weight=weight,
            hazard_trend_beta=round(beta, 3),
            data_range_months=data_range, n_data_points=n_pts,
            strategy="INSUFFICIENT_DATA",
            strategy_reason="최적 주기 탐색 실패 (데이터 범위 부족)"
        )

    H_opt = -math.log(max(S_opt, 1e-9))
    N_PM  = obs_period_months / T_opt
    N_RM  = N_PM * H_opt

    return PMOptimizationResult(
        part_name=part_name, equipment_type=equipment_type,
        obs_period_months=obs_period_months, weight=weight,
        hazard_trend_beta=round(beta, 3),
        data_range_months=data_range, n_data_points=n_pts,
        strategy="OPTIMAL_PM",
        strategy_reason=(
            f"마모 고장 패턴 (β={beta:.3f} > 1). "
            "실측 생존확률 데이터 기반 총비용 최소화 PM 주기 도출."
        ),
        optimal_pm_months=round(T_opt, 1),
        optimal_pm_days=round(T_opt * 30.44, 0),
        optimal_survival_pct=round(S_opt * 100, 1),
        min_total_cost=round(min_cost, 4),
        n_pm_per_period=round(N_PM, 3),
        n_rm_per_period=round(N_RM, 3),
    )


def compute_cost_curve(part_name: str, equipment_type: str,
                       obs_period_months: float, weight: float,
                       n_points: int = 120) -> list:
    """
    실측 생존확률 기반 비용 곡선 데이터 생성 (차트용).
    T 범위: 첫 고장 시점 ~ 데이터 최대 시점 (실측 범위 내)
    """
    from data.survival_data import SURVIVAL_DATA_OLD, SURVIVAL_DATA_NEW, SURVIVAL_DATA_ACCEL, SURVIVAL_DATA_ALL
    from data.survival_data import interpolate_survival

    et = equipment_type.upper()
    if et == "ACCEL":
        raw = SURVIVAL_DATA_ACCEL
    elif et == "NEW":
        raw = SURVIVAL_DATA_NEW
    elif et == "ALL":
        raw = SURVIVAL_DATA_ALL
    else:
        raw = SURVIVAL_DATA_OLD
    if part_name not in raw:
        return []

    curve = raw[part_name]
    valid_times = [t for t, s in curve.items() if s < 1.0]
    if not valid_times:
        return []

    T_min = max(1.0, min(valid_times) * 0.5)
    T_max = max(curve.keys())          # ← 실측 데이터 범위로 제한

    log_min = math.log(T_min)
    log_max = math.log(T_max)
    T_values = [math.exp(log_min + (log_max - log_min) * i / (n_points - 1))
                for i in range(n_points)]

    # 최적 포인트 계산
    T_opt, _, _ = _find_optimal_T(curve, obs_period_months, weight,
                                  interpolate_survival)

    # 최적 T가 범위 내면 명시적으로 추가
    if T_opt and T_min <= T_opt <= T_max:
        T_values.append(T_opt)
        T_values.sort()

    results = []
    for T in T_values:
        S = max(interpolate_survival(T, curve), 1e-9)
        H = -math.log(S)
        N_PM  = obs_period_months / T
        N_RM  = N_PM * H
        total = N_PM + weight * N_RM
        is_opt = T_opt is not None and abs(T - T_opt) / max(T_opt, 1) < 0.005

        results.append({
            "pm_interval_months": round(T, 1),
            "pm_interval_days":   round(T * 30.44, 0),
            "survival_prob":      round(S, 4),
            "cumulative_hazard":  round(H, 4),
            "n_pm":               round(N_PM, 3),
            "n_rm":               round(N_RM, 3),
            "total_cost":         round(total, 4),
            "is_optimal":         is_opt,
        })

    return results


def optimize_all_parts(equipment_type: str, obs_period_months: float,
                       weight: float) -> list:
    """전체 부품 PM 최적화 일괄 실행."""
    from data.survival_data import SURVIVAL_DATA_OLD, SURVIVAL_DATA_NEW, SURVIVAL_DATA_ACCEL, SURVIVAL_DATA_ALL
    et = equipment_type.upper()
    if et == "ACCEL":
        raw = SURVIVAL_DATA_ACCEL
    elif et == "NEW":
        raw = SURVIVAL_DATA_NEW
    elif et == "ALL":
        raw = SURVIVAL_DATA_ALL
    else:
        raw = SURVIVAL_DATA_OLD
    return [optimize_pm(p, equipment_type, obs_period_months, weight)
            for p in sorted(raw.keys())]
