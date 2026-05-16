"""
Feature Extractor for PdM Multiple Classifier
===============================================
CMMS 이력 데이터에서 ML 분류기용 feature를 추출.

Susto et al. (2015) 접근법 적용:
  - 시간 기반 feature (운전 개월수, 장비 나이)
  - 이력 기반 feature (과거 교체 패턴, 평균 수명, 편차)
  - 생존 모델 feature (현재 생존확률, 위험률, β)
  - 상대적 feature (평균 수명 대비 현재 운전 비율)
"""
import math
from datetime import date, datetime
from typing import Optional


def _months_between(start: date, end: date) -> float:
    return (end - start).days / 30.4375


def extract_features(
    part_name: str,
    equipment_type: str,
    operating_months: float,
    installation_year: int,
    replacement_history: list[str],
    reference_date: date,
    survival_curve: dict,
    beta: Optional[float] = None,
) -> dict:
    """
    단일 부품에 대한 PdM feature 벡터 추출.

    Args:
        part_name: 부품명
        equipment_type: "OLD" / "NEW"
        operating_months: 마지막 교체 이후 운전 개월수
        installation_year: 장비 설치 연도
        replacement_history: 과거 교체일 리스트 ["YYYY-MM-DD", ...]
        reference_date: 기준일
        survival_curve: {month: survival_prob} 생존곡선 데이터
        beta: Weibull β 추정치 (있으면 사용, 없으면 계산)

    Returns:
        Feature dictionary
    """
    # ── 시간 기반 feature ──
    equipment_age_years = reference_date.year - installation_year
    equipment_type_encoded = 1 if equipment_type.upper() == "NEW" else 0

    # ── 이력 기반 feature ──
    n_past_replacements = len(replacement_history)
    past_lifetimes = _compute_past_lifetimes(replacement_history, installation_year)

    if len(past_lifetimes) >= 2:
        mean_past_lifetime = sum(past_lifetimes) / len(past_lifetimes)
        std_past_lifetime = (
            sum((x - mean_past_lifetime) ** 2 for x in past_lifetimes)
            / len(past_lifetimes)
        ) ** 0.5
        last_lifetime = past_lifetimes[-1]
        last_lifetime_ratio = last_lifetime / mean_past_lifetime if mean_past_lifetime > 0 else 1.0
        cv_lifetime = std_past_lifetime / mean_past_lifetime if mean_past_lifetime > 0 else 0.0
    elif len(past_lifetimes) == 1:
        mean_past_lifetime = past_lifetimes[0]
        std_past_lifetime = 0.0
        last_lifetime = past_lifetimes[0]
        last_lifetime_ratio = 1.0
        cv_lifetime = 0.0
    else:
        mean_past_lifetime = 0.0
        std_past_lifetime = 0.0
        last_lifetime = 0.0
        last_lifetime_ratio = 1.0
        cv_lifetime = 0.0

    # 현재 운전시간 / 평균수명 비율
    age_ratio = (
        operating_months / mean_past_lifetime
        if mean_past_lifetime > 0
        else operating_months / 60.0  # fallback: 5년 기준
    )

    # ── 생존 모델 feature ──
    current_survival = _interpolate(operating_months, survival_curve)
    current_failure = 1.0 - current_survival
    current_hazard = -math.log(max(current_survival, 1e-9))

    # 위험률 변화율 (현재 시점 근처의 기울기)
    delta = 1.0
    s_before = _interpolate(max(0, operating_months - delta), survival_curve)
    s_after = _interpolate(operating_months + delta, survival_curve)
    h_before = -math.log(max(s_before, 1e-9))
    h_after = -math.log(max(s_after, 1e-9))
    hazard_slope = (h_after - h_before) / (2 * delta)

    # β 추정치
    if beta is None:
        beta = _estimate_beta(survival_curve)
    beta_value = beta if beta is not None else 1.0

    # ── 예측 보조 feature ──
    # 생존확률이 50%에 도달하는 시점 추정
    median_life = _find_threshold_month(survival_curve, 0.5)
    # 생존확률이 80%에 도달하는 시점 추정
    life_80pct = _find_threshold_month(survival_curve, 0.8)

    months_to_median = (
        max(0, median_life - operating_months) if median_life else 0.0
    )
    months_to_80pct = (
        max(0, life_80pct - operating_months) if life_80pct else 0.0
    )

    return {
        # 시간 기반
        "operating_months": round(operating_months, 2),
        "equipment_age_years": equipment_age_years,
        "equipment_type_encoded": equipment_type_encoded,
        # 이력 기반
        "n_past_replacements": n_past_replacements,
        "mean_past_lifetime": round(mean_past_lifetime, 2),
        "std_past_lifetime": round(std_past_lifetime, 2),
        "cv_lifetime": round(cv_lifetime, 4),
        "last_lifetime_ratio": round(last_lifetime_ratio, 4),
        "age_ratio": round(age_ratio, 4),
        # 생존 모델 기반
        "current_survival": round(current_survival, 6),
        "current_failure": round(current_failure, 6),
        "current_hazard": round(current_hazard, 6),
        "hazard_slope": round(hazard_slope, 6),
        "beta_estimate": round(beta_value, 4),
        # 예측 보조
        "months_to_median_life": round(months_to_median, 2),
        "months_to_80pct_surv": round(months_to_80pct, 2),
    }


def _compute_past_lifetimes(
    replacement_dates: list[str], installation_year: int
) -> list[float]:
    """교체 이력에서 각 교체 간 수명(개월)을 계산."""
    if not replacement_dates:
        return []

    dates = sorted(
        datetime.strptime(d, "%Y-%m-%d").date() for d in replacement_dates
    )
    # 첫 교체: 설치일로부터의 수명
    install_date = date(installation_year, 1, 1)
    lifetimes = [_months_between(install_date, dates[0])]

    for i in range(1, len(dates)):
        lt = _months_between(dates[i - 1], dates[i])
        if lt > 0:
            lifetimes.append(lt)

    return lifetimes


def _interpolate(month: float, curve: dict) -> float:
    """생존곡선에서 선형 보간."""
    if not curve:
        return 1.0
    times = sorted(curve.keys())
    if month <= times[0]:
        return curve[times[0]]
    if month >= times[-1]:
        return max(curve[times[-1]], 1e-9)

    for i in range(len(times) - 1):
        if times[i] <= month <= times[i + 1]:
            t0, t1 = times[i], times[i + 1]
            s0, s1 = curve[t0], curve[t1]
            ratio = (month - t0) / (t1 - t0)
            return s0 + (s1 - s0) * ratio
    return curve[times[-1]]


def _estimate_beta(curve: dict) -> Optional[float]:
    """ln(H) vs ln(T) 회귀로 β 추정."""
    valid = [(t, s) for t, s in sorted(curve.items()) if t > 0 and 0 < s < 1.0]
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


def _find_threshold_month(curve: dict, threshold_survival: float) -> Optional[float]:
    """생존확률이 threshold 이하로 떨어지는 시점(월) 찾기."""
    times = sorted(curve.keys())
    for i, t in enumerate(times):
        if curve[t] <= threshold_survival:
            if i == 0:
                return t
            t0, t1 = times[i - 1], t
            s0, s1 = curve[t0], curve[t1]
            if abs(s1 - s0) < 1e-9:
                return t
            ratio = (threshold_survival - s0) / (s1 - s0)
            return t0 + ratio * (t1 - t0)
    return None
