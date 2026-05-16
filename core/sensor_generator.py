"""
Synthetic Sensor Data Generator (Step 3a)
==========================================
생존곡선 기반 열화 프로파일에서 합성 센서 신호를 역생성.

센서 유형:
  - Vibration: RMS, Peak, Kurtosis, Crest Factor
  - Temperature: Mean, Trend (slope)
  - Current: Mean, Std (fluctuation)

열화 모델:
  health_index = S(t) → 1(정상) ~ 0(고장 임박)
  sensor_value = baseline + degradation_effect × (1 - health_index)^gamma + noise

SBM (Similarity-Based Modeling) feature:
  - 정상 상태 프로파일 대비 잔차(residual)
  - Health Index 종합 점수
  - 열화 속도 (health index 변화율)
"""
import math
import random


# ── 부품별 센서 프로파일 ─────────────────────────────────────────────────────
# baseline: 정상 상태 센서값, peak: 고장 임박 시 센서값
SENSOR_PROFILES = {
    # 회전/기계 부품 → 진동 우세
    "Motor": {"vib_base": 2.0, "vib_peak": 12.0, "temp_base": 45, "temp_peak": 85, "cur_base": 5.0, "cur_peak": 8.0},
    "Bearing": {"vib_base": 1.5, "vib_peak": 15.0, "temp_base": 40, "temp_peak": 90, "cur_base": 3.0, "cur_peak": 5.0},
    "Pump": {"vib_base": 3.0, "vib_peak": 18.0, "temp_base": 50, "temp_peak": 80, "cur_base": 8.0, "cur_peak": 12.0},
    # 전기/전자 부품 → 온도/전류 우세
    "Inverter": {"vib_base": 0.5, "vib_peak": 3.0, "temp_base": 55, "temp_peak": 95, "cur_base": 10.0, "cur_peak": 18.0},
    "Electronic Board": {"vib_base": 0.2, "vib_peak": 1.0, "temp_base": 50, "temp_peak": 85, "cur_base": 2.0, "cur_peak": 4.0},
    "Solid State Relay": {"vib_base": 0.1, "vib_peak": 0.5, "temp_base": 45, "temp_peak": 90, "cur_base": 3.0, "cur_peak": 6.0},
    "Magnetic Contactor": {"vib_base": 0.8, "vib_peak": 4.0, "temp_base": 40, "temp_peak": 75, "cur_base": 5.0, "cur_peak": 9.0},
    # 센서/측정 → 낮은 진동, 온도 민감
    "Temperature Sensor": {"vib_base": 0.1, "vib_peak": 0.3, "temp_base": 35, "temp_peak": 60, "cur_base": 0.5, "cur_peak": 1.0},
    "Pressure Sensor": {"vib_base": 0.2, "vib_peak": 0.8, "temp_base": 35, "temp_peak": 55, "cur_base": 0.5, "cur_peak": 1.2},
    # 공압/유압 → 진동+압력
    "Air Cylinder": {"vib_base": 2.5, "vib_peak": 10.0, "temp_base": 35, "temp_peak": 60, "cur_base": 0.0, "cur_peak": 0.0},
    "Solenoid Valve": {"vib_base": 1.0, "vib_peak": 5.0, "temp_base": 40, "temp_peak": 70, "cur_base": 2.0, "cur_peak": 4.0},
}

# 기본 프로파일 (매칭 안 되는 부품)
_DEFAULT_PROFILE = {"vib_base": 1.0, "vib_peak": 8.0, "temp_base": 45, "temp_peak": 80, "cur_base": 3.0, "cur_peak": 6.0}


def _get_profile(part_name: str) -> dict:
    """부품명에서 센서 프로파일 매칭."""
    for key, profile in SENSOR_PROFILES.items():
        if key.lower() in part_name.lower():
            return profile
    return _DEFAULT_PROFILE


def generate_sensor_features(
    part_name: str,
    health_index: float,
    operating_months: float,
    beta: float = 1.5,
) -> dict:
    """
    열화 상태(health_index)에서 합성 센서 feature 생성.

    Args:
        part_name: 부품명 (센서 프로파일 매칭)
        health_index: 0(고장) ~ 1(정상), 보통 S(t)
        operating_months: 운전 개월수
        beta: Weibull β (열화 곡선 형상)

    Returns:
        센서 feature dictionary
    """
    profile = _get_profile(part_name)
    degradation = 1.0 - max(min(health_index, 1.0), 0.0)

    # 열화 곡선: β에 따라 급격/완만 변화
    gamma = max(beta, 0.5)  # β가 클수록 후반부 급격 열화
    deg_effect = degradation ** gamma

    # 노이즈 스케일: 열화 진행될수록 변동성 증가
    noise_scale = 1.0 + degradation * 2.0

    # ── Vibration features ──
    vib_base = profile["vib_base"]
    vib_range = profile["vib_peak"] - vib_base

    vib_rms = vib_base + vib_range * deg_effect + random.gauss(0, vib_base * 0.1 * noise_scale)
    vib_rms = max(vib_rms, 0.01)

    vib_peak = vib_rms * (2.0 + degradation * 3.0) + random.gauss(0, vib_base * 0.15 * noise_scale)
    vib_peak = max(vib_peak, vib_rms)

    # Kurtosis: 정상=3(가우시안), 고장 임박 시 6~15
    vib_kurtosis = 3.0 + degradation ** 2 * 12.0 + random.gauss(0, 0.5 * noise_scale)
    vib_kurtosis = max(vib_kurtosis, 1.5)

    # Crest Factor: Peak / RMS
    vib_crest = vib_peak / max(vib_rms, 0.01)

    # ── Temperature features ──
    temp_base = profile["temp_base"]
    temp_range = profile["temp_peak"] - temp_base

    temp_mean = temp_base + temp_range * deg_effect * 0.8 + random.gauss(0, 2.0 * noise_scale)

    # 온도 추세: 열화 시 양의 기울기 (°C/month)
    temp_trend = temp_range * deg_effect * 0.05 + random.gauss(0, 0.2)
    if operating_months > 0:
        temp_trend += (temp_mean - temp_base) / max(operating_months, 1) * 0.3

    # ── Current features ──
    cur_base = profile["cur_base"]
    cur_range = profile["cur_peak"] - cur_base

    if cur_base > 0:
        cur_mean = cur_base + cur_range * deg_effect * 0.6 + random.gauss(0, cur_base * 0.08 * noise_scale)
        cur_mean = max(cur_mean, 0.0)
        cur_std = cur_base * 0.05 * (1 + degradation * 4.0) + random.gauss(0, cur_base * 0.02)
        cur_std = max(cur_std, 0.0)
    else:
        cur_mean = 0.0
        cur_std = 0.0

    # ── SBM (Similarity-Based Modeling) features ──
    # 정상 프로파일 대비 잔차
    vib_residual = abs(vib_rms - vib_base) / max(vib_range, 0.1)
    temp_residual = abs(temp_mean - temp_base) / max(temp_range, 1.0)
    cur_residual = abs(cur_mean - cur_base) / max(cur_range, 0.1) if cur_range > 0 else 0.0

    # 종합 잔차 (가중 평균)
    sensor_residual = (vib_residual * 0.4 + temp_residual * 0.35 + cur_residual * 0.25)

    # Health Index (센서 기반): 0(고장)~1(정상)
    sensor_health_index = max(0.0, 1.0 - sensor_residual)

    # 열화 속도 추정 (operating_months 기반)
    if operating_months > 1:
        sensor_deg_rate = (1.0 - sensor_health_index) / operating_months
    else:
        sensor_deg_rate = 0.0

    return {
        # Vibration
        "vibration_rms": round(vib_rms, 4),
        "vibration_peak": round(vib_peak, 4),
        "vibration_kurtosis": round(vib_kurtosis, 4),
        "vibration_crest_factor": round(vib_crest, 4),
        # Temperature
        "temperature_mean": round(temp_mean, 2),
        "temperature_trend": round(temp_trend, 4),
        # Current
        "current_mean": round(cur_mean, 4),
        "current_std": round(cur_std, 4),
        # SBM
        "sensor_health_index": round(sensor_health_index, 4),
        "sensor_residual": round(sensor_residual, 4),
        "sensor_degradation_rate": round(sensor_deg_rate, 6),
    }


def get_sensor_feature_names() -> list[str]:
    """센서 feature 이름 목록."""
    return [
        "vibration_rms",
        "vibration_peak",
        "vibration_kurtosis",
        "vibration_crest_factor",
        "temperature_mean",
        "temperature_trend",
        "current_mean",
        "current_std",
        "sensor_health_index",
        "sensor_residual",
        "sensor_degradation_rate",
    ]
