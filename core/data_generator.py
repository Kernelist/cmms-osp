"""
Synthetic Training Data Generator for PdM Multiple Classifier
==============================================================
기존 CMMS 생존곡선 데이터를 기반으로 ML 분류기 학습용 합성 데이터를 생성.

전략:
  1. 각 부품의 생존곡선에서 고장 시점(TTF) 샘플링
  2. TTF 이전의 다양한 운전 시점에서 feature 스냅샷 생성
  3. 각 스냅샷에 RUL 클래스 라벨 부여
  4. 장비 나이, 교체 이력 등 공변량을 랜덤 변동으로 다양화
"""
import math
import random
from datetime import date, timedelta

from core.feature_extractor import extract_features, _interpolate, _estimate_beta
from core.sensor_generator import generate_sensor_features, get_sensor_feature_names

# RUL 클래스 라벨 (월 기준)
RUL_CLASS_THRESHOLDS = [
    ("RED", 1.0),       # 30일 이내
    ("ORANGE", 3.0),    # 90일 이내
    ("YELLOW", 6.0),    # 180일 이내
    ("GREEN", float("inf")),
]


def _rul_to_class(rul_months: float) -> str:
    for cls, threshold in RUL_CLASS_THRESHOLDS:
        if rul_months <= threshold:
            return cls
    return "GREEN"


def _sample_ttf(curve: dict, n_samples: int = 10) -> list[float]:
    """
    생존곡선에서 고장 시점(Time-To-Failure)을 역변환 샘플링.
    S(t)가 감소하는 구간에서 균등 분포로 생존확률을 샘플링한 뒤
    해당 확률에 대응하는 시간을 보간.
    """
    times = sorted(curve.keys())
    if not times:
        return []

    s_min = max(curve[times[-1]], 0.01)
    s_max = min(curve[times[0]], 1.0)

    ttfs = []
    for _ in range(n_samples):
        # 균등 분포로 생존확률 샘플링
        target_s = random.uniform(s_min, s_max)
        # 보간으로 시점 찾기
        for i in range(len(times) - 1):
            s0 = curve[times[i]]
            s1 = curve[times[i + 1]]
            if s1 <= target_s <= s0:
                ratio = (s0 - target_s) / max(s0 - s1, 1e-9)
                t = times[i] + ratio * (times[i + 1] - times[i])
                # 약간의 노이즈 추가
                t *= random.uniform(0.85, 1.15)
                ttfs.append(max(t, 0.5))
                break
        else:
            ttfs.append(times[-1] * random.uniform(0.7, 1.0))

    return ttfs


def generate_training_data(
    n_samples_per_part: int = 60,
    seed: int = 42,
) -> tuple[list[dict], list[str]]:
    """
    전체 부품에 대해 합성 학습 데이터 생성.

    Returns:
        (feature_dicts_list, labels_list)
    """
    random.seed(seed)

    from data.survival_data import SURVIVAL_DATA_OLD, SURVIVAL_DATA_NEW

    all_features = []
    all_labels = []

    for eq_type, raw_data in [("OLD", SURVIVAL_DATA_OLD), ("NEW", SURVIVAL_DATA_NEW)]:
        for part_name, curve in raw_data.items():
            if not curve:
                continue

            beta = _estimate_beta(curve)
            ttfs = _sample_ttf(curve, n_samples=n_samples_per_part)

            for ttf in ttfs:
                # TTF 이전의 다양한 시점에서 스냅샷 생성
                n_snapshots = random.randint(3, 6)
                snapshot_ratios = sorted(random.uniform(0.05, 0.99) for _ in range(n_snapshots))

                for ratio in snapshot_ratios:
                    operating_months = ttf * ratio
                    rul_months = ttf - operating_months

                    # 공변량 랜덤화
                    install_year = random.choice(
                        range(1998, 2008) if eq_type == "OLD" else range(2018, 2024)
                    )
                    ref_date = date(2024, 2, 1)
                    equip_age = ref_date.year - install_year

                    # 과거 교체 이력 생성 (1~4회)
                    n_replacements = random.randint(1, 4)
                    replacement_dates = []
                    base_date = date(install_year, 1, 1)
                    for r in range(n_replacements):
                        gap_months = random.uniform(6, ttf * 1.2)
                        gap_days = int(gap_months * 30.44)
                        repl_date = base_date + timedelta(days=gap_days)
                        if repl_date < ref_date:
                            replacement_dates.append(repl_date.isoformat())
                            base_date = repl_date

                    if not replacement_dates:
                        replacement_dates = [
                            (ref_date - timedelta(days=int(operating_months * 30.44))).isoformat()
                        ]

                    features = extract_features(
                        part_name=part_name,
                        equipment_type=eq_type,
                        operating_months=operating_months,
                        installation_year=install_year,
                        replacement_history=replacement_dates,
                        reference_date=ref_date,
                        survival_curve=curve,
                        beta=beta,
                    )

                    # 부품 유형 인코딩 추가
                    features["part_encoded"] = hash(part_name) % 100 / 100.0

                    # 센서 feature 생성 (Step 3a)
                    health_index = _interpolate(operating_months, curve)
                    sensor_feats = generate_sensor_features(
                        part_name=part_name,
                        health_index=health_index,
                        operating_months=operating_months,
                        beta=beta if beta else 1.5,
                    )
                    features.update(sensor_feats)

                    label = _rul_to_class(rul_months)
                    all_features.append(features)
                    all_labels.append(label)

    return all_features, all_labels


def get_feature_names() -> list[str]:
    """ML 모델에 사용할 feature 이름 목록 (CMMS 17 + Sensor 11 = 28)."""
    return [
        # CMMS features (17)
        "operating_months",
        "equipment_age_years",
        "equipment_type_encoded",
        "n_past_replacements",
        "mean_past_lifetime",
        "std_past_lifetime",
        "cv_lifetime",
        "last_lifetime_ratio",
        "age_ratio",
        "current_survival",
        "current_failure",
        "current_hazard",
        "hazard_slope",
        "beta_estimate",
        "months_to_median_life",
        "months_to_80pct_surv",
        "part_encoded",
        # Sensor features (11) — Step 3a
    ] + get_sensor_feature_names()


def features_to_array(features: dict, feature_names: list[str]) -> list[float]:
    """Feature dict를 ML 모델 입력 배열로 변환."""
    return [features.get(name, 0.0) for name in feature_names]
