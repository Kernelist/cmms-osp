"""
PdM Multiple Classifier Predictor — Susto et al. (2015) Step 2
================================================================
실제 scikit-learn ML 분류기를 사용한 PdM 앙상블 예측.

Multiple Classifier Architecture (v4 - ML-trained):
  1. Survival Model Classifier  — 기존 생존곡선 기반 (PM 모델 계승, rule-based)
  2. Random Forest Classifier    — scikit-learn ML 학습 모델
  3. SVM Classifier              — scikit-learn ML 학습 모델
  4. Gradient Boosting Classifier — scikit-learn ML 학습 모델
  5. Ensemble                    — 가중 투표로 최종 RUL 등급 결정

RUL Class (Remaining Useful Life → 4-class 분류):
  RED    : 고장까지 30일 이내 → 즉시 교체
  ORANGE : 고장까지 30~90일  → 교체 계획
  YELLOW : 고장까지 90~180일 → 모니터링 강화
  GREEN  : 고장까지 180일 초과 → 정상 운전
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.feature_extractor import extract_features, _estimate_beta
from core.data_generator import get_feature_names, features_to_array
from core.model_trainer import load_models
from core.sensor_generator import generate_sensor_features


# ── RUL 클래스 정의 ──────────────────────────────────────────────────────────
RUL_CLASSES = ["GREEN", "YELLOW", "ORANGE", "RED"]
RUL_THRESHOLDS = {
    "RED": 30,
    "ORANGE": 90,
    "YELLOW": 180,
    "GREEN": float("inf"),
}

_THRESHOLD_MONTHS = {
    "RED": 1.0,
    "ORANGE": 3.0,
    "YELLOW": 6.0,
}


@dataclass
class ClassifierResult:
    name: str
    predicted_class: str
    confidence: float
    class_probabilities: dict = field(default_factory=dict)
    detail: str = ""
    model_type: str = "rule-based"  # "rule-based" or "ml-trained"
    accuracy: float = 0.0  # test accuracy (ML models only)


@dataclass
class PdMPrediction:
    part_name: str
    equipment_type: str
    operating_months: float

    # 앙상블 결과
    ensemble_class: str
    ensemble_confidence: float
    ensemble_probabilities: dict

    # 개별 분류기 결과
    classifiers: list
    agreement_ratio: float

    # PM 모델 대비 변화
    pm_risk_level: str
    pdm_vs_pm: str

    # 추정 RUL
    estimated_rul_months: Optional[float] = None
    estimated_rul_days: Optional[float] = None

    # Feature 벡터
    features: dict = field(default_factory=dict)

    # 모델 메타데이터
    model_info: dict = field(default_factory=dict)


# ── 분류기 구현 ──────────────────────────────────────────────────────────────

class SurvivalModelClassifier:
    """기존 PM 생존곡선 모델을 분류기로 래핑 (rule-based baseline)."""
    name = "Survival Model"
    weight = 0.15  # default; overridden by log-odds at runtime
    model_type = "rule-based"

    def predict(self, features: dict, **kwargs) -> ClassifierResult:
        surv = features["current_survival"]
        fail = features["current_failure"]

        if fail >= 0.35:
            cls = "RED"
        elif fail >= 0.20:
            cls = "ORANGE"
        elif fail >= 0.10:
            cls = "YELLOW"
        else:
            cls = "GREEN"

        probs = self._failure_to_probs(fail)
        confidence = probs[cls]

        return ClassifierResult(
            name=self.name,
            predicted_class=cls,
            confidence=confidence,
            class_probabilities=probs,
            detail=f"S(t)={surv:.3f}, F(t)={fail:.3f}",
            model_type=self.model_type,
        )

    def _failure_to_probs(self, fail: float) -> dict:
        raw = {
            "RED": self._sigmoid(fail, center=0.45, scale=15),
            "ORANGE": self._sigmoid(fail, center=0.27, scale=15)
            * (1 - self._sigmoid(fail, center=0.45, scale=15)),
            "YELLOW": self._sigmoid(fail, center=0.15, scale=15)
            * (1 - self._sigmoid(fail, center=0.27, scale=15)),
            "GREEN": 1 - self._sigmoid(fail, center=0.15, scale=15),
        }
        total = sum(raw.values())
        return {k: round(v / total, 4) for k, v in raw.items()}

    @staticmethod
    def _sigmoid(x, center=0.5, scale=10):
        return 1 / (1 + math.exp(-scale * (x - center)))


class MLClassifier:
    """scikit-learn 학습 모델 기반 분류기."""

    def __init__(self, name: str, model_key: str, weight: float):
        self.name = name
        self.model_key = model_key
        self.weight = weight
        self.model_type = "ml-trained"

    def predict(self, features: dict, feature_names: list, models: dict,
                scaler=None, metrics: dict = None) -> ClassifierResult:
        X = np.array([features_to_array(features, feature_names)])

        model = models[self.model_key]

        # SVM needs scaled input
        if self.model_key == "svm" and scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = X

        # predict_proba returns probabilities for each class
        proba = model.predict_proba(X_input)[0]
        classes = list(model.classes_)

        # Map to our RUL_CLASSES order
        class_probs = {}
        for cls in RUL_CLASSES:
            if cls in classes:
                idx = classes.index(cls)
                class_probs[cls] = round(float(proba[idx]), 4)
            else:
                class_probs[cls] = 0.0

        predicted_class = max(class_probs, key=class_probs.get)
        confidence = class_probs[predicted_class]

        # Get test accuracy from metrics
        acc = 0.0
        if metrics and self.model_key in metrics:
            acc = metrics[self.model_key].get("accuracy", 0.0)

        # Detail string with top features (for RF and GB)
        detail = self._get_detail(model, feature_names, X[0])

        return ClassifierResult(
            name=self.name,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            detail=detail,
            model_type=self.model_type,
            accuracy=acc,
        )

    def _get_detail(self, model, feature_names: list, x: np.ndarray) -> str:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-3:][::-1]
            top = [f"{feature_names[i]}={x[i]:.2f}({importances[i]:.2f})" for i in top_idx]
            return "top: " + ", ".join(top)
        return f"pred via {self.model_key}"


class SBMClassifier:
    """
    Similarity-Based Modeling 분류기 (Susto et al. 2015 핵심).
    학습된 정상 프로파일(μ, Σ)과 Mahalanobis 거리로 RUL 등급 판정.
    """
    name = "SBM (Sensor)"
    weight = 0.15  # default; overridden by log-odds
    model_type = "sensor-based"

    def predict(self, features: dict, sbm_profile: dict = None,
                feature_names: list = None, **kwargs) -> ClassifierResult:
        if sbm_profile is None or feature_names is None:
            return self._fallback_predict(features)

        # 센서 feature 추출
        sensor_names = sbm_profile["sensor_names"]
        x_sensor = np.array([features.get(s, 0.0) for s in sensor_names])

        # Mahalanobis distance
        mu = sbm_profile["mu"]
        cov_inv = sbm_profile["cov_inv"]
        diff = x_sensor - mu
        d = float(np.sqrt(diff @ cov_inv @ diff))

        # 임계값 기반 분류
        th = sbm_profile["thresholds"]
        if d <= th["green_yellow"]:
            cls = "GREEN"
        elif d <= th["yellow_orange"]:
            cls = "YELLOW"
        elif d <= th["orange_red"]:
            cls = "ORANGE"
        else:
            cls = "RED"

        # Health Index: exp(-D²/2σ²) 정규화
        d_max = sbm_profile.get("d_max_ref", 10.0)
        health_index = math.exp(-0.5 * (d / max(d_max * 0.5, 1.0)) ** 2)
        health_index = max(0.0, min(1.0, health_index))

        probs = self._distance_to_probs(d, th)
        return ClassifierResult(
            name=self.name,
            predicted_class=cls,
            confidence=probs[cls],
            class_probabilities=probs,
            detail=f"D={d:.2f}, HI={health_index:.3f}, th=[{th['green_yellow']:.1f},{th['yellow_orange']:.1f},{th['orange_red']:.1f}]",
            model_type=self.model_type,
        )

    def _distance_to_probs(self, d: float, th: dict) -> dict:
        """Mahalanobis 거리를 소프트 클래스 확률로 변환."""
        t1, t2, t3 = th["green_yellow"], th["yellow_orange"], th["orange_red"]
        raw = {
            "GREEN": math.exp(-max(d - 0, 0) / max(t1, 0.1)),
            "YELLOW": math.exp(-abs(d - (t1 + t2) / 2) / max(t2 - t1, 0.1)),
            "ORANGE": math.exp(-abs(d - (t2 + t3) / 2) / max(t3 - t2, 0.1)),
            "RED": 1 - math.exp(-max(d - t2, 0) / max(t3, 0.1)),
        }
        total = sum(raw.values()) or 1
        return {k: round(v / total, 4) for k, v in raw.items()}

    def _fallback_predict(self, features: dict) -> ClassifierResult:
        """프로파일 없을 때 기본 규칙 기반 fallback."""
        hi = features.get("sensor_health_index", 1.0)
        if hi >= 0.8:
            cls = "GREEN"
        elif hi >= 0.5:
            cls = "YELLOW"
        elif hi >= 0.3:
            cls = "ORANGE"
        else:
            cls = "RED"
        probs = {"GREEN": hi, "YELLOW": 0.0, "ORANGE": 0.0, "RED": 1 - hi}
        total = sum(probs.values()) or 1
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        return ClassifierResult(
            name=self.name, predicted_class=cls, confidence=probs[cls],
            class_probabilities=probs, detail=f"fallback HI={hi:.3f}",
            model_type=self.model_type,
        )


# ── 분류기 인스턴스 ──────────────────────────────────────────────────────────

_SURVIVAL_CLF = SurvivalModelClassifier()
_SBM_CLF = SBMClassifier()

_ML_CLASSIFIERS = [
    MLClassifier("Random Forest", "random_forest", weight=0.30),
    MLClassifier("SVM (RBF)", "svm", weight=0.25),
    MLClassifier("Gradient Boosting", "gradient_boosting", weight=0.30),
]

_ALL_CLASSIFIERS = [_SURVIVAL_CLF, _SBM_CLF] + _ML_CLASSIFIERS


# ── 앙상블 엔진 ──────────────────────────────────────────────────────────────

def _weighted_ensemble(results: list[ClassifierResult],
                       classifiers: list) -> tuple[str, float, dict]:
    """
    Susto et al. 스타일 가중 투표 앙상블.
    각 분류기의 클래스 확률 × 가중치로 합산.
    """
    class_scores = {c: 0.0 for c in RUL_CLASSES}

    for clf, result in zip(classifiers, results):
        for cls in RUL_CLASSES:
            prob = result.class_probabilities.get(cls, 0)
            class_scores[cls] += prob * clf.weight

    total = sum(class_scores.values()) or 1
    ensemble_probs = {k: round(v / total, 4) for k, v in class_scores.items()}

    best_class = max(ensemble_probs, key=ensemble_probs.get)
    best_confidence = ensemble_probs[best_class]

    return best_class, best_confidence, ensemble_probs


def _estimate_rul(features: dict, ensemble_class: str) -> Optional[float]:
    """앙상블 결과와 feature로 RUL(개월) 추정."""
    months_to_median = features.get("months_to_median_life", None)
    surv = features.get("current_survival", 1.0)

    if months_to_median and months_to_median > 0:
        return round(months_to_median * surv, 1)

    class_rul = {"RED": 0.5, "ORANGE": 2.0, "YELLOW": 4.5, "GREEN": 12.0}
    return class_rul.get(ensemble_class, 6.0)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_pdm(
    part_name: str,
    equipment_type: str,
    operating_months: float,
    installation_year: int,
    replacement_history: list[str],
    reference_date,
) -> PdMPrediction:
    """
    Multiple Classifier 앙상블 PdM 예측 (ML-trained models).
    """
    from data.survival_data import (
        SURVIVAL_DATA_OLD, SURVIVAL_DATA_NEW,
        SURVIVAL_DATA_ACCEL, SURVIVAL_DATA_ALL,
    )

    et = equipment_type.upper()
    if et == "ACCEL":
        raw = SURVIVAL_DATA_ACCEL
    elif et == "NEW":
        raw = SURVIVAL_DATA_NEW
    elif et == "ALL":
        raw = SURVIVAL_DATA_ALL
    else:
        raw = SURVIVAL_DATA_OLD

    survival_curve = raw.get(part_name, {})

    # Feature 추출
    features = extract_features(
        part_name=part_name,
        equipment_type=equipment_type,
        operating_months=operating_months,
        installation_year=installation_year,
        replacement_history=replacement_history,
        reference_date=reference_date,
        survival_curve=survival_curve,
    )
    # part_encoded 추가 (학습 데이터와 동일하게)
    features["part_encoded"] = hash(part_name) % 100 / 100.0

    # 센서 feature 생성 (Step 3a)
    health_index = features.get("current_survival", 1.0)
    beta_val = features.get("beta_estimate", 1.5)
    sensor_feats = generate_sensor_features(
        part_name=part_name,
        health_index=health_index,
        operating_months=operating_months,
        beta=beta_val,
    )
    features.update(sensor_feats)

    # 모델 로드
    trained = load_models()
    ml_models = trained["models"]
    scaler = trained["scaler"]
    feature_names = trained["feature_names"]
    metrics = trained["metrics"]
    sbm_profile = trained.get("sbm_profile")

    # Log-odds 기반 가중치 적용
    saved_weights = metrics.get("weights", {})
    if saved_weights:
        _SURVIVAL_CLF.weight = saved_weights.get("survival_model", 0.15)
        _SBM_CLF.weight = saved_weights.get("sbm", 0.15)
        for ml_clf in _ML_CLASSIFIERS:
            _key = ml_clf.model_key
            ml_clf.weight = saved_weights.get(_key, ml_clf.weight)

    # 각 분류기 예측
    results = []

    # 1. Survival Model (rule-based)
    surv_result = _SURVIVAL_CLF.predict(features)
    results.append(surv_result)

    # 2. SBM (Mahalanobis sensor-based)
    sbm_result = _SBM_CLF.predict(
        features, sbm_profile=sbm_profile, feature_names=feature_names
    )
    results.append(sbm_result)

    # 3~5. ML classifiers
    for ml_clf in _ML_CLASSIFIERS:
        ml_result = ml_clf.predict(
            features,
            feature_names=feature_names,
            models=ml_models,
            scaler=scaler,
            metrics=metrics,
        )
        results.append(ml_result)

    # 앙상블
    ensemble_class, ensemble_conf, ensemble_probs = _weighted_ensemble(
        results, _ALL_CLASSIFIERS
    )

    # 분류기 간 일치율
    predicted_classes = [r.predicted_class for r in results]
    most_common = max(set(predicted_classes), key=predicted_classes.count)
    agreement = predicted_classes.count(most_common) / len(predicted_classes)

    # 기존 PM 모델 판정
    pm_class = results[0].predicted_class

    # PM vs PdM 비교
    risk_order = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
    if risk_order[ensemble_class] > risk_order[pm_class]:
        pdm_vs_pm = "UPGRADED"
    elif risk_order[ensemble_class] < risk_order[pm_class]:
        pdm_vs_pm = "DOWNGRADED"
    else:
        pdm_vs_pm = "SAME"

    # RUL 추정
    rul_months = _estimate_rul(features, ensemble_class)
    rul_days = round(rul_months * 30.44, 0) if rul_months else None

    # 모델 메타정보
    model_info = {
        "ml_models": ["Random Forest", "SVM (RBF)", "Gradient Boosting"],
        "baseline": "Survival Model (rule-based)",
        "training_samples": "synthetic from CMMS survival curves",
        "accuracies": {
            k: v.get("accuracy", 0) for k, v in metrics.items()
        },
    }

    return PdMPrediction(
        part_name=part_name,
        equipment_type=equipment_type,
        operating_months=round(operating_months, 1),
        ensemble_class=ensemble_class,
        ensemble_confidence=round(ensemble_conf, 4),
        ensemble_probabilities=ensemble_probs,
        classifiers=[
            {
                "name": r.name,
                "predicted_class": r.predicted_class,
                "confidence": r.confidence,
                "class_probabilities": r.class_probabilities,
                "detail": r.detail,
                "weight": clf.weight,
                "model_type": r.model_type,
                "accuracy": r.accuracy,
            }
            for clf, r in zip(_ALL_CLASSIFIERS, results)
        ],
        agreement_ratio=round(agreement, 2),
        pm_risk_level=pm_class,
        pdm_vs_pm=pdm_vs_pm,
        estimated_rul_months=rul_months,
        estimated_rul_days=rul_days,
        features=features,
        model_info=model_info,
    )


def predict_all_parts(reference_date) -> list[dict]:
    """전체 부품에 대해 PdM 예측 실행."""
    import json
    from pathlib import Path
    from datetime import datetime

    data_path = Path(__file__).parent.parent / "data" / "sample_cmms_data.json"
    with open(data_path) as f:
        cmms_data = json.load(f)

    predictions = []
    for machine in cmms_data["machines"]:
        for part in machine["parts"]:
            last_repl = datetime.strptime(
                part["last_replacement_date"], "%Y-%m-%d"
            ).date()
            op_months = (reference_date - last_repl).days / 30.4375

            history = part.get("replacement_history", [part["last_replacement_date"]])

            pred = predict_pdm(
                part_name=part["part_name"],
                equipment_type=machine["equipment_type"],
                operating_months=max(0, op_months),
                installation_year=machine["installation_year"],
                replacement_history=history,
                reference_date=reference_date,
            )

            predictions.append({
                "machine_id": machine["machine_id"],
                "machine_name": machine["machine_name"],
                "part_id": part["part_id"],
                "part_name": part["part_name"],
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
                "model_info": pred.model_info,
            })

    return predictions
