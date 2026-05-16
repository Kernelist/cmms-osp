"""
ML Model Trainer for PdM Multiple Classifier (Step 2)
=====================================================
Susto et al. (2015) 기반 실제 ML 분류기 학습.

학습 파이프라인:
  1. data_generator에서 합성 학습 데이터 생성
  2. RandomForest, SVM, GradientBoosting 3개 분류기 학습
  3. 학습된 모델을 joblib으로 저장
  4. 예측 시 로드하여 사용
"""
import math
import os
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from core.data_generator import (
    generate_training_data,
    get_feature_names,
    features_to_array,
)
from core.sensor_generator import get_sensor_feature_names

MODEL_DIR = Path(__file__).parent.parent / "models"
LABEL_ORDER = ["GREEN", "YELLOW", "ORANGE", "RED"]


def train_all_models(
    n_samples_per_part: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    전체 ML 모델 학습 파이프라인.

    Returns:
        dict with keys: models, scaler, feature_names, metrics
    """
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. 합성 데이터 생성
    if verbose:
        print("[1/4] Generating synthetic training data...")
    feature_dicts, labels = generate_training_data(
        n_samples_per_part=n_samples_per_part, seed=seed
    )
    feature_names = get_feature_names()

    X = np.array([features_to_array(f, feature_names) for f in feature_dicts])
    y = np.array(labels)

    if verbose:
        print(f"  Total samples: {len(X)}")
        for cls in LABEL_ORDER:
            print(f"    {cls}: {(y == cls).sum()}")

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 3. Feature scaling (SVM에 필요)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 모델 학습
    models = {}
    metrics = {}

    # Random Forest
    if verbose:
        print("\n[2/4] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    models["random_forest"] = rf
    metrics["random_forest"] = {
        "accuracy": round(rf_acc, 4),
        "report": classification_report(y_test, rf_pred, output_dict=True, zero_division=0),
    }
    if verbose:
        print(f"  Accuracy: {rf_acc:.4f}")

    # SVM (with probability calibration)
    if verbose:
        print("\n[3/4] Training SVM...")
    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    models["svm"] = svm
    metrics["svm"] = {
        "accuracy": round(svm_acc, 4),
        "report": classification_report(y_test, svm_pred, output_dict=True, zero_division=0),
    }
    if verbose:
        print(f"  Accuracy: {svm_acc:.4f}")

    # Gradient Boosting
    if verbose:
        print("\n[4/4] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=seed,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    models["gradient_boosting"] = gb
    metrics["gradient_boosting"] = {
        "accuracy": round(gb_acc, 4),
        "report": classification_report(y_test, gb_pred, output_dict=True, zero_division=0),
    }
    if verbose:
        print(f"  Accuracy: {gb_acc:.4f}")

    # 5. Survival Model baseline 정확도 평가 (test set에서)
    if verbose:
        print("\n[+] Evaluating Survival Model baseline on test set...")
    sm_acc = _evaluate_survival_model(X_test, y_test, feature_names)
    metrics["survival_model"] = {"accuracy": round(sm_acc, 4)}
    if verbose:
        print(f"  Accuracy: {sm_acc:.4f}")

    # 5b. SBM 정상 프로파일 학습 (Mahalanobis distance)
    if verbose:
        print("\n[+] Learning SBM normal profile from GREEN samples...")
    sbm_profile = _learn_sbm_profile(X_train, y_train, feature_names)
    joblib.dump(sbm_profile, MODEL_DIR / "sbm_profile.pkl")
    if verbose:
        print(f"  Normal samples used: {sbm_profile['n_normal']}")
        print(f"  Sensor features: {len(sbm_profile['sensor_indices'])}")
        print(f"  Distance thresholds: " + ", ".join(
            f"{k}={v:.2f}" for k, v in sbm_profile['thresholds'].items()))

    if verbose:
        print("\n[+] Evaluating SBM (Mahalanobis) on test set...")
    sbm_acc = _evaluate_sbm_mahalanobis(X_test, y_test, sbm_profile)
    metrics["sbm"] = {"accuracy": round(sbm_acc, 4)}
    if verbose:
        print(f"  Accuracy: {sbm_acc:.4f}")

    # 6. Log-odds 기반 가중치 자동 도출 (Susto et al. 방법)
    all_accuracies = {
        "survival_model": sm_acc,
        "sbm": sbm_acc,
        "random_forest": rf_acc,
        "svm": svm_acc,
        "gradient_boosting": gb_acc,
    }
    weights = _compute_log_odds_weights(all_accuracies)
    metrics["weights"] = weights

    if verbose:
        print("\n[*] Log-odds derived weights:")
        for name, w in weights.items():
            acc = all_accuracies[name]
            log_odd = math.log(max(acc, 0.01) / max(1 - acc, 0.01))
            print(f"  {name:25s}  acc={acc:.4f}  log-odds={log_odd:.4f}  W={w:.4f}")

    # 7. 모델 저장
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")
    joblib.dump(svm, MODEL_DIR / "svm.pkl")
    joblib.dump(gb, MODEL_DIR / "gradient_boosting.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)

    if verbose:
        print(f"\nModels saved to {MODEL_DIR}/")
        print(f"  Survival Model:     {sm_acc:.4f}  (W={weights['survival_model']:.4f})")
        print(f"  SBM (Sensor):       {sbm_acc:.4f}  (W={weights['sbm']:.4f})")
        print(f"  Random Forest:      {rf_acc:.4f}  (W={weights['random_forest']:.4f})")
        print(f"  SVM:                {svm_acc:.4f}  (W={weights['svm']:.4f})")
        print(f"  Gradient Boosting:  {gb_acc:.4f}  (W={weights['gradient_boosting']:.4f})")

    return {
        "models": models,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": metrics,
    }


def _compute_log_odds_weights(accuracies: dict) -> dict:
    """
    Log-odds 변환으로 분류기 가중치 도출 (Susto et al. 2015).

    W_i = log(Acc_i / (1 - Acc_i))
    정규화하여 합이 1이 되도록.
    """
    raw = {}
    for name, acc in accuracies.items():
        acc_clipped = max(min(acc, 0.999), 0.501)  # 0.5 이하면 찬스 수준 → 최소 보정
        raw[name] = math.log(acc_clipped / (1 - acc_clipped))

    total = sum(raw.values())
    return {name: round(v / total, 4) for name, v in raw.items()}


def _evaluate_survival_model(X_test, y_test, feature_names: list) -> float:
    """
    Survival Model (rule-based)의 test set 정확도 평가.
    동일한 test set에서 F(t) 기반 규칙으로 예측하여 정확도를 측정.
    """
    fail_idx = feature_names.index("current_failure")
    correct = 0
    for x, true_label in zip(X_test, y_test):
        fail = x[fail_idx]
        if fail >= 0.35:
            pred = "RED"
        elif fail >= 0.20:
            pred = "ORANGE"
        elif fail >= 0.10:
            pred = "YELLOW"
        else:
            pred = "GREEN"
        if pred == true_label:
            correct += 1
    return correct / len(y_test)


def _learn_sbm_profile(X_train, y_train, feature_names: list) -> dict:
    """
    SBM 정상 프로파일 학습.
    GREEN 샘플의 센서 feature에서 평균(μ)과 공분산(Σ)을 계산.
    Mahalanobis 거리 임계값을 분위수에서 자동 도출.
    """
    sensor_names = get_sensor_feature_names()
    sensor_indices = [feature_names.index(s) for s in sensor_names if s in feature_names]

    # GREEN 샘플만 추출
    green_mask = y_train == "GREEN"
    X_green_sensor = X_train[green_mask][:, sensor_indices]

    # 정상 프로파일: 평균 + 공분산
    mu = np.mean(X_green_sensor, axis=0)
    cov = np.cov(X_green_sensor, rowvar=False)
    # 정칙화 (특이행렬 방지)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    # 전체 학습 데이터에서 Mahalanobis 거리 계산
    X_all_sensor = X_train[:, sensor_indices]
    distances = np.array([
        _mahalanobis(x, mu, cov_inv) for x in X_all_sensor
    ])

    # 클래스별 거리 분포에서 임계값 도출
    d_green = distances[y_train == "GREEN"]
    d_yellow = distances[y_train == "YELLOW"]
    d_orange = distances[y_train == "ORANGE"]
    d_red = distances[y_train == "RED"]

    thresholds = {
        "green_yellow": float(np.percentile(d_green, 90)),
        "yellow_orange": float(np.percentile(d_yellow, 75) if len(d_yellow) > 0 else np.percentile(d_green, 95)),
        "orange_red": float(np.percentile(d_orange, 75) if len(d_orange) > 0 else np.percentile(d_green, 99)),
    }

    return {
        "mu": mu,
        "cov_inv": cov_inv,
        "sensor_indices": sensor_indices,
        "sensor_names": [feature_names[i] for i in sensor_indices],
        "thresholds": thresholds,
        "n_normal": int(green_mask.sum()),
        "d_max_ref": float(np.percentile(distances, 99)),
    }


def _mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """Mahalanobis distance: D = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))"""
    diff = x - mu
    return float(np.sqrt(diff @ cov_inv @ diff))


def _evaluate_sbm_mahalanobis(X_test, y_test, sbm_profile: dict) -> float:
    """Mahalanobis 기반 SBM의 test set 정확도."""
    sensor_idx = sbm_profile["sensor_indices"]
    mu = sbm_profile["mu"]
    cov_inv = sbm_profile["cov_inv"]
    th = sbm_profile["thresholds"]

    correct = 0
    for x, true_label in zip(X_test, y_test):
        x_sensor = x[sensor_idx]
        d = _mahalanobis(x_sensor, mu, cov_inv)

        if d <= th["green_yellow"]:
            pred = "GREEN"
        elif d <= th["yellow_orange"]:
            pred = "YELLOW"
        elif d <= th["orange_red"]:
            pred = "ORANGE"
        else:
            pred = "RED"

        if pred == true_label:
            correct += 1
    return correct / len(y_test)


# ── Cached model loader ──────────────────────────────────────────────────────
_cached_models = None


def load_models() -> dict:
    """학습된 모델 로드 (캐시). 없으면 자동 학습."""
    global _cached_models
    if _cached_models is not None:
        return _cached_models

    rf_path = MODEL_DIR / "random_forest.pkl"
    if not rf_path.exists():
        print("No trained models found. Training now...")
        result = train_all_models(verbose=True)
        _cached_models = result
        return _cached_models

    sbm_path = MODEL_DIR / "sbm_profile.pkl"
    _cached_models = {
        "models": {
            "random_forest": joblib.load(MODEL_DIR / "random_forest.pkl"),
            "svm": joblib.load(MODEL_DIR / "svm.pkl"),
            "gradient_boosting": joblib.load(MODEL_DIR / "gradient_boosting.pkl"),
        },
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "feature_names": joblib.load(MODEL_DIR / "feature_names.pkl"),
        "sbm_profile": joblib.load(sbm_path) if sbm_path.exists() else None,
        "metrics": _load_metrics(),
    }
    return _cached_models


def _load_metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    train_all_models(verbose=True)
