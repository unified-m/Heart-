"""
Machine Learning Risk Prediction Module.
Trains and loads a temporary ML model for cardiovascular risk prediction
based on HRV features extracted from heart rate data.
"""

import logging
import os
import pickle
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = "risk_model.pkl"
SCALER_PATH = "feature_scaler.pkl"


def train_dummy_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a dummy RandomForest classifier on synthetic HRV data.
    This is a temporary model for testing the end-to-end pipeline.
    Replace with a real model trained on actual clinical data.

    Returns:
        Tuple of (trained_model, fitted_scaler)
    """
    logger.info("Training dummy risk prediction model on synthetic data...")

    np.random.seed(42)
    n_samples = 1000

    low_risk_samples = n_samples // 3
    moderate_risk_samples = n_samples // 3
    high_risk_samples = n_samples - low_risk_samples - moderate_risk_samples

    low_risk_features = np.random.normal(
        loc=[70, 8, 45, 25],
        scale=[5, 2, 10, 8],
        size=(low_risk_samples, 4)
    )

    moderate_risk_features = np.random.normal(
        loc=[80, 12, 35, 15],
        scale=[8, 3, 12, 7],
        size=(moderate_risk_samples, 4)
    )

    high_risk_features = np.random.normal(
        loc=[95, 18, 25, 8],
        scale=[10, 4, 15, 5],
        size=(high_risk_samples, 4)
    )

    X = np.vstack([low_risk_features, moderate_risk_features, high_risk_features])
    y = np.array(
        [0] * low_risk_samples +
        [1] * moderate_risk_samples +
        [2] * high_risk_samples
    )

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)

    logger.info(f"Dummy model trained. Training accuracy: {model.score(X_scaled, y):.3f}")

    return model, scaler


def load_or_create_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Load existing ML model or create and train a new dummy model.

    Returns:
        Tuple of (model, scaler)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            logger.info("Loading existing risk prediction model...")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Model and scaler loaded successfully")
            return model, scaler
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Creating new model instead...")

    model, scaler = train_dummy_model()

    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Failed to save model: {e}")

    return model, scaler


def extract_features_from_hrv(hrv_metrics: dict) -> np.ndarray:
    """
    Extract feature vector from HRV metrics for ML prediction.

    Args:
        hrv_metrics: Dictionary containing mean_hr, std_hr, rmssd, pnn50

    Returns:
        Feature array of shape (1, 4)
    """
    features = np.array([[
        hrv_metrics.get('mean_hr', 70.0),
        hrv_metrics.get('std_hr', 5.0),
        hrv_metrics.get('rmssd', 30.0),
        hrv_metrics.get('pnn50', 15.0)
    ]])
    return features


def predict_risk(
    hrv_metrics: dict,
    model: RandomForestClassifier,
    scaler: StandardScaler
) -> Tuple[str, float]:
    """
    Predict cardiovascular risk level from HRV metrics.

    Args:
        hrv_metrics: Dictionary containing HRV statistics
        model: Trained ML model
        scaler: Fitted feature scaler

    Returns:
        Tuple of (risk_level, confidence_score)
        - risk_level: One of "Low", "Moderate", "High"
        - confidence_score: Confidence percentage (0-100)
    """
    features = extract_features_from_hrv(hrv_metrics)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    risk_levels = ["Low", "Moderate", "High"]
    risk_level = risk_levels[prediction]
    confidence = float(probabilities[prediction] * 100)

    logger.info(f"Risk prediction: {risk_level} (confidence: {confidence:.1f}%)")

    return risk_level, confidence


model_instance = None
scaler_instance = None


def get_model_and_scaler() -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Get singleton instances of model and scaler.

    Returns:
        Tuple of (model, scaler)
    """
    global model_instance, scaler_instance

    if model_instance is None or scaler_instance is None:
        model_instance, scaler_instance = load_or_create_model()

    return model_instance, scaler_instance
