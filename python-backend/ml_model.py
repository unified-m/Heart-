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
    logger.info("="*60)
    logger.info("Training dummy risk prediction model on synthetic data...")
    logger.info("NOTE: This is a temporary model for testing purposes")
    logger.info("="*60)

    np.random.seed(42)
    n_samples = 1000
    logger.info(f"Generating {n_samples} synthetic training samples...")

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
    logger.info("Training RandomForest classifier...")
    model.fit(X_scaled, y)

    training_accuracy = model.score(X_scaled, y)
    logger.info(f"✓ Model training complete")
    logger.info(f"✓ Training accuracy: {training_accuracy:.3f}")
    logger.info(f"✓ Model features: mean_hr, std_hr, rmssd, pnn50")
    logger.info(f"✓ Risk classes: Low (0), Moderate (1), High (2)")

    return model, scaler


def load_or_create_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Load existing ML model or create and train a new dummy model.
    This function ensures a model is always available.

    Returns:
        Tuple of (model, scaler)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            logger.info(f"Attempting to load existing model from {MODEL_PATH}...")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("✓ Model loaded")

            logger.info(f"Loading scaler from {SCALER_PATH}...")
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✓ Scaler loaded")

            logger.info("✓ Successfully loaded existing risk prediction model")
            return model, scaler
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.warning("⚠ Will create a new model instead...")

    else:
        logger.info("No existing model found")
        logger.info(f"Checked paths: {MODEL_PATH}, {SCALER_PATH}")

    logger.info("Creating new model...")
    model, scaler = train_dummy_model()

    try:
        logger.info(f"Saving model to {MODEL_PATH}...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info("✓ Model saved")

        logger.info(f"Saving scaler to {SCALER_PATH}...")
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("✓ Scaler saved")

        logger.info("✓ Model and scaler successfully saved to disk")
    except Exception as e:
        logger.warning(f"⚠ Failed to save model: {e}")
        logger.warning("Model will only be available for this session")

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
    try:
        logger.info("Extracting features from HRV metrics...")
        features = extract_features_from_hrv(hrv_metrics)
        logger.info(f"Features: {features[0]}")

        logger.info("Scaling features...")
        features_scaled = scaler.transform(features)

        logger.info("Running ML prediction...")
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        risk_levels = ["Low", "Moderate", "High"]
        risk_level = risk_levels[prediction]
        confidence = float(probabilities[prediction] * 100)

        logger.info(f"✓ Risk prediction: {risk_level}")
        logger.info(f"✓ Confidence: {confidence:.1f}%")
        logger.info(f"Class probabilities: Low={probabilities[0]*100:.1f}%, "
                   f"Moderate={probabilities[1]*100:.1f}%, "
                   f"High={probabilities[2]*100:.1f}%")

        return risk_level, confidence

    except Exception as e:
        logger.error(f"✗ Error during risk prediction: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.warning("⚠ Returning default low risk prediction")
        return "Low", 50.0


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
