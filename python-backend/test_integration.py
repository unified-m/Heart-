"""
Integration test script for the Python backend.
Tests PyVHR integration, ML model, and API endpoints.
"""

import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    try:
        import numpy
        import pandas
        import sklearn
        import requests
        import fastapi
        import pydantic
        logger.info("✓ Core dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_pyvhr_availability():
    """Test if PyVHR is available."""
    logger.info("Testing PyVHR availability...")
    try:
        from pyVHR.analysis.pipeline import Pipeline
        logger.info("✓ PyVHR is available")
        return True
    except ImportError:
        logger.warning("✗ PyVHR not available (will use simulated data)")
        return False


def test_heart_rate_extractor():
    """Test heart rate extraction module."""
    logger.info("Testing heart rate extractor...")
    try:
        from heart_rate_extractor import (
            compute_hrv_metrics,
            generate_simulated_heart_rate
        )
        import numpy as np

        hr_series, avg_bpm, hrv_metrics = generate_simulated_heart_rate()

        assert len(hr_series) > 0, "Heart rate series is empty"
        assert 50 <= avg_bpm <= 120, f"Average BPM out of range: {avg_bpm}"
        assert 'mean_hr' in hrv_metrics, "Missing mean_hr in HRV metrics"
        assert 'std_hr' in hrv_metrics, "Missing std_hr in HRV metrics"
        assert 'rmssd' in hrv_metrics, "Missing rmssd in HRV metrics"
        assert 'pnn50' in hrv_metrics, "Missing pnn50 in HRV metrics"

        logger.info(f"✓ Heart rate extractor works (avg BPM: {avg_bpm:.2f})")
        logger.info(f"  HRV metrics: {hrv_metrics}")
        return True
    except Exception as e:
        logger.error(f"✗ Heart rate extractor test failed: {e}")
        return False


def test_ml_model():
    """Test ML model training and prediction."""
    logger.info("Testing ML model...")
    try:
        from ml_model import (
            train_dummy_model,
            extract_features_from_hrv,
            predict_risk
        )

        model, scaler = train_dummy_model()

        test_hrv = {
            'mean_hr': 75.0,
            'std_hr': 8.0,
            'rmssd': 35.0,
            'pnn50': 20.0
        }

        features = extract_features_from_hrv(test_hrv)
        assert features.shape == (1, 4), f"Wrong feature shape: {features.shape}"

        risk_level, confidence = predict_risk(test_hrv, model, scaler)
        assert risk_level in ["Low", "Moderate", "High"], f"Invalid risk level: {risk_level}"
        assert 0 <= confidence <= 100, f"Confidence out of range: {confidence}"

        logger.info(f"✓ ML model works (prediction: {risk_level}, confidence: {confidence:.1f}%)")
        return True
    except Exception as e:
        logger.error(f"✗ ML model test failed: {e}")
        return False


def test_api_models():
    """Test Pydantic models."""
    logger.info("Testing API models...")
    try:
        from main import (
            HeartRateDataPoint,
            RiskInsights,
            RiskPrediction,
            AnalysisRequest,
            AnalysisResponse
        )

        hr_point = HeartRateDataPoint(
            timestamp_ms=1000,
            heart_rate_bpm=72.5,
            confidence_score=0.9
        )

        insights = RiskInsights(
            variability="Normal",
            trend="Stable",
            recommendations=["Test recommendation"],
            anomalies=[]
        )

        risk_pred = RiskPrediction(
            risk_level="low",
            risk_score=25.0,
            insights=insights
        )

        request = AnalysisRequest(
            recording_id="test-123",
            video_url="test.mp4"
        )

        response = AnalysisResponse(
            heart_rate_data=[hr_point],
            risk_prediction=risk_pred
        )

        json_data = response.model_dump_json()
        assert len(json_data) > 0, "Failed to serialize response"

        logger.info("✓ API models work correctly")
        return True
    except Exception as e:
        logger.error(f"✗ API models test failed: {e}")
        return False


def test_end_to_end():
    """Test end-to-end video processing pipeline."""
    logger.info("Testing end-to-end pipeline...")
    try:
        from main import extract_heart_rate_from_video, predict_cardiovascular_risk

        test_video_url = "simulated"

        heart_rate_data = extract_heart_rate_from_video(test_video_url)
        assert len(heart_rate_data) > 0, "No heart rate data generated"

        risk_prediction = predict_cardiovascular_risk(heart_rate_data)
        assert risk_prediction.risk_level in ["low", "medium", "high"], \
            f"Invalid risk level: {risk_prediction.risk_level}"

        logger.info(f"✓ End-to-end pipeline works")
        logger.info(f"  Generated {len(heart_rate_data)} heart rate points")
        logger.info(f"  Risk: {risk_prediction.risk_level} ({risk_prediction.risk_score:.1f}%)")
        return True
    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("Running Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("PyVHR Availability", test_pyvhr_availability),
        ("Heart Rate Extractor", test_heart_rate_extractor),
        ("ML Model", test_ml_model),
        ("API Models", test_api_models),
        ("End-to-End Pipeline", test_end_to_end),
    ]

    results = []
    for name, test_func in tests:
        logger.info("")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
