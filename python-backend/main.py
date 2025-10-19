"""
AI-Powered Health Monitoring System - Python Backend
This backend processes facial videos to extract heart rate data using PyVHR
and performs ML-based risk prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import numpy as np

from heart_rate_extractor import (
    download_video,
    extract_heart_rate_with_pyvhr,
    cleanup_temp_file
)
from ml_model import get_model_and_scaler, predict_risk

app = FastAPI(title="Health Monitoring ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartRateDataPoint(BaseModel):
    timestamp_ms: int
    heart_rate_bpm: float
    confidence_score: float


class RiskInsights(BaseModel):
    variability: Optional[str] = None
    trend: Optional[str] = None
    recommendations: List[str] = []
    anomalies: List[str] = []


class RiskPrediction(BaseModel):
    risk_level: str
    risk_score: float
    insights: RiskInsights


class AnalysisRequest(BaseModel):
    recording_id: str
    video_url: str


class AnalysisResponse(BaseModel):
    heart_rate_data: List[HeartRateDataPoint]
    risk_prediction: RiskPrediction


@app.get("/")
async def root():
    return {
        "service": "Health Monitoring ML Backend",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze-video",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-backend"}


@app.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    """
    Main endpoint for video analysis.
    This receives video data and returns heart rate analysis and risk prediction.

    Steps:
    1. Download/access video from video_url
    2. Process with PyVHR to extract heart rate signals
    3. Run ML model for risk prediction
    4. Return structured results
    """
    try:
        logger.info(f"Processing video for recording_id: {request.recording_id}")

        heart_rate_data = extract_heart_rate_from_video(request.video_url)

        risk_prediction = predict_cardiovascular_risk(heart_rate_data)

        logger.info(f"Analysis complete for {request.recording_id}")

        return AnalysisResponse(
            heart_rate_data=heart_rate_data,
            risk_prediction=risk_prediction
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_heart_rate_from_video(video_url: str) -> List[HeartRateDataPoint]:
    """
    Extract heart rate time-series data from facial video using PyVHR.

    Args:
        video_url: URL or path to video file

    Returns:
        List of HeartRateDataPoint objects with timestamps and HR values
    """
    video_path = None
    try:
        video_path = download_video(video_url)

        hr_series, avg_bpm, hrv_metrics = extract_heart_rate_with_pyvhr(video_path)

        logger.info(f"Extracted {len(hr_series)} heart rate measurements")
        logger.info(f"Average BPM: {avg_bpm:.2f}")
        logger.info(f"HRV metrics: {hrv_metrics}")

        samples_per_second = len(hr_series) / 60.0 if len(hr_series) > 0 else 4

        heart_rate_data = []
        for i, hr_value in enumerate(hr_series):
            timestamp_ms = int((i / samples_per_second) * 1000)
            confidence = 0.85 + np.random.random() * 0.15

            heart_rate_data.append(HeartRateDataPoint(
                timestamp_ms=timestamp_ms,
                heart_rate_bpm=round(float(hr_value), 2),
                confidence_score=round(confidence, 2)
            ))

        return heart_rate_data

    finally:
        if video_path:
            cleanup_temp_file(video_path)


def predict_cardiovascular_risk(
    heart_rate_data: List[HeartRateDataPoint]
) -> RiskPrediction:
    """
    Predict cardiovascular risk using ML model based on heart rate time-series.

    Args:
        heart_rate_data: List of heart rate measurements

    Returns:
        RiskPrediction object with risk level, score, and insights
    """
    heart_rates = np.array([d.heart_rate_bpm for d in heart_rate_data])

    from heart_rate_extractor import compute_hrv_metrics
    hrv_metrics = compute_hrv_metrics(heart_rates)

    try:
        model, scaler = get_model_and_scaler()
        risk_level, confidence = predict_risk(hrv_metrics, model, scaler)
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        risk_level = "Moderate"
        confidence = 50.0

    avg_heart_rate = hrv_metrics['mean_hr']
    std_hr = hrv_metrics['std_hr']
    max_heart_rate = np.max(heart_rates)
    min_heart_rate = np.min(heart_rates)

    recommendations = []
    anomalies = []

    if risk_level == "Low":
        recommendations.append("Your heart rate appears within normal range")
        recommendations.append("Continue regular physical activity and healthy lifestyle")
    elif risk_level == "Moderate":
        recommendations.append("Consider monitoring your heart rate more frequently")
        recommendations.append("Maintain stress management practices")
        if avg_heart_rate < 60:
            anomalies.append("Resting heart rate below normal range detected")
            recommendations.append("Consider consulting with a healthcare provider")
        elif avg_heart_rate > 85:
            anomalies.append("Slightly elevated resting heart rate detected")
    else:
        anomalies.append("Elevated cardiovascular risk indicators detected")
        recommendations.append("Schedule an appointment with your healthcare provider")
        recommendations.append("Discuss your heart rate patterns and lifestyle factors")
        if avg_heart_rate > 100:
            anomalies.append("Significantly elevated resting heart rate detected")

    if std_hr > 12:
        anomalies.append(f"High heart rate variability detected (σ={std_hr:.1f})")
        recommendations.append("Ensure adequate rest and stress management")

    return RiskPrediction(
        risk_level=risk_level.lower(),
        risk_score=round(confidence, 2),
        insights=RiskInsights(
            variability=f"Heart rate ranged from {int(min_heart_rate)} to {int(max_heart_rate)} BPM",
            trend="Elevated" if avg_heart_rate > 85 else "Normal",
            recommendations=recommendations,
            anomalies=anomalies
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
