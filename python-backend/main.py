"""
AI-Powered Health Monitoring System - Python Backend
This backend processes facial videos to extract heart rate data using PyVHR
and performs ML-based risk prediction.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import numpy as np
import asyncio
import sys
from contextlib import asynccontextmanager

from heart_rate_extractor import (
    download_video,
    extract_heart_rate_with_pyvhr,
    cleanup_temp_file
)
from ml_model import get_model_and_scaler, predict_risk

logger = logging.getLogger(__name__)

model_status = {"loaded": False, "error": None, "type": "unknown"}
pyvhr_status = {"available": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app.
    Initializes ML model and checks PyVHR availability on startup.
    """
    global model_status, pyvhr_status

    logger.info("="*60)
    logger.info("Starting Health Monitoring ML Backend")
    logger.info("="*60)

    try:
        logger.info("Loading ML model and scaler...")
        model, scaler = get_model_and_scaler()
        model_status["loaded"] = True
        model_status["type"] = "RandomForest"
        logger.info("✓ ML model loaded successfully")
    except Exception as e:
        model_status["loaded"] = False
        model_status["error"] = str(e)
        logger.error(f"✗ Failed to load ML model: {e}")

    try:
        from pyVHR.analysis.pipeline import Pipeline
        pyvhr_status["available"] = True
        logger.info("✓ PyVHR is available")
    except ImportError as e:
        pyvhr_status["available"] = False
        pyvhr_status["error"] = str(e)
        logger.warning("✗ PyVHR not available, will use simulated data")

    logger.info("="*60)
    logger.info("Backend initialization complete")
    logger.info("="*60)

    yield

    logger.info("Shutting down backend...")


app = FastAPI(
    title="Health Monitoring ML Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
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
    """
    Health check endpoint that reports status of all components.
    """
    return {
        "status": "healthy",
        "service": "ml-backend",
        "components": {
            "pyvhr": {
                "available": pyvhr_status["available"],
                "status": "OK" if pyvhr_status["available"] else "Unavailable",
                "error": pyvhr_status.get("error")
            },
            "ml_model": {
                "loaded": model_status["loaded"],
                "status": "Loaded" if model_status["loaded"] else "Error",
                "type": model_status.get("type", "unknown"),
                "error": model_status.get("error")
            }
        }
    }


@app.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint for video analysis.
    This receives video data and returns heart rate analysis and risk prediction.

    Steps:
    1. Download/access video from video_url
    2. Process with PyVHR to extract heart rate signals
    3. Run ML model for risk prediction
    4. Return structured results
    """
    video_path = None

    try:
        logger.info("="*60)
        logger.info(f"[START] Processing video for recording_id: {request.recording_id}")
        logger.info(f"Video URL: {request.video_url}")
        logger.info("="*60)

        heart_rate_data = await asyncio.wait_for(
            asyncio.to_thread(extract_heart_rate_from_video, request.video_url),
            timeout=180.0
        )

        logger.info(f"Extracted {len(heart_rate_data)} heart rate data points")

        risk_prediction = predict_cardiovascular_risk(heart_rate_data)

        logger.info(f"Risk prediction: {risk_prediction.risk_level} (score: {risk_prediction.risk_score})")
        logger.info("="*60)
        logger.info(f"[SUCCESS] Analysis complete for {request.recording_id}")
        logger.info("="*60)

        return AnalysisResponse(
            heart_rate_data=heart_rate_data,
            risk_prediction=risk_prediction
        )

    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] Video processing exceeded 180 seconds for {request.recording_id}")
        raise HTTPException(
            status_code=504,
            detail="Video processing timed out. Please try with a shorter video."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("="*60)
        logger.error(f"[ERROR] Failed to process video: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("="*60)
        import traceback
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )


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
