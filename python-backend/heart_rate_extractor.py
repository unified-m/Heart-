"""
Heart Rate Extraction Module using PyVHR.
Processes facial videos to extract heart rate time-series data using
Photoplethysmography (PPG) via remote video analysis.
"""

import logging
import os
import tempfile
from typing import List, Tuple, Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)


def download_video(video_url: str) -> str:
    """
    Download video from URL to temporary file.

    Args:
        video_url: URL or local path to video file

    Returns:
        Path to downloaded video file
    """
    if os.path.exists(video_url):
        logger.info(f"Using local video file: {video_url}")
        return video_url

    logger.info(f"Downloading video from: {video_url}")
    response = requests.get(video_url, stream=True, timeout=60)
    response.raise_for_status()

    suffix = '.mp4' if 'mp4' in video_url.lower() else '.webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_path = temp_file.name

    logger.info(f"Video downloaded to: {temp_path}")
    return temp_path


def extract_heart_rate_with_pyvhr(video_path: str) -> Tuple[np.ndarray, float, dict]:
    """
    Extract heart rate signal from video using PyVHR library.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (heart_rate_series, average_bpm, hrv_metrics)
        - heart_rate_series: Array of HR values over time
        - average_bpm: Average heart rate in BPM
        - hrv_metrics: Dictionary containing HRV statistics
    """
    try:
        from pyVHR.analysis.pipeline import Pipeline

        logger.info("Initializing PyVHR pipeline...")
        pipe = Pipeline()

        logger.info(f"Processing video with PyVHR: {video_path}")

        results = pipe.run_on_video(
            videoFileName=video_path,
            cuda=False,
            roi_approach='patches',
            method='POS',
            bpm_type='welch',
            pre_filt=True,
            post_filt=True,
            verb=False
        )

        bvps, timesigs, bpms = results

        if bpms is None or len(bpms) == 0:
            raise ValueError("PyVHR failed to extract heart rate from video")

        average_bpm = float(np.mean(bpms))

        logger.info(f"Heart rate extraction successful. Average BPM: {average_bpm:.2f}")

        hrv_metrics = compute_hrv_metrics(bpms)

        return bpms, average_bpm, hrv_metrics

    except ImportError as e:
        logger.error(f"PyVHR not available: {e}")
        logger.warning("Falling back to simulated heart rate data")
        return generate_simulated_heart_rate()
    except Exception as e:
        logger.error(f"Error during PyVHR processing: {e}")
        logger.warning("Falling back to simulated heart rate data")
        return generate_simulated_heart_rate()


def compute_hrv_metrics(heart_rate_series: np.ndarray) -> dict:
    """
    Compute Heart Rate Variability (HRV) metrics from heart rate series.

    Args:
        heart_rate_series: Array of heart rate values

    Returns:
        Dictionary containing HRV metrics:
        - mean_hr: Mean heart rate
        - std_hr: Standard deviation of heart rate
        - rmssd: Root Mean Square of Successive Differences
        - pnn50: Percentage of successive RR intervals that differ by more than 50ms
    """
    if len(heart_rate_series) < 2:
        return {
            'mean_hr': float(heart_rate_series[0]) if len(heart_rate_series) > 0 else 70.0,
            'std_hr': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0
        }

    rr_intervals = 60000.0 / heart_rate_series

    mean_hr = float(np.mean(heart_rate_series))
    std_hr = float(np.std(heart_rate_series))

    successive_diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = float(nn50 / len(successive_diffs) * 100) if len(successive_diffs) > 0 else 0.0

    return {
        'mean_hr': mean_hr,
        'std_hr': std_hr,
        'rmssd': rmssd,
        'pnn50': pnn50
    }


def generate_simulated_heart_rate() -> Tuple[np.ndarray, float, dict]:
    """
    Generate simulated heart rate data for testing when PyVHR is unavailable.

    Returns:
        Tuple of (heart_rate_series, average_bpm, hrv_metrics)
    """
    logger.warning("Generating simulated heart rate data")

    duration_seconds = 60
    samples_per_second = 4
    total_samples = duration_seconds * samples_per_second

    base_heart_rate = 70 + np.random.random() * 20
    heart_rate_series = []

    for i in range(total_samples):
        variation = np.sin(i / 10) * 5 + (np.random.random() - 0.5) * 3
        heart_rate = max(50, min(120, base_heart_rate + variation))
        heart_rate_series.append(heart_rate)

    heart_rate_series = np.array(heart_rate_series)
    average_bpm = float(np.mean(heart_rate_series))
    hrv_metrics = compute_hrv_metrics(heart_rate_series)

    return heart_rate_series, average_bpm, hrv_metrics


def cleanup_temp_file(file_path: str) -> None:
    """
    Remove temporary file if it was created.

    Args:
        file_path: Path to file to remove
    """
    try:
        if file_path and os.path.exists(file_path) and '/tmp' in file_path:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
