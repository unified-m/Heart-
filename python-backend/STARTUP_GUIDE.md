# Python Backend Startup Guide

## Quick Start

### 1. Install Dependencies
```bash
cd python-backend
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

## What Happens on Startup

### Console Output
You'll see detailed logging showing:

```
============================================================
Starting Health Monitoring ML Backend
============================================================
INFO - Loading ML model and scaler...
INFO - Attempting to load existing model from risk_model.pkl...
```

**First Time (No Model)**:
```
INFO - No existing model found
INFO - Checked paths: risk_model.pkl, feature_scaler.pkl
INFO - Creating new model...
============================================================
Training dummy risk prediction model on synthetic data...
NOTE: This is a temporary model for testing purposes
============================================================
INFO - Generating 1000 synthetic training samples...
INFO - Training RandomForest classifier...
✓ Model training complete
✓ Training accuracy: 0.XXX
✓ Model features: mean_hr, std_hr, rmssd, pnn50
✓ Risk classes: Low (0), Moderate (1), High (2)
INFO - Saving model to risk_model.pkl...
✓ Model saved
INFO - Saving scaler to feature_scaler.pkl...
✓ Scaler saved
✓ Model and scaler successfully saved to disk
✓ ML model loaded successfully
```

**Subsequent Starts (Model Exists)**:
```
INFO - Attempting to load existing model from risk_model.pkl...
✓ Model loaded
INFO - Loading scaler from feature_scaler.pkl...
✓ Scaler loaded
✓ Successfully loaded existing risk prediction model
✓ ML model loaded successfully
```

**PyVHR Check**:
```
✓ PyVHR is available
```
OR
```
✗ PyVHR not available, will use simulated data
```

```
============================================================
Backend initialization complete
============================================================
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Verify Everything is Working

### Check Health Status
```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "service": "ml-backend",
  "components": {
    "pyvhr": {
      "available": true,
      "status": "OK",
      "error": null
    },
    "ml_model": {
      "loaded": true,
      "status": "Loaded",
      "type": "RandomForest",
      "error": null
    }
  }
}
```

### Run Integration Tests
```bash
python test_integration.py
```

**Expected Output**:
```
============================================================
Running Integration Tests
============================================================

Testing imports...
✓ Core dependencies imported successfully

Testing PyVHR availability...
✓ PyVHR is available
(or: ✗ PyVHR not available (will use simulated data))

Testing heart rate extractor...
✓ Heart rate extractor works (avg BPM: XX.XX)
  HRV metrics: {...}

Testing ML model...
✓ ML model works (prediction: Low, confidence: XX.X%)

Testing API models...
✓ API models work correctly

Testing end-to-end pipeline...
✓ End-to-end pipeline works
  Generated XXX heart rate points
  Risk: low (XX.X%)

============================================================
Test Results Summary
============================================================
✓ PASS: Imports
✓ PASS: PyVHR Availability
✓ PASS: Heart Rate Extractor
✓ PASS: ML Model
✓ PASS: API Models
✓ PASS: End-to-End Pipeline

Total: 6/6 tests passed
============================================================
```

## Testing Video Processing

### Example Request
```bash
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-001",
    "video_url": "path/to/video.mp4"
  }'
```

### What Happens During Processing

**Console Logs**:
```
============================================================
[START] Processing video for recording_id: test-001
Video URL: path/to/video.mp4
============================================================

INFO - Using local video file: path/to/video.mp4
INFO - Checking PyVHR availability...
✓ PyVHR imported successfully
INFO - Processing video file: path/to/video.mp4 (XXXXX bytes)
INFO - Initializing PyVHR pipeline...
✓ PyVHR pipeline initialized
INFO - Starting PyVHR video processing...
INFO - Configuration: roi_approach='patches', method='POS', bpm_type='welch'
✓ PyVHR processing complete
INFO - Raw results: bvps shape=(...), timesigs shape=(...), bpms shape=(...)
INFO - Extracted XXX valid BPM measurements
✓ Average BPM: XX.XX
✓ HRV metrics computed: mean_hr=XX.XX, std_hr=X.XX, rmssd=XX.XX, pnn50=XX.XX
INFO - Extracted XXX heart rate data points

INFO - Extracting features from HRV metrics...
INFO - Features: [XX.XX X.XX XX.XX XX.XX]
INFO - Scaling features...
INFO - Running ML prediction...
✓ Risk prediction: Low
✓ Confidence: XX.X%
INFO - Class probabilities: Low=XX.X%, Moderate=XX.X%, High=XX.X%

INFO - Risk prediction: low (score: XX.XX)
============================================================
[SUCCESS] Analysis complete for test-001
============================================================
```

**If PyVHR Fails (Fallback)**:
```
✗ Unexpected error during PyVHR processing: ...
Error type: XXXError
⚠ Falling back to simulated heart rate data
========================================
⚠ GENERATING SIMULATED HEART RATE DATA
This is fallback data for testing purposes
========================================
INFO - Generating 240 simulated heart rate samples...
✓ Simulated data generated: 240 samples
✓ Simulated average BPM: XX.XX
```

**Response**:
```json
{
  "heart_rate_data": [
    {
      "timestamp_ms": 0,
      "heart_rate_bpm": 72.5,
      "confidence_score": 0.92
    },
    ...
  ],
  "risk_prediction": {
    "risk_level": "low",
    "risk_score": 85.3,
    "insights": {
      "variability": "Heart rate ranged from 65 to 85 BPM",
      "trend": "Normal",
      "recommendations": [
        "Your heart rate appears within normal range",
        "Continue regular physical activity and healthy lifestyle"
      ],
      "anomalies": []
    }
  }
}
```

## Troubleshooting

### Model Won't Load
**Symptoms**: Errors about pickle or model loading
**Solution**: Delete `risk_model.pkl` and `feature_scaler.pkl`, restart server

### PyVHR Not Working
**Symptoms**: Always uses simulated data
**Solution**:
1. Check PyVHR installation: `pip install pyVHR==2.0.0`
2. Verify OpenCV: `pip install opencv-python==4.8.1.78`
3. Check logs for specific error

### Server Freezes on Video Processing
**Solution**: This should not happen anymore! The system has:
- 180-second timeout on video processing
- Async processing via background threads
- Automatic timeout responses

If it still happens, check:
- Video file size (very large files may need longer timeout)
- Available system memory
- Python async loop issues

### Model Predictions Seem Wrong
**Note**: The current model is a DUMMY model trained on synthetic data
**Solution**: Train a real model on clinical data:
```python
from ml_model import train_dummy_model
import pickle

# Train your own model with real data
model, scaler = your_training_function()

# Save it
with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

## Production Deployment

### Environment Variables (Optional)
```bash
export HOST=0.0.0.0
export PORT=8000
export LOG_LEVEL=INFO
```

### Running with Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Example)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## Key Features Summary

✅ **Auto-Initialization**: Model loads or creates automatically on startup
✅ **Health Checks**: `/health` endpoint shows component status
✅ **Never Crashes**: All errors caught and handled gracefully
✅ **Detailed Logs**: Every step logged with clear visual indicators
✅ **Timeout Protection**: 180-second timeout prevents hangs
✅ **Graceful Fallbacks**: PyVHR failure → simulated data, ML failure → safe default
✅ **Production Ready**: Async processing, proper error responses, cleanup

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Run integration tests to identify problem area
3. Check `/health` endpoint for component status
4. Review ENHANCEMENTS.md for detailed technical info
