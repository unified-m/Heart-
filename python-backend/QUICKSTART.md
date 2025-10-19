# Quick Start Guide - Python Backend

## Installation

### 1. Navigate to backend directory
```bash
cd python-backend
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (web framework)
- PyVHR (heart rate extraction)
- OpenCV (video processing)
- Scikit-learn (ML models)
- NumPy, Pandas, SciPy (data processing)

### 3. Start the server
```bash
python main.py
```

The server will start on `http://localhost:8000`

## First Run Behavior

On first startup, the system will:
1. Initialize logging
2. Train a dummy ML model on synthetic HRV data (takes ~2 seconds)
3. Save `risk_model.pkl` and `feature_scaler.pkl`
4. Start FastAPI server

You'll see logs like:
```
INFO: Training dummy risk prediction model on synthetic data...
INFO: Dummy model trained. Training accuracy: 0.953
INFO: Model saved to risk_model.pkl
INFO: Uvicorn running on http://0.0.0.0:8000
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "ml-backend"
}
```

### Analyze Video
```bash
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-001",
    "video_url": "path/to/video.mp4"
  }'
```

**Note:** On first run without PyVHR properly configured, the system will use simulated heart rate data and display:
```
WARNING: Using simulated heart rate data. Replace with PyVHR implementation.
```

This is expected behavior for testing!

## Expected Response

```json
{
  "heart_rate_data": [
    {
      "timestamp_ms": 0,
      "heart_rate_bpm": 72.34,
      "confidence_score": 0.89
    },
    {
      "timestamp_ms": 250,
      "heart_rate_bpm": 73.21,
      "confidence_score": 0.91
    }
  ],
  "risk_prediction": {
    "risk_level": "low",
    "risk_score": 23.45,
    "insights": {
      "variability": "Heart rate ranged from 68 to 78 BPM",
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

## Testing with Real Videos

### Requirements for PyVHR to work:
1. Video must contain a visible face
2. Good lighting conditions
3. Minimal motion (person should be relatively still)
4. Duration: 30-60 seconds recommended
5. Format: MP4, WebM, AVI

### Example with local video:
```bash
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-real-001",
    "video_url": "/path/to/face-video.mp4"
  }'
```

### Example with URL:
```bash
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-url-001",
    "video_url": "https://example.com/video.mp4"
  }'
```

## Understanding the Logs

### Successful PyVHR Processing:
```
INFO: Downloading video from: https://example.com/video.mp4
INFO: Video downloaded to: /tmp/tmp123abc.mp4
INFO: Initializing PyVHR pipeline...
INFO: Processing video with PyVHR: /tmp/tmp123abc.mp4
INFO: Heart rate extraction successful. Average BPM: 72.45
INFO: Extracted 240 heart rate measurements
INFO: Average BPM: 72.45
INFO: HRV metrics: {'mean_hr': 72.45, 'std_hr': 5.23, 'rmssd': 32.1, 'pnn50': 18.5}
INFO: Risk prediction: Low (confidence: 78.9%)
```

### Fallback to Simulated Data:
```
WARNING: PyVHR not available: No module named 'pyVHR'
WARNING: Falling back to simulated heart rate data
WARNING: Generating simulated heart rate data
```

### ML Prediction:
```
INFO: Loading existing risk prediction model...
INFO: Model and scaler loaded successfully
INFO: Risk prediction: Low (confidence: 78.9%)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pyVHR'"
**Solution:**
```bash
pip install pyVHR opencv-python
```

### Issue: PyVHR fails to detect face
**Symptoms:**
```
ERROR: Error during PyVHR processing: PyVHR failed to extract heart rate from video
WARNING: Falling back to simulated heart rate data
```

**Solutions:**
- Ensure face is visible and well-lit
- Use higher resolution video
- Check video file is not corrupted
- Try different video with clearer face

### Issue: Model file not found
**Symptoms:**
```
ERROR: Failed to load model: [Errno 2] No such file or directory: 'risk_model.pkl'
INFO: Creating new model instead...
```

**Solution:** This is normal on first run. The model will be auto-created.

### Issue: Port 8000 already in use
**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Tips

### Enable Debug Logging
Edit `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Test with Python
```python
import requests

response = requests.post(
    'http://localhost:8000/analyze-video',
    json={
        'recording_id': 'test-123',
        'video_url': 'test.mp4'
    }
)

print(response.json())
```

### Performance Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test endpoint
ab -n 100 -c 10 http://localhost:8000/health
```

## Next Steps

1. **Test with real videos** - Try different face videos
2. **Monitor performance** - Check processing times
3. **Tune PyVHR parameters** - Adjust for your use case
4. **Train real model** - Collect clinical data
5. **Deploy to cloud** - Choose hosting platform

## Support

For issues:
1. Check logs in console
2. Review PyVHR documentation: https://github.com/phuselab/pyVHR
3. Verify video quality and format
4. Test with simulated data first

## Production Deployment

Before deploying to production:
- [ ] Replace dummy ML model with real trained model
- [ ] Add authentication/authorization
- [ ] Enable HTTPS
- [ ] Set up monitoring and alerting
- [ ] Implement rate limiting
- [ ] Add input validation and sanitization
- [ ] Configure proper CORS settings
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Add health checks and readiness probes
- [ ] Optimize for your expected load
