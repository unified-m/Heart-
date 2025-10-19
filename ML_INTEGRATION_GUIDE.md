# ML Integration Guide - AI Health Monitoring System

This guide explains how the real ML/AI components integrate with your health monitoring system.

## System Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Frontend  │─────▶│  Supabase Edge   │─────▶│  Python ML      │
│   (React)   │      │    Function      │      │   Backend       │
└─────────────┘      └──────────────────┘      └─────────────────┘
       │                      │                          │
       │                      ▼                          ▼
       │              ┌──────────────┐         ┌─────────────────┐
       └─────────────▶│   Supabase   │◀────────│  PyVHR + ML     │
                      │   Database   │         │    Models       │
                      └──────────────┘         └─────────────────┘
```

## Data Flow

### 1. Video Recording (Frontend)
```typescript
// User records 60-second facial video using WebRTC
const videoBlob = await recordFacialVideo();

// Create recording entry in database
const recording = await createRecording(userId, videoUrl, 60);

// Trigger ML processing via Edge Function
await processVideoWithML(recording.id, videoUrl);
```

### 2. Edge Function Processing
```typescript
// Edge Function receives request
{
  recording_id: "uuid",
  video_url: "storage-url"
}

// Forwards to Python backend
POST https://your-python-backend.com/analyze-video

// On success, saves results to database
- Insert heart_rate_data (240 data points for 60s @ 4 samples/sec)
- Insert risk_prediction
- Update video_recordings.processing_status = 'completed'
```

### 3. Python Backend Analysis
```python
# 1. Extract heart rate using PyVHR
heart_rate_data = extract_heart_rate_from_video(video_url)
# Returns: List of HeartRateDataPoint with timestamp, BPM, confidence

# 2. Predict risk using ML model
risk_prediction = predict_cardiovascular_risk(heart_rate_data)
# Returns: RiskPrediction with level, score, insights
```

### 4. Results Display (Frontend)
```typescript
// Frontend loads results from database
const heartRateData = await getHeartRateData(recordingId);
const riskPrediction = await getRiskPrediction(recordingId);

// Displays:
- Interactive heart rate chart (time-series visualization)
- Risk assessment card (low/medium/high with score)
- Health insights and recommendations
```

## Current Implementation Status

### ✅ Completed
- **Frontend**: Full video recording UI with WebRTC
- **Database**: Complete schema with RLS policies
- **Edge Function**: Deployed and ready to forward requests
- **API Layer**: All endpoints connected properly
- **Visualization**: Heart rate charts and risk displays
- **Python Backend Template**: FastAPI server with placeholder functions

### ⚠️ Requires Implementation
1. **PyVHR Integration** - Replace simulation with actual heart rate extraction
2. **ML Model Training** - Train and deploy risk prediction model
3. **Video Storage** - Connect to Supabase Storage for video access

---

## Implementation Steps

## Step 1: Set Up Python Backend

### Install Dependencies
```bash
cd python-backend
pip install -r requirements.txt

# Install PyVHR
pip install pyVHR opencv-python scipy scikit-learn
```

### Start the Server
```bash
python main.py
# Server runs on http://localhost:8000
```

### Test the Server
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "service": "ml-backend"}
```

---

## Step 2: Integrate PyVHR

### What is PyVHR?
PyVHR (Python Video-based Heart Rate) extracts heart rate from facial videos using:
- **Photoplethysmography (PPG)**: Detects blood volume changes in skin
- **Computer Vision**: Tracks facial regions and color changes
- **Signal Processing**: Converts color variations to heart rate

### Replace Simulation Code

In `python-backend/main.py`, replace the `extract_heart_rate_from_video()` function:

```python
from pyVHR.analysis.pipeline import Pipeline
import tempfile
import requests

def extract_heart_rate_from_video(video_url: str) -> List[HeartRateDataPoint]:
    """
    Extract heart rate from video using PyVHR
    """
    # Step 1: Download video
    response = requests.get(video_url)
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
        tmp_file.write(response.content)
        video_path = tmp_file.name

    # Step 2: Initialize PyVHR pipeline
    pipe = Pipeline()

    # Step 3: Configure parameters
    try:
        # Run PyVHR analysis
        bvps, timesigs, bpm = pipe.run_on_video(
            videoFileName=video_path,
            cuda=False,  # Set True if GPU available
            roi_approach='patches',  # or 'holistic'
            method='POS',  # Plane-Orthogonal-to-Skin (best for motion)
            winsize=6,  # 6-second windows
            bpm_type='median'
        )

        # Step 4: Convert to required format
        heart_rate_data = []
        for i in range(len(timesigs)):
            heart_rate_data.append(HeartRateDataPoint(
                timestamp_ms=int(timesigs[i] * 1000),
                heart_rate_bpm=float(bpm[i]),
                confidence_score=calculate_signal_quality(bvps[i])
            ))

        return heart_rate_data

    finally:
        # Cleanup
        os.unlink(video_path)


def calculate_signal_quality(bvp_signal) -> float:
    """
    Calculate confidence score based on signal quality
    """
    # Calculate SNR, power, stability
    snr = calculate_snr(bvp_signal)
    # Map SNR to 0-1 confidence score
    confidence = min(1.0, max(0.0, snr / 30.0))
    return round(confidence, 2)
```

### PyVHR Method Comparison

| Method | Description | Best For |
|--------|-------------|----------|
| **POS** | Plane-Orthogonal-to-Skin | Motion robustness |
| **CHROM** | Chrominance-based | Good lighting |
| **GREEN** | Green channel only | Simple, fast |
| **ICA** | Independent Component | Multiple signals |
| **LGI** | Local Group Invariance | Varying illumination |

**Recommended**: Start with **POS** method for best motion tolerance.

---

## Step 3: Train ML Risk Prediction Model

### Data Collection Requirements

To train a cardiovascular risk prediction model, you need:

1. **Heart Rate Time-Series Data**
   - 60-second recordings
   - 4+ samples per second
   - Timestamp, BPM, confidence

2. **Risk Labels**
   - Clinical diagnoses (if available)
   - Risk scores from medical assessments
   - Historical cardiovascular events

3. **Demographics** (optional)
   - Age, gender, weight, height
   - Medical history
   - Lifestyle factors

### Feature Engineering

Extract meaningful features from heart rate data:

```python
import numpy as np
from scipy import stats
from scipy.fft import fft

def extract_hrv_features(heart_rate_data: List[HeartRateDataPoint]) -> dict:
    """
    Extract Heart Rate Variability (HRV) features
    """
    hr_values = np.array([d.heart_rate_bpm for d in heart_rate_data])

    # Time-domain features
    features = {
        # Basic statistics
        'mean_hr': np.mean(hr_values),
        'std_hr': np.std(hr_values),
        'min_hr': np.min(hr_values),
        'max_hr': np.max(hr_values),
        'range_hr': np.ptp(hr_values),

        # HRV metrics (assuming RR intervals can be derived)
        'rmssd': calculate_rmssd(hr_values),  # Root Mean Square of Successive Differences
        'sdnn': np.std(hr_values),  # Standard Deviation of NN intervals
        'pnn50': calculate_pnn50(hr_values),  # % of consecutive HR diffs > 50 BPM

        # Statistical measures
        'skewness': stats.skew(hr_values),
        'kurtosis': stats.kurtosis(hr_values),
        'cv': np.std(hr_values) / np.mean(hr_values),  # Coefficient of variation
    }

    # Frequency-domain features (requires FFT)
    freq_features = extract_frequency_features(hr_values)
    features.update(freq_features)

    return features


def extract_frequency_features(hr_values: np.ndarray) -> dict:
    """
    Extract frequency-domain features using FFT
    """
    # Compute power spectral density
    fft_vals = np.abs(fft(hr_values))
    frequencies = np.fft.fftfreq(len(hr_values), d=0.25)  # 4 samples/sec

    # Define frequency bands
    vlf_band = (0.0033, 0.04)  # Very Low Frequency
    lf_band = (0.04, 0.15)     # Low Frequency
    hf_band = (0.15, 0.4)      # High Frequency

    # Calculate power in each band
    vlf_power = calculate_band_power(fft_vals, frequencies, vlf_band)
    lf_power = calculate_band_power(fft_vals, frequencies, lf_band)
    hf_power = calculate_band_power(fft_vals, frequencies, hf_band)

    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
        'total_power': vlf_power + lf_power + hf_power
    }
```

### Model Training Example

#### Option 1: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Prepare training data
X = []  # Features extracted from each recording
y = []  # Risk labels: 0=low, 1=medium, 2=high

for recording in training_recordings:
    features = extract_hrv_features(recording.heart_rate_data)
    X.append(list(features.values()))
    y.append(recording.risk_label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/risk_prediction_rf.pkl')
```

#### Option 2: LSTM for Time-Series

```python
import torch
import torch.nn as nn

class HeartRateLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(HeartRateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

# Training loop
model = HeartRateLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), 'models/lstm_model.pth')
```

### Use Trained Model in Production

```python
import joblib

# Load model at startup
risk_model = joblib.load('models/risk_prediction_rf.pkl')

def predict_cardiovascular_risk(
    heart_rate_data: List[HeartRateDataPoint]
) -> RiskPrediction:
    # Extract features
    features = extract_hrv_features(heart_rate_data)
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Predict
    risk_class = risk_model.predict(feature_vector)[0]
    risk_probabilities = risk_model.predict_proba(feature_vector)[0]

    risk_levels = ['low', 'medium', 'high']
    risk_level = risk_levels[risk_class]
    risk_score = risk_probabilities[risk_class] * 100

    # Generate insights
    insights = generate_insights_from_features(features, risk_level)

    return RiskPrediction(
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        insights=insights
    )
```

---

## Step 4: Deploy Python Backend

### Option 1: Docker

```dockerfile
# python-backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t health-ml-backend .
docker run -p 8000:8000 health-ml-backend
```

### Option 2: Cloud Services

#### Google Cloud Run
```bash
gcloud run deploy health-ml-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS Lambda + API Gateway
```bash
# Package your FastAPI app with Mangum adapter
pip install mangum
# Deploy using AWS SAM or Serverless Framework
```

### Configure Edge Function

Once deployed, the Edge Function will automatically use your Python backend URL through the `PYTHON_BACKEND_URL` environment variable. No manual configuration needed!

---

## Step 5: Testing the Complete Flow

### 1. Test Python Backend
```bash
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-123",
    "video_url": "https://example.com/test-video.webm"
  }'
```

### 2. Test Edge Function
```bash
curl -X POST https://your-project.supabase.co/functions/v1/process-heart-rate \
  -H "Authorization: Bearer YOUR_ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "uuid",
    "video_url": "storage-url"
  }'
```

### 3. Test Full Frontend Flow
1. Sign up / Sign in
2. Record 60-second facial video
3. Click "Analyze Video"
4. Wait for processing (check dashboard for status)
5. View results with heart rate chart and risk assessment

---

## Monitoring and Optimization

### Add Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/analyze-video")
async def analyze_video(request: AnalysisRequest):
    logger.info(f"Received request for recording_id: {request.recording_id}")
    # ... processing ...
    logger.info(f"Completed analysis in {duration}s")
```

### Performance Optimization
- **GPU Acceleration**: Enable CUDA for PyVHR (`cuda=True`)
- **Video Compression**: Reduce video size before processing
- **Caching**: Cache processed results
- **Batch Processing**: Process multiple videos in parallel
- **Async Processing**: Use background tasks for long operations

---

## Troubleshooting

### PyVHR Issues
- **"No face detected"**: Ensure good lighting and face visibility
- **Low confidence scores**: Check video quality and frame rate
- **Slow processing**: Enable GPU or reduce video resolution

### ML Model Issues
- **Poor predictions**: Need more training data
- **High error rate**: Check feature engineering
- **Model not loading**: Verify file paths and dependencies

### Integration Issues
- **Edge Function timeout**: Increase timeout or use async processing
- **Database constraints**: Ensure user profile exists before recording
- **CORS errors**: Check Edge Function headers

---

## Next Steps

1. ✅ **Test with PyVHR**: Replace simulation with real heart rate extraction
2. ✅ **Collect Training Data**: Gather labeled datasets for model training
3. ✅ **Train ML Model**: Build and validate risk prediction model
4. ✅ **Deploy Backend**: Choose cloud provider and deploy
5. ✅ **Monitor Performance**: Add logging, metrics, and alerts
6. ✅ **Iterate**: Improve model accuracy with more data

---

## Resources

- **PyVHR GitHub**: https://github.com/phuselab/pyVHR
- **Heart Rate Variability**: https://www.frontiersin.org/articles/10.3389/fpubh.2017.00258/full
- **Remote PPG**: https://ieeexplore.ieee.org/document/7565547
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Supabase Edge Functions**: https://supabase.com/docs/guides/functions
