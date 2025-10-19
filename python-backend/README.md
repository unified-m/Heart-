# Python ML Backend - Health Monitoring System

This is the Python backend for the AI-powered health monitoring system. It processes facial videos using PyVHR to extract heart rate data and performs ML-based cardiovascular risk prediction.

## Architecture

The backend receives video analysis requests from the Supabase Edge Function and returns:
1. **Heart Rate Time-Series Data**: Extracted from facial videos using photoplethysmography (PPG)
2. **Risk Prediction**: ML-based cardiovascular risk assessment with insights and recommendations

## Setup

### 1. Install Dependencies

```bash
cd python-backend
pip install -r requirements.txt
```

### 2. Install PyVHR

PyVHR (Python Video-based Heart Rate) is the core library for extracting heart rate from videos.

```bash
pip install pyVHR
```

**PyVHR Documentation**: https://github.com/phuselab/pyVHR

### 3. Run the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

## Integration with PyVHR

### Current State
The backend now includes **full PyVHR integration** in the `heart_rate_extractor.py` module. If PyVHR is not available or fails, it automatically falls back to simulated data for testing.

### PyVHR Integration Steps

1. **Install PyVHR and dependencies**:
```bash
pip install pyVHR opencv-python scipy scikit-learn
```

2. **Replace the placeholder function** in `main.py`:

```python
from pyVHR.analysis.pipeline import Pipeline
import cv2

def extract_heart_rate_from_video(video_url: str) -> List[HeartRateDataPoint]:
    # Download video if it's a URL
    video_path = download_video(video_url)

    # Initialize PyVHR pipeline
    pipe = Pipeline()

    # Configure parameters
    params = {
        'videoFileName': video_path,
        'cuda': True,  # Use GPU if available
        'roi_approach': 'patches',  # or 'holistic'
        'method': 'POS',  # or 'GREEN', 'ICA', 'CHROM', 'LGI'
        'winsize': 6,  # Window size in seconds
        'bpm_type': 'median'
    }

    # Run analysis
    bvps, timesigs, bpm = pipe.run_on_video(**params)

    # Convert to heart rate data points
    heart_rate_data = []
    for i, (time, hr) in enumerate(zip(timesigs, bpm)):
        heart_rate_data.append(HeartRateDataPoint(
            timestamp_ms=int(time * 1000),
            heart_rate_bpm=float(hr),
            confidence_score=0.9  # Calculate based on signal quality
        ))

    return heart_rate_data
```

3. **PyVHR Methods Available**:
   - **POS** (Plane Orthogonal to Skin): Robust to motion
   - **GREEN**: Simple green channel method
   - **CHROM**: Chrominance-based method
   - **ICA**: Independent Component Analysis
   - **LGI**: Local Group Invariance

4. **Signal Quality Assessment**:
```python
def calculate_confidence_score(bvp_signal):
    # Implement signal quality metrics
    # - SNR (Signal-to-Noise Ratio)
    # - Signal strength
    # - Stability
    return confidence_score
```

## ML Model Integration

### Current State
The backend now includes **ML-based risk prediction** using a RandomForest classifier in the `ml_model.py` module. On first run, it automatically trains a dummy model on synthetic data and saves it as `risk_model.pkl`. This temporary model enables end-to-end testing and can be replaced with a real trained model later.

### Steps to Add ML Model

#### 1. Feature Engineering

Extract features from heart rate time-series:

```python
def extract_features(heart_rate_data: List[HeartRateDataPoint]) -> np.ndarray:
    heart_rates = np.array([d.heart_rate_bpm for d in heart_rate_data])

    features = {
        # Time domain features
        'mean_hr': np.mean(heart_rates),
        'std_hr': np.std(heart_rates),
        'min_hr': np.min(heart_rates),
        'max_hr': np.max(heart_rates),
        'range_hr': np.max(heart_rates) - np.min(heart_rates),

        # Heart Rate Variability (HRV) metrics
        'rmssd': calculate_rmssd(heart_rates),
        'sdnn': np.std(heart_rates),
        'pnn50': calculate_pnn50(heart_rates),

        # Frequency domain features (requires FFT)
        'lf_power': calculate_lf_power(heart_rates),
        'hf_power': calculate_hf_power(heart_rates),
        'lf_hf_ratio': calculate_lf_hf_ratio(heart_rates),

        # Statistical features
        'skewness': scipy.stats.skew(heart_rates),
        'kurtosis': scipy.stats.kurtosis(heart_rates),
    }

    return np.array(list(features.values()))
```

#### 2. Train Your Model

Example using scikit-learn:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Prepare training data
X_train = [extract_features(hr_data) for hr_data in training_data]
y_train = [label for label in training_labels]  # 0=low, 1=medium, 2=high

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/risk_prediction_model.pkl')
```

#### 3. Use Trained Model

```python
import joblib

# Load model once at startup
risk_model = joblib.load('models/risk_prediction_model.pkl')

def predict_cardiovascular_risk(heart_rate_data: List[HeartRateDataPoint]) -> RiskPrediction:
    # Extract features
    features = extract_features(heart_rate_data).reshape(1, -1)

    # Predict risk
    risk_probabilities = risk_model.predict_proba(features)[0]
    risk_class = risk_model.predict(features)[0]

    risk_levels = ['low', 'medium', 'high']
    risk_level = risk_levels[risk_class]
    risk_score = risk_probabilities[risk_class] * 100

    # Generate insights
    insights = generate_insights(heart_rate_data, risk_level, risk_score)

    return RiskPrediction(
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        insights=insights
    )
```

### Alternative: Deep Learning Models

For time-series analysis, consider:

**LSTM Model (PyTorch)**:
```python
import torch
import torch.nn as nn

class HeartRateLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_classes=3):
        super(HeartRateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load model
model = HeartRateLSTM()
model.load_state_dict(torch.load('models/lstm_model.pth'))
model.eval()
```

**1D CNN Model (TensorFlow)**:
```python
import tensorflow as tf

model = tf.keras.models.load_model('models/cnn_model.h5')

def predict_with_cnn(heart_rate_data):
    # Prepare sequence
    sequence = np.array([d.heart_rate_bpm for d in heart_rate_data])
    sequence = sequence.reshape(1, -1, 1)

    # Predict
    predictions = model.predict(sequence)
    return predictions
```

## Deployment

### Local Development
```bash
python main.py
```

### Production Deployment Options

#### Option 1: Docker Container
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Option 2: Cloud Services
- **Google Cloud Run**
- **AWS Lambda** (with API Gateway)
- **Azure Functions**
- **Heroku**

### Environment Variables

Set the `PYTHON_BACKEND_URL` environment variable in your Supabase Edge Function:

```bash
# In Supabase Dashboard > Edge Functions > Secrets
PYTHON_BACKEND_URL=https://your-backend-url.com
```

## API Endpoints

### POST /analyze-video
Analyzes a video and returns heart rate data and risk prediction.

**Request**:
```json
{
  "recording_id": "uuid",
  "video_url": "https://storage.url/video.webm"
}
```

**Response**:
```json
{
  "heart_rate_data": [
    {
      "timestamp_ms": 0,
      "heart_rate_bpm": 72.5,
      "confidence_score": 0.92
    }
  ],
  "risk_prediction": {
    "risk_level": "low",
    "risk_score": 25.5,
    "insights": {
      "variability": "Heart rate ranged from 68 to 78 BPM",
      "trend": "Normal",
      "recommendations": ["Continue healthy lifestyle"],
      "anomalies": []
    }
  }
}
```

## Testing

```bash
# Test the API
curl -X POST http://localhost:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test-123",
    "video_url": "test-video.webm"
  }'
```

## Project Structure

```
python-backend/
├── main.py                    # FastAPI application and endpoints
├── heart_rate_extractor.py    # PyVHR integration for HR extraction
├── ml_model.py                # ML model training and risk prediction
├── requirements.txt           # Python dependencies
├── risk_model.pkl            # Trained ML model (auto-generated)
└── feature_scaler.pkl        # Feature scaler (auto-generated)
```

## Implementation Details

### Heart Rate Extraction (`heart_rate_extractor.py`)
- Downloads videos from URLs or uses local files
- Uses PyVHR POS method for robust heart rate extraction
- Computes HRV metrics: mean_hr, std_hr, rmssd, pnn50
- Gracefully falls back to simulated data if PyVHR fails
- Cleans up temporary files automatically

### ML Risk Prediction (`ml_model.py`)
- Auto-trains dummy RandomForest model on synthetic HRV data
- Three risk classes: Low, Moderate, High
- Features: mean_hr, std_hr, rmssd, pnn50
- Saves model and scaler for reuse
- Can be replaced with real model by overwriting `risk_model.pkl`

## Next Steps

1. **Test with real videos**: Upload facial videos and verify PyVHR extraction
2. **Replace dummy model**: Train on real clinical data with actual risk labels
3. **Add video storage**: Implement video download/access from Supabase Storage
4. **Optimize performance**: Add caching, batch processing, GPU acceleration
5. **Add monitoring**: Implement logging, metrics, and error tracking
6. **Deploy to production**: Choose deployment platform and configure CI/CD

## Resources

- **PyVHR**: https://github.com/phuselab/pyVHR
- **Heart Rate Variability Analysis**: https://www.frontiersin.org/articles/10.3389/fpubh.2017.00258/full
- **Remote Photoplethysmography**: https://ieeexplore.ieee.org/document/7565547
