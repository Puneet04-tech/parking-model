# Parking Congestion Intelligence System

AI-driven parking violation detection and enforcement prioritization system for smart city traffic management.

**Team**: TrafficMind Solutions
**Author**: Puneet Chaturvedi

## 🚀 Project Overview

This system uses machine learning to:
1. **Automate Report Validation**: Detect and filter invalid parking violation reports using LightGBM binary classification
2. **Identify Spatial Hotspots**: Cluster violation coordinates using DBSCAN to detect chronic violation zones
3. **Prioritize Enforcement**: Predict congestion impact and dispatch priority using multi-class classification

## 📊 Dataset

- **Source**: Bangalore Police violation records (Jan-May 2024)
- **Size**: 298,450 parking violation reports
- **Features**: Location, vehicle type, violation types, timestamps, reporter metadata
- **Labeled Data**: 165,173 reports with validation status
- **Unlabeled Data**: 125,000+ reports for semi-supervised learning

## 🏗️ Architecture

### Model Pipeline

1. **Model 1 - Report Validator** (Binary Classification)
   - Input: 55 enhanced features (spatial, temporal, violation types, reporter reliability, user behavior, device patterns, interaction features)
   - Output: Approved/Rejected
   - **Final Accuracy**: 80.76%
   - **Final ROC-AUC**: 84.07%
   - **Training Data**: 165k+ samples (labeled + pseudo-labeled)
   - **Key Improvements**:
     - Enhanced feature engineering (13 new features beyond base 42)
     - Class imbalance handling with scale_pos_weight
     - Semi-supervised learning using 125K unlabeled samples
     - Hyperparameter optimization with Optuna (40 trials)
     - Out-of-Fold target encoding for user/device reliability

2. **Model 2 - Hotspot Detector** (Unsupervised Clustering)
   - Algorithm: DBSCAN (Density-Based Spatial Clustering)
   - Parameters: eps=0.005, min_samples=10
   - Output: Distinct parking hotspots with cluster labels
   - Coverage: Identifies chronic violation zones for patrol planning
   - Use Cases: Patrol route optimization, resource allocation, infrastructure planning

3. **Model 3 - Priority Predictor** (Multi-class Classification)
   - Input: 11 ambient features (location, time, area density, reporter reliability)
   - Output: Low/Medium/High priority
   - Accuracy: 85.45% (10-fold CV: 85.42% ± 0.01)
   - Use Case: Quick triage when full violation details aren't available

## 🔧 Technical Highlights

- **Anti-leakage design**: Model 3 uses only ambient features to prevent mathematical proxy leakage
- **Advanced feature engineering**: Cyclic time encoding, OOF target encoding, frequency encoding, interaction features
- **Semi-supervised learning**: Pseudo-labeling on 125K unlabeled samples to improve Model 1
- **Class imbalance handling**: Weighted training with computed class weights
- **Regularization**: L1/L2 penalties, early stopping, depth limits, dropout (neural network)
- **Cross-validation**: 10-fold stratified CV with statistical significance testing
- **Model comparison**: LightGBM vs XGBoost vs Random Forest vs Neural Network
- **Hyperparameter tuning**: Optuna-based optimization with TPE sampler
- **Ensemble methods**: Combining gradient boosting and neural networks

## 📦 Deployment

### Live Deployments

- **API**: https://parking-model.onrender.com
- **Frontend**: https://parking-intelligence-api.netlify.app
- **API Documentation**: https://parking-model.onrender.com/docs

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook parking_congestion_intelligence.ipynb

# Run the API locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run the frontend locally
cd frontend
python -m http.server 8000
```

### API Deployment (Render)

The FastAPI backend is deployed on Render with automatic scaling.

**Configuration**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Python Version**: 3.11

### Frontend Deployment (Netlify)

The HTML/CSS/JavaScript frontend is deployed on Netlify as a static site.

**Features**:
- Modern gradient-styled UI
- Two tabs: Priority Prediction and Report Validation
- Real-time API status monitoring
- Responsive design for mobile/desktop
- Color-coded results with confidence scores

### API Endpoints

```bash
# Health check
GET /
GET /health

# Predict priority (11 features)
POST /predict_priority
Content-Type: application/json

{
  "latitude": 12.9782,
  "longitude": 77.6011,
  "hour": 17,
  "dayofweek": 0,
  "is_weekend": 0,
  "month": 6,
  "police_station_freq": 1500.0,
  "pincode_freq": 800.0,
  "junction_name_freq": 500.0,
  "user_target_enc": 0.85,
  "device_target_enc": 0.90
}

# Validate report (55 features)
POST /validate_report
Content-Type: application/json

{
  "latitude": 12.9782,
  "longitude": 77.6011,
  "vehicle_type": "Car",
  "center_code": 100.0,
  "police_station": "Station1",
  "junction_name": "MG Road",
  "pincode": "560001",
  "hour": 17,
  "dayofweek": 0,
  "month": 6,
  "is_weekend": 0,
  "vehicle_type_freq": 5000.0,
  "police_station_freq": 1500.0,
  "junction_name_freq": 800.0,
  "pincode_freq": 3000.0,
  "user_target_enc": 0.85,
  "device_target_enc": 0.90,
  "dist_from_center": 0.05,
  "user_report_count": 0,
  "last_report_gap_hours": 0.0,
  "device_report_count": 0,
  "pincode_report_density": 0,
  "has_multiple_violations": 0,
  "hour_x_weekend": 0,
  "lat_x_lon": 1005.0,
  "hour_sin_x_cos": 0.433,
  "police_station_activity": 0,
  "junction_complexity": 2,
  "is_morning_peak": 0,
  "is_evening_peak": 0
}
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Metrics

### Model 1 (Report Validator)
- **Final Accuracy**: 80.76%
- **Final ROC-AUC**: 84.07%
- **Training Data**: 165k+ samples (labeled + pseudo-labeled)
- **Features**: 55 enhanced features
- **Improvement**: +5% ROC-AUC from semi-supervised learning

### Model 3 (Priority Predictor)
- **Accuracy**: 85.45%
- **10-fold CV**: 85.42% ± 0.01
- **95% CI**: [85.40, 85.44]
- **Statistical Significance**: p < 0.001 vs random guessing

### Class-wise Performance
- **Low Priority**: 92% F1-score (97% recall)
- **Medium Priority**: 54% F1-score (43% recall)
- **High Priority**: 71% F1-score (69% recall)

## 🔬 Model Comparison Results

| Model | Mean Accuracy | Std Dev | Training Time |
|-------|--------------|---------|---------------|
| LightGBM | 0.8542 | 0.0012 | 45.2s |
| XGBoost | 0.8498 | 0.0015 | 52.1s |
| Random Forest | 0.8234 | 0.0021 | 38.7s |

**Winner**: LightGBM (best accuracy with reasonable training time)

## 🎯 Ablation Study Results

| Feature Group | Accuracy | Drop from Full |
|--------------|----------|----------------|
| All Features | 0.8545 | 0.0000 |
| Spatial + Temporal | 0.8421 | 0.0124 |
| Without Spatial | 0.7892 | 0.0653 |
| Without Temporal | 0.8123 | 0.0422 |
| Without Reporter Reliability | 0.8489 | 0.0056 |

**Key Insight**: Spatial features contribute most to performance (6.53% drop when removed)

## 🚨 Error Analysis

- **Overall Error Rate**: 14.55%
- **Confidence Gap**: Correct predictions have 0.15 higher mean confidence
- **Spatial Error Distribution**: Errors are geographically clustered in transition zones

## 📝 Files Structure

```
parking-model/
├── parking_congestion_intelligence.ipynb  # Main notebook with ML pipeline
├── app.py                                  # FastAPI application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── render.yaml                             # Render deployment configuration
├── ambient_features.json                   # Feature list for Model 3
├── model_report_validator.joblib           # Trained Model 1 (55 features)
├── model_congestion_prioritizer.joblib     # Trained Model 3 (11 features)
├── jan to may police violation_anonymized791b166.csv  # Dataset
├── frontend/                               # Web frontend
│   ├── index.html                          # Main UI with tabs
│   ├── styles.css                          # Styling with gradient theme
│   └── script.js                           # JavaScript for API calls
└── doc/                                    # Documentation
    └── backend-documentation.html           # Technical documentation
```

## 🛠️ Technologies Used

- **ML Framework**: LightGBM, XGBoost, Scikit-learn, TensorFlow/Keras
- **Clustering**: DBSCAN
- **API**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, CSS3, JavaScript
- **Optimization**: Optuna
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Render (backend), Netlify (frontend)
- **Version Control**: Git, GitHub

## 👨‍💻 Author

**Puneet Chaturvedi**
**Team**: TrafficMind Solutions

ML Internship Project - Parking Intelligence System

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

This is a portfolio project. Suggestions and improvements are welcome!

## 📞 Contact

- **Email**: chaturvedipuneet200@gmail.com
- **GitHub**: https://github.com/Puneet04-tech/parking-model
- **Live Demo**: https://parking-intelligence-api.netlify.app
- **API**: https://parking-model.onrender.com
