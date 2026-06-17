# Parking Congestion Intelligence System

AI-driven parking violation detection and enforcement prioritization system for smart city traffic management.

## 🚀 Project Overview

This system uses machine learning to:
1. **Automate Report Validation**: Detect and filter invalid parking violation reports using LightGBM binary classification
2. **Identify Spatial Hotspots**: Cluster violation coordinates using DBSCAN to detect chronic violation zones
3. **Prioritize Enforcement**: Predict congestion impact and dispatch priority using multi-class classification

## 📊 Dataset

- **Source**: Bangalore Police violation records (Jan-May 2024)
- **Size**: 298,450 parking violation reports
- **Features**: Location, vehicle type, violation types, timestamps, reporter metadata

## 🏗️ Architecture

### Model Pipeline

1. **Model 1 - Report Validator** (Binary Classification)
   - Input: 32 enhanced features (spatial, temporal, violation types, reporter reliability, user behavior, device patterns)
   - Output: Approved/Rejected
   - **Original Accuracy**: 75.77% (ROC-AUC: 0.7582)
   - **Improved Accuracy**: >85% (target) with semi-supervised learning
   - **Key Improvements**:
     - Enhanced feature engineering (10 new features)
     - Class imbalance handling with weighted training
     - Semi-supervised learning using 125K unlabeled samples
     - Hyperparameter optimization with Optuna
     - Neural network ensemble comparison

2. **Model 2 - Hotspot Detector** (Unsupervised Clustering)
   - Algorithm: DBSCAN
   - Output: 101 distinct parking hotspots
   - Coverage: 72.19% of reports in hotspots

3. **Model 3 - Priority Predictor** (Multi-class Classification)
   - Input: 12 ambient features (location, time, area density)
   - Output: Low/Medium/High priority
   - Accuracy: 85.45% (10-fold CV: 85.42% ± 0.01)

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

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook parking_congestion_intelligence.ipynb

# Export models
# Run the model export cell in the notebook
```

### API Deployment (Render)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Render**
   - Create a new Web Service on Render
   - Connect your GitHub repository
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
     - **Python Version**: 3.11

3. **API Endpoints**

```bash
# Health check
GET /

# Predict priority
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
- **Original Accuracy**: 75.77%
- **Original ROC-AUC**: 0.7582
- **Improved Accuracy**: >85% (target with semi-supervised learning)
- **Improved ROC-AUC**: >0.90 (target)
- **Training Data**: 5K labeled + ~50K pseudo-labeled (from 125K unlabeled)
- **Features**: 32 enhanced features

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
├── parking_congestion_intelligence.ipynb  # Main notebook
├── app.py                                  # FastAPI application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── model_report_validator.joblib           # Trained Model 1 (Improved)
├── model_report_validator_final.joblib     # Final Model 1 (Semi-Supervised)
├── model_congestion_prioritizer.joblib     # Trained Model 3
├── ambient_features.json                   # Feature list for Model 3
├── model1_features.json                    # Feature list for Model 1
└── jan to may police violation_anonymized791b166.csv  # Dataset
```

## 🛠️ Technologies Used

- **ML Framework**: LightGBM, XGBoost, Scikit-learn, TensorFlow/Keras
- **Clustering**: DBSCAN
- **API**: FastAPI, Uvicorn
- **Optimization**: Optuna
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## 👨‍💻 Author

Built for ML Internship Application - Parking Intelligence System

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

This is a portfolio project. Suggestions and improvements are welcome!

## 📞 Contact

For questions about this project, please open an issue on GitHub.
