# ❤️ CardioSense AI — Heart Disease Prediction Platform

A **production-ready** Streamlit web application for heart disease prediction using a trained **RBF-SVM** model.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (one-time)
```bash
python train_model.py
```
This generates `models/svm_model.pkl`, `models/scaler.pkl`, and `models/metrics.json`.

### 3. Run the App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
heart_disease_app/
├── app.py              ← Main Streamlit application
├── train_model.py      ← Model training script (run once)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── models/
    ├── svm_model.pkl   ← Trained RBF-SVM model
    ├── scaler.pkl      ← Fitted StandardScaler
    ├── feature_cols.pkl← Feature column list
    └── metrics.json    ← Model evaluation metrics
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | SVM with RBF Kernel |
| Hyperparameter Tuning | GridSearchCV |
| Cross-Validation | 5-Fold Stratified |
| Test Accuracy | ~83-86% |
| ROC-AUC | ~0.90-0.93 |
| Feature Scaling | StandardScaler |
| Probability Output | Platt Scaling |

---

## 🔬 Feature Engineering

| Feature | Description |
|---------|-------------|
| `age_risk` | Age bucketed: <40, 40-50, 50-60, >60 |
| `chol_risk` | Cholesterol: Optimal / Borderline / High |
| `bp_risk` | BP: Normal / Elevated / Stage-1 / Stage-2 |
| `age_chol` | Interaction: age × cholesterol |
| `bp_hr` | Interaction: blood pressure × max heart rate |

---

## 🌐 Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as main file
4. Add `requirements.txt` — deploy!

> ⚠️ **Disclaimer**: For research and educational use only. Not a clinical diagnostic tool.
