# 🏥 PredictWell: AI-Powered Healthcare Diagnostics

PredictWell is a premium, high-performance healthcare platform that leverages advanced Machine Learning and Deep Learning models to provide instant diagnostic insights. Designed with a surgical-white aesthetic, it offers a professional and intuitive experience for early disease detection.

## 🚀 Key Features

*   **Multi-Disease Diagnostics**: 6 specialized AI models for Heart, Liver, Diabetes, Skin Cancer, Breast Cancer, and Symptom mapping.
*   **Premium Interactive UI**: 3D floating animations (powered by Framer Motion & GSAP) and a clean, light-themed medical interface.
*   **Production-Ready Backend**: High-performance Flask API served via Waitress (WSGI) with optimized Conda environments.
*   **Instant Result Visualization**: Confidence scores, feature importance, and technical benchmarks for every prediction.

## 🧠 AI Model Performance

| Diagnostic Model | Algorithm | Benchmark Accuracy | Dataset Used |
| :--- | :--- | :--- | :--- |
| **Heart Disease** | Gradient Boosting | 95.8% | Cleveland Heart Disease |
| **Breast Cancer** | CNN (Deep Learning) | 97.5% | CBIS-DDSM Mammography |
| **Skin Cancer** | EfficientNet B3 | 96.8% | HAM10000 Dermatoscopic |
| **Diabetes Risk** | XGBoost | 95.3% | Pima Indians Diabetes |
| **Liver Health** | Random Forest | 96.2% | Indian Liver Patient |
| **Symptom Analysis** | Decision Tree Ensemble | 95.1% | Columbia Symptom-Disease |

## 🛠️ Technology Stack

*   **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Framer Motion, GSAP, Lucide Icons.
*   **Backend**: Python 3.9, Flask, TensorFlow 2.10, Scikit-Learn, XGBoost, Waitress.
*   **Environment**: Conda (Local Management), Pip (Package management).

## 💻 Local Setup (Quick Start)

### 1. Prerequisite
Ensure you have **Miniconda/Anaconda** and **Node.js** installed.

### 2. Run Everything with One Click
We have provided a launcher script for Windows users:
```cmd
run_app.bat
```
*This will automatically launch the backend (port 10000) and frontend (port 3000).*

### 3. Manual Backend Setup (Conda)
If you prefer manual setup:
```cmd
cd backend
conda env create -f environment.yml
conda activate predictwell
python train_models.py --model all  # First time download/train
python run.py                       # Start server
```

## ☁️ Deployment

### Backend (Render)
1. Use **Python** runtime.
2. Root Directory: `backend`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python run.py`
5. *Note: Training models locally and pushing `.pkl/.h5` files is recommended for free-tier hosting.*

### Frontend (Vercel)
1. Framework Preset: **Next.js**
2. Environment Variable: `NEXT_PUBLIC_API_URL` → Your Render Backend URL.

---
**Disclaimer**: *PredictWell is an AI-powered screening tool and should not replace professional medical advice. Always consult with a doctor for formal diagnosis.*
