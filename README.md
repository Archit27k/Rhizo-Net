# 🌿 Rhizo-Net
## AI-Powered Rhizome Disease Detection System

Rhizo-Net is a professional AI-driven agricultural disease detection framework designed for early diagnosis of rhizome crop diseases using deep learning, computer vision, transfer learning, and morphological analysis.

The system combines:
- MobileNetV2 Transfer Learning
- OpenCV Morphological Analytics
- Bi-Phasic Decision Fusion
- TensorFlow Lite Edge Deployment
- Streamlit-based AI Dashboards
- Multilingual Farmer Assistance

---

# 🚀 Features

## 🌱 AI Disease Detection
- Ginger Bacterial Wilt Detection
- Ginger Pythium Soft Rot Detection
- Healthy Crop Classification
- Blotch Classification
- Leaf Spot Classification

---

## 🧠 Deep Learning Pipeline
- MobileNetV2 Backbone
- Cross-Domain Transfer Learning
- Maize-to-Ginger Knowledge Adaptation
- Frozen Lower Layers
- Fine-Tuned Upper Layers

---

## 🔬 Morphological Analysis
- HSV Segmentation
- Edge Density Estimation
- Canny Edge Detection
- Green Stress Analytics

---

## ⚡ Edge AI
- TensorFlow Lite Conversion
- Offline Inference
- Lightweight Deployment

---

## 🌍 Farmer Assistance
- Multilingual Support
- Voice Guidance
- Simplified Farmer Mode

---

## 📊 Expert Analytics Dashboard
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curves
- Confusion Matrix
- Confidence Distribution
- GPU Monitoring

---

# 🏗️ Project Structure

```text
rhizo-net/
│
├── app/
│   ├── streamlit_app.py
│   ├── farmer_mode.py
│   ├── expert_dashboard.py
│   └── localization/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── augmented/
│
├── models/
│   ├── checkpoints/
│   ├── saved/
│   └── tflite/
│
├── outputs/
│   └── evaluation/
│
├── scripts/
│   ├── download_maize.py
│   ├── organize_dataset.py
│   ├── augment_dataset.py
│   └── verify_dataset.py
│
├── src/
│   ├── morphology/
│   ├── fusion/
│   ├── training/
│   ├── voice/
│   └── database/
│
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone Repository
git clone <repository_url>
cd rhizo-net
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\Activate
3. Install Dependencies
pip install -r requirements.txt

📥 Dataset Preparation
Download Maize Dataset
python scripts/download_maize.py
Organize Dataset
python scripts/organize_dataset.py
Verify Dataset
python scripts/verify_dataset.py
Augment Ginger Dataset
python scripts/augment_dataset.py

🧠 Model Training
1. Train Maize Source Model
python src/training/train_maize.py
2. Fine-Tune on Ginger Dataset
python src/training/fine_tune_ginger.py
3. Convert to TensorFlow Lite
python src/training/convert_tflite.py
4. Evaluate Model
python src/training/evaluate_model.py

📊 Launch Applications
Main Streamlit App
python -m streamlit run app/streamlit_app.py
Farmer Mode
python -m streamlit run app/farmer_mode.py
Expert Dashboard
python -m streamlit run app/expert_dashboard.py

📈 Evaluation Metrics
The system evaluates:
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
Inference Time

🧩 Technologies Used
Python
TensorFlow
MobileNetV2
OpenCV
Streamlit
Scikit-Learn
NumPy
Pandas
Matplotlib

🌐 Future Enhancements
Grad-CAM Explainability
Real-Time Webcam Inference
Unknown Disease Rejection
Severity Estimation
Cloud Deployment
Mobile App Integration
Raspberry Pi Deployment
Multi-Crop Support

👨‍💻 Authors
Developed as part of an AI-driven agricultural healthcare initiative for rhizome disease detection and smart farming assistance.

📜 License
This project is intended for educational, research, and agricultural innovation purposes.