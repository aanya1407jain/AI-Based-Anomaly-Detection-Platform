🤖 AI-Based System Anomaly Detection
📌 Overview

This project implements an AI-powered system monitoring tool that detects abnormal behavior in system resources such as CPU usage, memory usage, and disk I/O. It uses machine learning to learn normal system patterns and identify unusual activity that may indicate performance issues or system faults.

The system collects real-time system metrics, trains an anomaly detection model, and exposes a REST API for predicting whether the current system state is normal or anomalous.

🚀 Features

Real-time system metric collection

Machine learning based anomaly detection

REST API for predictions

Automated testing script

Prediction logging system

Easy integration with monitoring tools

🧠 Machine Learning Model

This project uses IsolationForest from scikit-learn.

Isolation Forest detects anomalies by isolating unusual data points from normal patterns. If a system metric significantly differs from normal behavior learned during training, it is classified as an anomaly.

🛠️ Tech Stack

Python

FastAPI – API backend

scikit-learn – Machine learning

📂 Project Structure

AI-Anomaly-Detection
│
├── api
│   └── app.py                # FastAPI server
│
├── data
│   ├── collect_metrics.py    # Collect real system metrics
│   └── system_metrics.csv    # Dataset
│
├── model
│   ├── train_model.py        # Train anomaly detection model
│   └── anomaly_model.pkl     # Saved model
│
├── tester
│   └── tester.py             # Script to test API
│
├── logs
│   └── prediction_log.json   # Stores prediction logs
│
└── README.md

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/AI-Anomaly-Detection.git
cd AI-Anomaly-Detection

Create virtual environment:

python -m venv .venv

Activate environment:

Windows

.venv\Scripts\activate

Install dependencies:

pip install fastapi uvicorn scikit-learn pandas psutil requests
📊 Collect System Data

Generate system metrics dataset:

python data/collect_metrics.py

This will create:

data/system_metrics.csv
🧠 Train the Model

Train the anomaly detection model:

python model/train_model.py

This creates:

model/anomaly_model.pkl
🌐 Run the API

Start the API server:

uvicorn api.app:app --reload

Open API documentation:

http://127.0.0.1:8000/docs
📡 Example Prediction Request

Input:

cpu_usage = 70
memory_usage = 60
disk_io = 50

Output:

{
  "cpu_usage": 70,
  "memory_usage": 60,
  "disk_io": 50,
  "prediction": "NORMAL"
}

If the system behavior is unusual:

{
  "prediction": "ANOMALY"
}
🧪 Testing the API

Run the testing script:

python tester/tester.py

This sends multiple requests to the API and prints anomaly detection results.

🎯 Use Cases

Server performance monitoring

Detecting unusual system activity

Infrastructure monitoring tools

AI-powered system health analysis

📌 Future Improvements

Real-time monitoring dashboard

Alert system for anomalies

Visualization of system metrics

Integration with cloud monitoring tools

✅ Project Goal:
Demonstrate how machine learning can be integrated with system monitoring to automatically detect abnormal system behavior.

psutil – System metrics collection

Uvicorn – API server
