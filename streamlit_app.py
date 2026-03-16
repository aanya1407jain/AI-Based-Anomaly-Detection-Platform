import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import IsolationForest
import psutil
from pathlib import Path
import json

# Allow large dataframe styling
pd.set_option("styler.render.max_elements", 2500000)

# ================= Load Pre-Trained Model (for system metrics) ===================
model_path = "model/anomaly_model.pkl"
pretrained_model = load(model_path)

# =============== Page Layout ===================
st.set_page_config(page_title="AI-Based Anomaly Detection", layout="wide")
st.title("AI-Based Anomaly Detection Platform")
st.markdown("""
This application detects anomalies in your datasets or using real-time system metrics (CPU, Memory, Disk).
Works with **any numeric columns** in uploaded files.
""")

# ================== FILE UPLOAD =================
st.header("📁 Upload Dataset for Anomaly Detection")
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

result_df = None  # initialize

if uploaded_file:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("**📝 Data Preview:**")
        st.dataframe(df.head())

        # Select numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.error("No numeric columns found. Upload a dataset with numeric features.")
        else:
            if st.button("➡️ Predict Anomaly for Upload"):

                # Fit IsolationForest on uploaded numeric data
                iso_model = IsolationForest(contamination=0.05, random_state=42)
                predictions = iso_model.fit_predict(numeric_df)
                numeric_df["Prediction"] = pd.Series(predictions).map({1: "NORMAL", -1: "ANOMALY"})

                # Include anomaly scores
                try:
                    numeric_df["Anomaly Score"] = iso_model.decision_function(numeric_df.select_dtypes(include=['number']).values)
                except:
                    numeric_df["Anomaly Score"] = None

                result_df = numeric_df  # for display/download

                # Highlight anomalies
                st.header("📌 Anomaly Detection Result")

                anomalies_df = result_df[result_df["Prediction"]=="ANOMALY"]

                # -------- Detected Anomalies (NO COLOR) ----------
                st.subheader(f"🚨 Detected Anomalies ({len(anomalies_df)})")

                if not anomalies_df.empty:
                    st.dataframe(anomalies_df)   # normal table (no coloring)
                else:
                    st.write("No anomalies detected in the uploaded dataset.")


                # -------- All Predictions (COLOR ONLY HERE) ----------
                st.subheader("📊 All Predictions (last 500 rows)")

                st.dataframe(result_df.tail(500))

                # Download full predictions
                st.download_button(
                    label="📥 Download Full Predictions",
                    data=result_df.to_csv(index=False),
                    file_name="anomaly_results.csv",
                    mime="text/csv"
                )

                # Logging
                log_file = Path("logs/prediction_log.json")
                log_file.parent.mkdir(parents=True, exist_ok=True)
                if not log_file.exists():
                    log_file.write_text("[]")

                try:
                    with log_file.open("r") as f:
                        logs = json.load(f)
                except:
                    logs = []

                logs.append({
                    "total_records": len(result_df),
                    "total_anomalies": int((result_df["Prediction"] == "ANOMALY").sum()),
                    "total_normal": int((result_df["Prediction"] == "NORMAL").sum())
                })

                with log_file.open("w") as f:
                    json.dump(logs, f, indent=4)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# =========== REAL-TIME SYSTEM METRICS ===========
st.header("🖥️ Real-Time System Metrics Prediction")
if st.button("📊 Use Current CPU, Memory & Disk"):
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    try:
        disk = psutil.disk_usage('/').percent
    except:
        disk = 0

    st.write(f"**📌 CPU:** {cpu}%, **💾 Memory:** {memory}%, **📀 Disk:** {disk}%")

    x = np.array([[cpu, memory, disk]])
    pred = pretrained_model.predict(x)[0]
    try:
        score = pretrained_model.decision_function(x)[0]
    except:
        score = None

    status = "🚨 ANOMALY" if pred == -1 else "✔️ NORMAL"
    st.write(f"**📍 Prediction:** {status}")
    if score is not None:
        st.write(f"**📊 Anomaly Score:** {score:.4f}")

    # Logging
    log_file = Path("logs/prediction_log.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        log_file.write_text("[]")
    with log_file.open("r+") as f:
        logs = json.load(f)
        logs.append({
            "features": [cpu, memory, disk],
            "prediction": status,
            "score": float(score) if score else None
        })
        f.seek(0)
        json.dump(logs, f, indent=4)

# =============== Prediction Log ===============
st.header("📌 Recent Predictions Log")
log_file = Path("logs/prediction_log.json")
if log_file.exists():
    with log_file.open() as f:
        logs = json.load(f)
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs.tail(10))
        else:
            st.write("No predictions logged yet.")