import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime

# Basic page config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Predictor")
st.write("Predict the approximate price of a laptop in INR based on its configuration.")

# Load the model and data
try:
    pipe = joblib.load("pipe.pkl")
    data = joblib.load("data.pkl")
except FileNotFoundError:
    st.error("Model or data file not found. Please make sure 'pipe.pkl' and 'data.pkl' are present.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# ========== INPUT WIDGETS ==========

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", sorted(data["Company"].unique()))
    type_ = st.selectbox("Type", sorted(data["TypeName"].unique()))
    ram = st.selectbox("RAM (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    ips = st.selectbox("IPS", ["No", "Yes"])
    os_ = st.selectbox("OS", sorted(data["os"].unique()))

with col2:
    weight = st.number_input(
        "Weight of the Laptop (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Typical laptops are between 1.0–3.0 kg"
    )
    screen_size = st.slider(
        "Screen Size (in inches)",
        min_value=10.0,
        max_value=18.0,
        value=13.0,
        step=0.1,
        help="13–15 inches is most common for consumer laptops"
    )
    resolution = st.selectbox(
        "Screen Resolution",
        [
            "1920x1080",
            "1366x768",
            "1600x900",
            "3840x2160",
            "3200x1800",
            "2880x1800",
            "2560x1600",
            "2560x1440",
            "2304x1440",
        ],
    )
    cpu = st.selectbox("CPU", sorted(data["cpu_brand"].unique()))
    hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox("GPU", sorted(data["GpuBrand"].unique()))

st.markdown("---")

# ========== PREDICTION BUTTON ==========
reset = st.button("Reset inputs")

if reset:
    st.experimental_rerun()


if st.button("Predict Price"):
    try:
        # Basic validation
        if hdd == 0 and ssd == 0:
            st.warning("Please set at least some HDD or SSD storage before predicting.")
        else:
            # Convert categorical yes/no to numeric
            touchscreen_val = 1 if touchscreen == "Yes" else 0
            ips_val = 1 if ips == "Yes" else 0

            # Calculate PPI from resolution and screen size
            X_res = int(resolution.split("x")[0])
            Y_res = int(resolution.split("x")[1])
            ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

            # Create query in the expected order
            query = np.array(
                [
                    company,
                    type_,
                    ram,
                    weight,
                    touchscreen_val,
                    ips_val,
                    ppi,
                    cpu,
                    hdd,
                    ssd,
                    gpu,
                    os_,
                ],
                dtype=object,
            ).reshape(1, -1)

            # Predict
            prediction = pipe.predict(query)

            # If model was trained on log(price), convert back with exp
            if np.all(np.isfinite(prediction)):
                price = float(prediction[0])
                if price < 1e5:  # simple heuristic to guess if log was used
                    actual_price = int(np.exp(price))
                else:
                    actual_price = int(price)
            else:
                actual_price = int(prediction[0])

            # Round to nearest 1000 and format
            rounded_price = int(round(actual_price / 1000)) * 1000
            st.subheader(
                f"The predicted price of this configuration is **₹{rounded_price:,.0f}** (approx)"
            )

            # ========== LOGGING ==========

            log_dict = {
                "Company": company,
                "TypeName": type_,
                "RAM_GB": ram,
                "Weight_kg": weight,
                "Touchscreen": touchscreen_val,
                "IPS": ips_val,
                "PPI": ppi,
                "CPU": cpu,
                "HDD_GB": hdd,
                "SSD_GB": ssd,
                "GPU": gpu,
                "OS": os_,
                "predicted_price": actual_price,
                "predicted_price_rounded": rounded_price,
                "timestamp": datetime.now().isoformat(),
            }

            log_df = pd.DataFrame([log_dict])
            log_path = "prediction_logs.csv"

            if os.path.exists(log_path):
                existing = pd.read_csv(log_path)
                updated = pd.concat([existing, log_df], ignore_index=True)
                updated.to_csv(log_path, index=False)
            else:
                log_df.to_csv(log_path, index=False)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ========== ABOUT MODEL SECTION ==========

with st.expander("ℹ️ About this model"):
    st.write(
        """
        - Trained on laptop configuration data with features like brand, RAM, CPU, GPU, storage, screen size, and weight.  
        - Uses a machine learning regression model to estimate price in Indian Rupees (INR).  
        - Predictions are approximate and may differ from actual market prices.
        """
    )
    st.markdown("---")
    st.caption("Made by Md Rizwan · Laptop Price Predictor (ML & Streamlit)")

