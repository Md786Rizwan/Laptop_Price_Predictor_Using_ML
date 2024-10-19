import streamlit as st
import pickle
import numpy as np

# Load the model and data
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    data = pickle.load(open("data.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or data file not found. Please check the file path.")
    st.stop()

st.title("Laptop Price Predictor")

# Widgets for user input
company = st.selectbox('Brand', data['Company'].unique())
type = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop', min_value=0.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['cpu_brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', data['GpuBrand'].unique())
os = st.selectbox('OS', data['os'].unique())

if st.button('Predict Price'):
    try:
        # Convert inputs to expected format
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Create the query array
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)

        # Reshape the query array
        query = query.reshape(1, -1)  # Ensures dynamic column count based on the pipeline

        # Make prediction
        prediction = pipe.predict(query)

        # Handle potential log transformation (adjust if your model outputs actual price)
        predicted_price = int(np.exp(prediction[0])) if np.all(np.isfinite(prediction)) else int(prediction[0])
        st.title(f"The predicted price of this configuration is {predicted_price}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
