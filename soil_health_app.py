import streamlit as st
import joblib
import numpy as np

st.title("🧪 Soil pH Prediction")

model = joblib.load("D:/java/acR/soil_pH_model.pkl")
scaler = joblib.load("D:/java/acR/soil_pH_scaler.pkl")

crop_type = st.number_input("Crop Type (Encoded)", value=1)
fertilizer = st.number_input("Fertilizer Usage (kg)", value=50.0)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

if st.button("Predict Soil pH"):
    features = np.array([[crop_type, fertilizer, rainfall]])
    prediction = model.predict(scaler.transform(features))
    st.success(f"🧪 Predicted Soil pH: {prediction[0]:.2f}")
