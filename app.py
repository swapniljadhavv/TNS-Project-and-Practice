import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age")
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0,1])
cp = st.number_input("Chest Pain Type (0-3)")
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.number_input("Fasting Blood Sugar (0/1)")
restecg = st.number_input("Resting ECG (0-2)")
thalach = st.number_input("Max Heart Rate")
exang = st.number_input("Exercise Induced Angina (0/1)")
oldpeak = st.number_input("ST Depression")
slope = st.number_input("Slope (0-2)")
ca = st.number_input("Number of Major Vessels (0-3)")
thal = st.number_input("Thal (1-3)")

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")