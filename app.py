import streamlit as st
import joblib
import pandas as pd

st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave.")

model = joblib.load("churn_model.pkl")
encoder = joblib.load("target_encoder.pkl")

st.success("Model loaded successfully!")
st.info("Next, we will add customer input fields and prediction.")
