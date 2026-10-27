
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate the likelihood of churn.")

@st.cache_resource
def load_models():
    model = joblib.load("churn_model.pkl")
    encoder = joblib.load("target_encoder.pkl")
    return model, encoder

model, encoder = load_models()

st.success("Model loaded successfully!")

st.subheader("Customer Details")

with st.form("customer_form"):
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    customer_id = st.number_input(
        "Customer ID code", min_value=0, value=0
    )

    st.caption(
        "The categorical fields below use numeric codes from your "
        "training data. Their category-to-code mappings must match "
        "your original preprocessing."
    )

    gender = st.selectbox("Gender code", [0, 1])
    partner = st.selectbox("Partner code", [0, 1])
    dependents = st.selectbox("Dependents code", [0, 1])
    phone_service = st.selectbox("Phone Service code", [0, 1])
    multiple_lines = st.selectbox("Multiple Lines code", [0, 1, 2])
    internet_service = st.selectbox("Internet Service code", [0, 1, 2])
    online_security = st.selectbox("Online Security code", [0, 1, 2])
    online_backup = st.selectbox("Online Backup code", [0, 1, 2])
    device_protection = st.selectbox("Device Protection code", [0, 1, 2])
    tech_support = st.selectbox("Tech Support code", [0, 1, 2])
    streaming_tv = st.selectbox("Streaming TV code", [0, 1, 2])
    streaming_movies = st.selectbox("Streaming Movies code", [0, 1, 2])
    contract = st.selectbox("Contract code", [0, 1, 2])
    paperless_billing = st.selectbox("Paperless Billing code", [0, 1])
    payment_method = st.selectbox("Payment Method code", [0, 1, 2, 3])

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    columns = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "customerID",
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "TotalCharges"
    ]

    values = [[
        senior_citizen, tenure, monthly_charges, customer_id,
        gender, partner, dependents, phone_service,
        multiple_lines, internet_service, online_security,
        online_backup, device_protection, tech_support,
        streaming_tv, streaming_movies, contract,
        paperless_billing, payment_method, total_charges
    ]]

    customer = pd.DataFrame(values, columns=columns)

    probability = float(model.predict_proba(customer)[0][1])
    prediction = int(probability >= 0.35)
    churn_label = encoder.inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.1%}")

    if churn_label == "Yes":
        st.warning("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted not to churn.")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "The prediction uses a 0.35 probability threshold. "
        "Results depend on correct feature encoding and model quality."
    )
