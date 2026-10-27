
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


@st.cache_resource
def load_models():
    model = joblib.load("churn_model.pkl")
    encoder = joblib.load("target_encoder.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, encoder, features


model, encoder, feature_columns = load_models()

st.title("📊 Customer Churn Prediction")
st.write("Estimate whether a customer may leave the company.")

mappings = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {
        "No": 0, "No phone service": 1, "Yes": 2
    },
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "OnlineBackup": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "DeviceProtection": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "TechSupport": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "StreamingTV": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "StreamingMovies": {
        "No": 0, "No internet service": 1, "Yes": 2
    },
    "Contract": {
        "Month-to-month": 0, "One year": 1, "Two year": 2
    },
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
}

with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        phone = st.selectbox("Phone Service", ["No", "Yes"])

        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "No phone service", "Yes"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["No", "No internet service", "Yes"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["No", "No internet service", "Yes"]
        )

    with col2:
        device = st.selectbox(
            "Device Protection",
            ["No", "No internet service", "Yes"]
        )
        support = st.selectbox(
            "Tech Support",
            ["No", "No internet service", "Yes"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["No", "No internet service", "Yes"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["No", "No internet service", "Yes"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["No", "Yes"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

    st.subheader("Account Details")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    customer = {
        "SeniorCitizen": int(senior == "Yes"),
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "gender": mappings["gender"][gender],
        "Partner": mappings["Partner"][partner],
        "Dependents": mappings["Dependents"][dependents],
        "PhoneService": mappings["PhoneService"][phone],
        "MultipleLines": mappings["MultipleLines"][multiple],
        "InternetService": mappings["InternetService"][internet],
        "OnlineSecurity": mappings["OnlineSecurity"][security],
        "OnlineBackup": mappings["OnlineBackup"][backup],
        "DeviceProtection": mappings["DeviceProtection"][device],
        "TechSupport": mappings["TechSupport"][support],
        "StreamingTV": mappings["StreamingTV"][tv],
        "StreamingMovies": mappings["StreamingMovies"][movies],
        "Contract": mappings["Contract"][contract],
        "PaperlessBilling": mappings["PaperlessBilling"][paperless],
        "PaymentMethod": mappings["PaymentMethod"][payment],
        "TotalCharges": total
    }

    input_df = pd.DataFrame([customer]).reindex(
        columns=feature_columns
    )

    # Match the model's expected feature order
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        input_df = input_df.reindex(columns=expected_features)

    probability = float(model.predict_proba(input_df)[0][1])

    predicted_label = encoder.inverse_transform([
        int(probability >= 0.35)
    ])[0]

    st.subheader("Prediction Result")
    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    if predicted_label == "Yes":
        st.warning("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted not to churn.")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "The 35% decision threshold is preliminary. "
        "Probability is a model estimate, not a guarantee."
    )
