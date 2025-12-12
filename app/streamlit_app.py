import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------
# Load model + preprocessor
# ---------------------------
MODEL_PATH = os.path.join("models", "best_churn_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

# ---------------------------
# App layout
# ---------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title(" Customer Churn Predictor")
st.write("Fill the customer information below and click **Predict**.")

# ---------------------------
# Input form
# ---------------------------
with st.form(key="churn_form"):
    st.subheader("Customer Information")

    # Required fields noticed missing from preprocessor
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=1)

    # Add Unnamed: 0 (dummy, because preprocessor expects it)
    unnamed_0 = 0  # Always 0 – only needed to satisfy preprocessor columns

    # Basic info
    customer_id = st.text_input("Customer ID", value="CUST_0001")
    name = st.text_input("Name", value="Jane Doe")

    # Numeric fields
    age = st.number_input("Age", min_value=10, max_value=120, value=30)
    tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=600, value=12)

    # Categorical
    region_category = st.text_input("Region category", value="Urban")
    membership_category = st.selectbox("Membership category", ["Basic", "Silver", "Gold", "Platinum"], index=1)
    joined_through_referral = st.selectbox("Joined through referral?", ["Yes", "No"], index=1)
    referral_id = st.text_input("Referral ID", value="")
    preferred_offer_types = st.text_input("Preferred offer types", value="Discount")
    medium_of_operation = st.selectbox("Medium of operation", ["Desktop", "Mobile", "Both"], index=2)
    internet_option = st.selectbox("Internet option", ["Fiber", "DSL", "Mobile", "None"], index=2)

    # Dates / Activity
    joining_date = st.date_input("Joining date")
    last_visit_time = st.date_input("Last visit date")
    days_since_last_login = st.number_input("Days since last login", min_value=0, value=3)
    avg_time_spent = st.number_input("Average time spent (minutes)", min_value=0.0, value=15.0)
    avg_transaction_value = st.number_input("Average transaction value", min_value=0.0, value=45.0)
    avg_frequency_login_days = st.number_input("Average frequency login days", min_value=0.0, value=7.0)
    points_in_wallet = st.number_input("Points in wallet", min_value=0, value=120)

    used_special_discount = st.selectbox("Used special discount?", ["Yes", "No"], index=1)
    offer_application_preference = st.selectbox("Offer application preference?", ["Yes", "No"], index=1)
    past_complaint = st.selectbox("Past complaint?", ["Yes", "No"], index=1)
    complaint_status = st.selectbox("Complaint resolved?", ["Yes", "No", "N/A"], index=2)
    feedback = st.text_area("Feedback", value="")

    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction
# ---------------------------
if submitted:
    try:
        # Build sample row with ALL columns expected by preprocessor
        sample = {
            "unnamed:_0": unnamed_0,
            "customer_id": customer_id,
            "name": name,
            "gender": gender,
            "age": float(age),
            "security_no": "",
            "region_category": region_category,
            "membership_category": membership_category,
            "joining_date": pd.to_datetime(joining_date),
            "joined_through_referral": joined_through_referral,
            "referral_id": referral_id,
            "preferred_offer_types": preferred_offer_types,
            "medium_of_operation": medium_of_operation,
            "internet_option": internet_option,
            "last_visit_time": pd.to_datetime(last_visit_time),
            "days_since_last_login": float(days_since_last_login),
            "avg_time_spent": float(avg_time_spent),
            "avg_transaction_value": float(avg_transaction_value),
            "avg_frequency_login_days": float(avg_frequency_login_days),
            "points_in_wallet": float(points_in_wallet),
            "used_special_discount": used_special_discount,
            "offer_application_preference": offer_application_preference,
            "past_complaint": past_complaint,
            "complaint_status": complaint_status,
            "feedback": feedback,
            "churn_risk_score": np.nan
        }

        sample_df = pd.DataFrame([sample])

        # Drop target
        X_sample = sample_df.drop(columns=["churn_risk_score"])

        # Preprocess
        X_prepared = preprocessor.transform(X_sample)

        # Predict
        pred = model.predict(X_prepared)[0]
        prob = model.predict_proba(X_prepared)[0][1]

        # Results
        st.subheader(" Prediction Results")
        st.write(f"**Predicted churn:** `{int(pred)}` (0 = active, 1 = churn)")
        st.write(f"**Churn probability:** {prob:.2%}")

        if pred == 1:
            st.warning("⚠️ High churn risk detected.")
        else:
            st.success("✅ Customer predicted to stay.")

        with st.expander("Show Input Data"):
            st.dataframe(sample_df.T)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        raise
