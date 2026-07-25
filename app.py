import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")

# Paste the new code here 

customer_id = st.number_input("Customer ID", min_value=1)

age = st.number_input("Age", min_value=18, max_value=100)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

subscription = st.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Standard"]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0
)

tickets = st.number_input(
    "Support Tickets",
    min_value=0
)

payment = st.selectbox(
    "Payment Method",
    ["Bank Transfer", "Credit Card", "PayPal"]
)

score = st.slider(
    "Satisfaction Score",
    1,
    5
)

usage = st.number_input(
    "Usage Hours Per Week",
    min_value=0
)

predict = st.button("Predict")

if predict:

    gender = 1 if gender == "Male" else 0

    subscription = {
        "Basic": 0,
        "Standard": 1,
        "Premium": 2
    }[subscription]

    payment = {
        "Bank Transfer": 0,
        "Credit Card": 1,
        "PayPal": 2
    }[payment]

    input_data = pd.DataFrame([{
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "SubscriptionType": subscription,
        "MonthlyCharges": monthly,
        "TenureMonths": tenure,
        "SupportTickets": tickets,
        "PaymentMethod": payment,
        "SatisfactionScore": score,
        "UsageHoursPerWeek": usage
    }])

    # imports

# load model

# Streamlit page

# input fields

# Predict button

if predict:
    # create input_data

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    prob = model.predict_proba(input_data)
    st.write(f"Confidence: {max(prob[0]) * 100:.2f}%")

# Footer
st.markdown("---")
st.caption("Developed by Sharath Chandra Chary Bairoju")