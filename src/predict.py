import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../model/churn_model.pkl")
print("Customer Churn Prediction")
print("-" * 30)

customer_id = int(input("Customer ID: "))
age = int(input("Age: "))
gender = int(input("Gender (0=Female, 1=Male): "))
subscription = int(input("Subscription Type (0=Basic, 1=Premium, 2=Standard): "))
monthly = float(input("Monthly Charges: "))
tenure = int(input("Tenure Months: "))
support = int(input("Support Tickets: "))
payment = int(input("Payment Method (0=Bank Transfer, 1=Credit Card, 2=PayPal): "))
satisfaction = int(input("Satisfaction Score (1-5): "))
usage = float(input("Usage Hours Per Week: "))

customer = pd.DataFrame([[
    customer_id,
    age,
    gender,
    subscription,
    monthly,
    tenure,
    support,
    payment,
    satisfaction,
    usage
]], columns=[
    "CustomerID",
    "Age",
    "Gender",
    "SubscriptionType",
    "MonthlyCharges",
    "TenureMonths",
    "SupportTickets",
    "PaymentMethod",
    "SatisfactionScore",
    "UsageHoursPerWeek"
])

prediction = model.predict(customer)

if prediction[0] == 1:
    print("\nPrediction: Customer will CHURN")
else:
    print("\nPrediction: Customer will NOT CHURN")