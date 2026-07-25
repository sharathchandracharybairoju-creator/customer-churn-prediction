import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of customers
num_customers = 3000

# Generate customer data
customer_data = {
    "CustomerID": np.arange(1, num_customers + 1),
    "Age": np.random.randint(18, 70, num_customers),
    "Gender": np.random.choice(["Male", "Female"], num_customers),
    "SubscriptionType": np.random.choice(
        ["Basic", "Premium", "Enterprise"],
        num_customers,
        p=[0.5, 0.3, 0.2]
    ),
    "MonthlyCharges": np.random.randint(200, 2000, num_customers),
    "TenureMonths": np.random.randint(1, 60, num_customers),
    "SupportTickets": np.random.randint(0, 10, num_customers),
    "PaymentMethod": np.random.choice(
        ["Credit Card", "UPI", "Net Banking"],
        num_customers
    ),
    "SatisfactionScore": np.random.randint(1, 11, num_customers),
    "UsageHoursPerWeek": np.random.randint(1, 50, num_customers),
}

# Churn logic
customer_data["Churn"] = np.where(
    (customer_data["SatisfactionScore"] < 5) &
    (customer_data["SupportTickets"] > 5),
    "Yes",
    "No"
)

# Create DataFrame
df = pd.DataFrame(customer_data)

# Save dataset
df.to_csv("data/customer_churn.csv", index=False)

print("Dataset created successfully!")
print(df.head())





