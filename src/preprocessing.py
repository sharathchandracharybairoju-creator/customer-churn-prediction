import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Original Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "SubscriptionType",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Save processed dataset
df.to_csv("data/customer_churn_processed.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print(df.head())