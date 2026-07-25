import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load processed dataset
data = pd.read_csv("../data/customer_churn_processed.csv")

# Features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../model/churn_model.pkl")

print("Model trained successfully!")
print("Model saved as ../model/churn_model.pkl")