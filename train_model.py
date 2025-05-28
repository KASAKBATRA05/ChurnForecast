# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_excel("Telco_customer_churn.xlsx")

# Drop unnecessary columns
drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude', 'Churn Reason'
]
df.drop(columns=drop_cols, inplace=True)

# Clean 'Total Charges'
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')
df.dropna(inplace=True)

# Encode target
df["Churn Label"] = df["Churn Label"].map({"Yes": 1, "No": 0})

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

# Define features and target
X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]
feature_names = list(X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with scaler + XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save pipeline and metadata
joblib.dump(pipeline, "pipeline.pkl")
joblib.dump(feature_names, "feature_names.pkl")

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
