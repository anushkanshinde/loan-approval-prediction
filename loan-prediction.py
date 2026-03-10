import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =======================
# 1. Load data
# =======================
df = pd.read_csv(
    r"C:\Users\aditi\Downloads\Loan-Approval-Prediction - Loan-Approval-Prediction.csv"
)

# =======================
# 2. Drop ID column
# =======================
df.drop("Loan_ID", axis=1, inplace=True)

# =======================
# 3. Clean categorical text (DO NOT convert NaN to string)
# =======================
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

for col in cat_cols:
    df[col] = df[col].str.strip()

# Normalize text for mapping
df["Gender"] = df["Gender"].str.title()
df["Married"] = df["Married"].str.title()
df["Self_Employed"] = df["Self_Employed"].str.title()

# =======================
# 4. Handle missing categorical values
# =======================
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Education"] = df["Education"].fillna(df["Education"].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Property_Area"] = df["Property_Area"].fillna(df["Property_Area"].mode()[0])

# =======================
# 5. Fix Dependents column
# =======================
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# =======================
# 6. Numeric conversion (CRITICAL)
# =======================
num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric NaNs
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# =======================
# 7. Encode categorical variables
# =======================
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# =======================
# 8. One-hot encoding
# =======================
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

# =======================
# 9. Split data BEFORE scaling
# =======================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 10. Scale numerical columns
# =======================
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# =======================
# 11. Final verification
# =======================
print("Train nulls:\n", X_train.isnull().sum())
print("Test nulls:\n", X_test.isnull().sum())
print("Data type:", type(X_train))
print("\nSample data:")
print(X_train.head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))