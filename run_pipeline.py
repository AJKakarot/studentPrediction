#!/usr/bin/env python3
"""
Student Performance Prediction - Run full pipeline.
Outputs: Trained model, accuracy, confusion matrix.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib

os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv("student-mat.csv", sep=";")
# Ensure grade columns are numeric (CSV may read them as strings)
for col in ["G1", "G2", "G3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Dataset shape:", df.shape)

# Create grade category (Fail <10, Pass 10-13, Good 14-16, Excellent 17+)
def grade_to_category(g):
    if g < 10: return "Fail"
    elif g < 14: return "Pass"
    elif g < 17: return "Good"
    else: return "Excellent"

df["grade_category"] = df["G3"].apply(grade_to_category)

# Encode categoricals
categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "grade_category"]
df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

feature_cols = [c for c in df_encoded.columns if c not in ["G1", "G2", "G3", "grade_category"]]
X = df_encoded[feature_cols]
y = df["grade_category"]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
}

results = []
for name, model in models.items():
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled if name == "Logistic Regression" else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results.append((name, acc, cm, model))
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Best model
best_idx = np.argmax([r[1] for r in results])
best_name, best_acc, best_cm, best_model = results[best_idx]
print(f"\n--- Best Model: {best_name} (Accuracy: {best_acc:.4f}) ---")

# Save artifacts
joblib.dump(best_model, "model/student_performance_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_target, "model/label_encoder_target.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(feature_cols, "model/feature_cols.pkl")
joblib.dump(best_cm, "model/confusion_matrix.pkl")
print("\nSaved: model/student_performance_model.pkl, scaler, encoders, confusion_matrix")
