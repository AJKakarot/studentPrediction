#!/usr/bin/env python3
"""Predict grade category for new student data using the trained model."""
import os
import sys
import pandas as pd
import joblib

MODEL_DIR = "model"

def load_pipeline():
    """Load saved model and encoders."""
    model = joblib.load(os.path.join(MODEL_DIR, "student_performance_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le_target = joblib.load(os.path.join(MODEL_DIR, "label_encoder_target.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    use_scaled_path = os.path.join(MODEL_DIR, "use_scaled.pkl")
    use_scaled = joblib.load(use_scaled_path) if os.path.exists(use_scaled_path) else False
    return model, scaler, le_target, label_encoders, feature_cols, use_scaled

def prepare_data(df, label_encoders, feature_cols):
    """Encode categoricals and select features."""
    df = df.copy()
    for col in label_encoders:
        if col in df.columns:
            le = label_encoders[col]
            def _encode(x):
                x = str(x).strip()
                if x in le.classes_:
                    return le.transform([x])[0]
                return 0  # fallback for unseen
            df[col] = df[col].astype(str).apply(_encode)
    # Ensure feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0  # Fill missing with 0
    return df[feature_cols]

def predict_csv(csv_path):
    """Predict for all rows in a CSV file."""
    model, scaler, le_target, label_encoders, feature_cols, use_scaled = load_pipeline()
    df = pd.read_csv(csv_path, sep=";")
    if "G1" in df.columns:
        df = df.drop(columns=["G1", "G2", "G3"], errors="ignore")
    if "grade_category" in df.columns:
        df = df.drop(columns=["grade_category"], errors="ignore")
    X = prepare_data(df, label_encoders, feature_cols)
    if use_scaled:
        X = scaler.transform(X)
    preds = model.predict(X)
    labels = le_target.inverse_transform(preds)
    return labels

def predict_single(row_dict, model, scaler, le_target, label_encoders, feature_cols, use_scaled):
    """Predict for a single row (dict of column -> value)."""
    df = pd.DataFrame([row_dict])
    X = prepare_data(df, label_encoders, feature_cols)
    if use_scaled:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return le_target.inverse_transform([pred])[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_new_data.csv>")
        print("\nCSV format: same columns as student-mat.csv (semicolon-separated)")
        print("G1, G2, G3 columns optional (excluded from prediction)")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    labels = predict_csv(csv_path)
    for i, label in enumerate(labels):
        print(f"Row {i+1}: Predicted Grade = {label}")
    if len(labels) == 1:
        print(f"\nPredicted: {labels[0]}")

if __name__ == "__main__":
    main()
