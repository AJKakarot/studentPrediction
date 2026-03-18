#!/usr/bin/env python3
"""Flask app for predicting student grade from web form."""
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, render_template_string, jsonify, send_from_directory, send_file

app = Flask(__name__)

# Load model once at startup
model = None
results_meta = None
scaler = None
le_target = None
label_encoders = None
feature_cols = None
use_scaled = False

def load_model():
    global model, scaler, le_target, label_encoders, feature_cols, use_scaled
    import joblib
    model = joblib.load("model/student_performance_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le_target = joblib.load("model/label_encoder_target.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    try:
        use_scaled = joblib.load("model/use_scaled.pkl")
    except Exception:
        use_scaled = False

def predict_one(row):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame([row])
    for col in label_encoders:
        if col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str).replace('nan', 'other'))
            except Exception:
                df[col] = 0
    X = df[feature_cols] if all(c in df.columns for c in feature_cols) else df[[c for c in feature_cols if c in df.columns]]
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]
    if use_scaled:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return le_target.inverse_transform([pred])[0]

def get_model_results():
    """Load model accuracy/results for display."""
    import json
    p = "model/results.json"
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return None

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predict Student Grade</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #f0f9ff; }
    .container { max-width: 600px; margin: 0 auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #0c4a6e; margin-bottom: 0.5rem; }
    .subtitle { color: #64748b; margin-bottom: 1.5rem; }
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
    label { display: block; font-size: 0.875rem; font-weight: 500; color: #334155; margin-bottom: 0.25rem; }
    input, select { padding: 0.5rem 0.75rem; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 1rem; }
    .full { grid-column: 1 / -1; }
    button { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; font-size: 1rem; cursor: pointer; margin-top: 1rem; }
    button:hover { background: #0369a1; }
    .result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; font-size: 1.125rem; font-weight: 600; }
    .result.fail { background: #fee2e2; color: #b91c1c; }
    .result.pass { background: #fef3c7; color: #b45309; }
    .result.good { background: #d1fae5; color: #047857; }
    .result.excellent { background: #dbeafe; color: #1d4ed8; }
    .error { background: #fee2e2; color: #b91c1c; padding: 1rem; border-radius: 8px; margin-top: 1rem; }
    a { color: #0284c7; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Predict Student Grade</h1>
    <p class="subtitle">Enter key student data to predict grade (Fail / Pass / Good / Excellent)</p>
    <form method="POST" action="/predict">
      <div class="form-row">
        <div><label>Age</label><input type="number" name="age" value="16" min="15" max="22"></div>
        <div><label>Study Time (1-4)</label><input type="number" name="studytime" value="2" min="1" max="4"></div>
      </div>
      <div class="form-row">
        <div><label>Mother Education (0-4)</label><input type="number" name="Medu" value="2" min="0" max="4"></div>
        <div><label>Father Education (0-4)</label><input type="number" name="Fedu" value="2" min="0" max="4"></div>
      </div>
      <div class="form-row">
        <div><label>Failures (0-4)</label><input type="number" name="failures" value="0" min="0" max="4"></div>
        <div><label>Absences</label><input type="number" name="absences" value="0" min="0" max="93"></div>
      </div>
      <input type="hidden" name="school" value="GP">
      <input type="hidden" name="sex" value="M">
      <input type="hidden" name="address" value="U">
      <input type="hidden" name="famsize" value="GT3">
      <input type="hidden" name="Pstatus" value="T">
      <input type="hidden" name="Mjob" value="teacher">
      <input type="hidden" name="Fjob" value="other">
      <input type="hidden" name="reason" value="course">
      <input type="hidden" name="guardian" value="mother">
      <input type="hidden" name="traveltime" value="1">
      <input type="hidden" name="schoolsup" value="no">
      <input type="hidden" name="famsup" value="yes">
      <input type="hidden" name="paid" value="no">
      <input type="hidden" name="activities" value="no">
      <input type="hidden" name="nursery" value="yes">
      <input type="hidden" name="higher" value="yes">
      <input type="hidden" name="internet" value="yes">
      <input type="hidden" name="famrel" value="4">
      <input type="hidden" name="freetime" value="3">
      <input type="hidden" name="goout" value="3">
      <input type="hidden" name="Dalc" value="1">
      <input type="hidden" name="Walc" value="1">
      <input type="hidden" name="health" value="3">
      <button type="submit">Predict Grade</button>
    </form>
    {% if prediction %}
    <div class="result {{ prediction|lower }}">Predicted Grade: {{ prediction }}</div>
    {% endif %}
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <p style="margin-top: 1.5rem;"><a href="/results">View Full Results</a> | <a href="/">Predict Again</a></p>
  </div>
</body>
</html>'''

@app.route("/")
def index():
    return render_template_string(HTML, prediction=None, error=None, model_results=get_model_results())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {k: v for k, v in request.form.items()}
        pred = predict_one(row)
        return render_template_string(HTML, prediction=pred, error=None, model_results=get_model_results())
    except Exception as e:
        return render_template_string(HTML, prediction=None, error=str(e), model_results=get_model_results())

@app.route("/results")
def results():
    """Serve the results.html page."""
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.html")
    if os.path.exists(results_path):
        return send_file(results_path, mimetype="text/html")
    return "Results not found. Run run_all.py first.", 404

@app.route("/model/<path:filename>")
def serve_model(filename):
    """Serve files from model/ (e.g. confusion_matrix.png)."""
    return send_from_directory("model", filename)

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5001))  # Cloud platforms set PORT
    print(f"Predict app at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
