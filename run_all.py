#!/usr/bin/env python3
"""Run full pipeline + save confusion matrix. Handles XGBoost/libomp failure."""
import os

os.makedirs("model", exist_ok=True)

# Try XGBoost; skip if libomp/etc. fails
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    print("Note: XGBoost skipped (install libomp: brew install libomp). Using LR + RF only.")
    HAS_XGB = False

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# --- Load & Preprocess ---
df = pd.read_csv("student-mat.csv", sep=";")
for col in ["G1", "G2", "G3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Dataset shape:", df.shape)

def grade_to_category(g):
    if g < 10: return "Fail"
    elif g < 14: return "Pass"
    elif g < 17: return "Good"
    else: return "Excellent"

df["grade_category"] = df["G3"].apply(grade_to_category)

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

# --- Train Models ---
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), False),
]
if HAS_XGB:
    models.append(("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=42), False))

results = []
for name, model, use_scaled in models:
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results.append((name, acc, cm, model))
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

best_idx = np.argmax([r[1] for r in results])
best_name, best_acc, best_cm, best_model = results[best_idx]
print(f"\n--- Best Model: {best_name} (Accuracy: {best_acc:.4f}) ---")

# --- Save model artifacts ---
use_scaled_best = models[best_idx][2]  # True for Logistic Regression only
joblib.dump(best_model, "model/student_performance_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le_target, "model/label_encoder_target.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(feature_cols, "model/feature_cols.pkl")
joblib.dump(use_scaled_best, "model/use_scaled.pkl")
print("\nSaved: model/*.pkl")

# --- Save confusion matrix PNG ---
try:
    import matplotlib
    matplotlib.use("Agg")  # No GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model/confusion_matrix.png")
except Exception as e:
    print("Could not save confusion matrix image:", e)

# --- Save results metadata for predict app UI ---
import json
results_meta = {
    "best_model": best_name,
    "best_accuracy": round(float(best_acc), 4),
    "model_comparison": [{"name": r[0], "accuracy": round(float(r[1]), 4)} for r in results],
    "test_samples": len(y_test),
}
with open("model/results.json", "w") as f:
    json.dump(results_meta, f, indent=2)
print("Saved: model/results.json")

# --- Generate HTML results page ---
def _cm_to_html(cm, labels):
    rows = []
    for i, row in enumerate(cm):
        cells = "".join(f'<td>{v}</td>' for v in row)
        rows.append(f'<tr><td><b>{labels[i]}</b></td>{cells}</tr>')
    header = "<tr><th></th>" + "".join(f'<th>{l}</th>' for l in labels) + "</tr>"
    return f'<table class="cm-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student Performance - Results</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #4ade80;
      --border: #334155;
    }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem;
      line-height: 1.6;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 0.5rem;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      background: var(--card);
      border-radius: 16px;
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      border: 1px solid var(--border);
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2), 0 2px 4px -2px rgba(0,0,0,0.15);
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{ box-shadow: 0 10px 15px -3px rgba(0,0,0,0.25), 0 4px 6px -4px rgba(0,0,0,0.2); }}
    .card h2 {{
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .card p {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1rem; }}
    .metrics {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
    .metric {{
      background: rgba(56, 189, 248, 0.1);
      padding: 1.25rem 1.5rem;
      border-radius: 12px;
      flex: 1;
      min-width: 140px;
      border: 1px solid var(--border);
      transition: background 0.2s;
    }}
    .metric:hover {{ background: rgba(56, 189, 248, 0.15); }}
    .metric-value {{ font-size: 1.75rem; font-weight: 700; color: var(--accent); }}
    .metric-label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .cm-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.9rem;
      border-radius: 12px;
      overflow: hidden;
    }}
    .cm-table th, .cm-table td {{
      border: 1px solid var(--border);
      padding: 0.65rem 0.85rem;
      text-align: center;
    }}
    .cm-table th {{ background: rgba(56, 189, 248, 0.15); font-weight: 600; color: var(--accent); }}
    .cm-table tbody tr:hover td {{ background: rgba(56, 189, 248, 0.08); }}
    .cm-table tbody tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
    img {{
      max-width: 100%;
      border-radius: 12px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .model-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem 0;
      border-bottom: 1px solid var(--border);
      transition: background 0.2s;
    }}
    .model-row:hover {{ background: rgba(255,255,255,0.03); }}
    .model-row:last-child {{ border-bottom: none; }}
    .best {{ color: var(--success); font-weight: 700; }}
    .best::after {{ content: ' ★'; font-size: 0.8em; }}
    @media (max-width: 640px) {{
      body {{ padding: 1rem; }}
      .metrics {{ flex-direction: column; }}
      .metric {{ min-width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Student Performance Prediction</h1>
    <p class="subtitle">ML Results • UCI Dataset</p>

    <div class="card">
      <h2>Accuracy Summary</h2>
      <div class="metrics">
        <div class="metric">
          <div class="metric-value">{best_acc*100:.2f}%</div>
          <div class="metric-label">Best Model Accuracy</div>
        </div>
        <div class="metric">
          <div class="metric-value">{best_name}</div>
          <div class="metric-label">Best Model</div>
        </div>
        <div class="metric">
          <div class="metric-value">{len(X_test)}</div>
          <div class="metric-label">Test Samples</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Model Comparison</h2>
      {"".join(f'<div class="model-row"><span>{r[0]}</span><span class="{"best" if r[1]==best_acc else ""}">{r[1]*100:.2f}%</span></div>' for r in results)}
    </div>

    <div class="card">
      <h2>Confusion Matrix ({best_name})</h2>
      <p>True vs Predicted labels (Fail, Pass, Good, Excellent)</p>
      {_cm_to_html(best_cm, le_target.classes_.tolist())}
    </div>

    <div class="card">
      <h2>Confusion Matrix Plot</h2>
      <img src="model/confusion_matrix.png" alt="Confusion Matrix" width="560">
    </div>
  </div>
</body>
</html>"""

with open("results.html", "w") as f:
    f.write(html_content)
print("Saved: results.html")

# --- Save results metadata for predict app ---
import json
results_meta = {
    "best_model": best_name,
    "best_accuracy": float(best_acc),
    "test_samples": int(len(X_test)),
    "models": [(r[0], float(r[1])) for r in results],
}
with open("model/results_meta.json", "w") as f:
    json.dump(results_meta, f, indent=2)
print("Saved: model/results_meta.json")

print("\nDone.")
