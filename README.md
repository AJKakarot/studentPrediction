# Student Performance Prediction (ML)

**Task:** Predict student exam result / grade category  
**Dataset:** UCI Student Performance Data Set (Mathematics)

## Deliverables

- Data preprocessing & EDA
- Classification models: Logistic Regression, Random Forest, XGBoost
- Feature importance analysis
- Trained model + Accuracy + Confusion Matrix

## Setup

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|------|-------------|
| `student-mat.csv` | Mathematics course data (UCI) |
| `student_performance_prediction.ipynb` | Full notebook with EDA, models, feature importance |
| `run_pipeline.py` | Script to run full pipeline and save model |

## Quick Run

If `python3` gives `ModuleNotFoundError: No module named 'pandas'`, use the Python that has your packages:

```bash
# Option 1: Use run script (auto-detects Python)
./run.sh

# Option 2: Use Python 3.14 (if you used python.org installer)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 run_all.py

# Option 3: Or run_pipeline (requires XGBoost + libomp)
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 run_pipeline.py
```

For XGBoost on Mac, install OpenMP: `brew install libomp`

Output: Accuracy, confusion matrix, and trained model in `model/`.

## Jupyter Notebook

```bash
jupyter notebook student_performance_prediction.ipynb
```

Run all cells for EDA, preprocessing, model training, feature importance, and saving the best model.

## Grade Categories

- **Fail**: G3 < 10  
- **Pass**: 10 ≤ G3 < 14  
- **Good**: 14 ≤ G3 < 17  
- **Excellent**: G3 ≥ 17  

(G3 = final grade, 0–20 scale)

## Predict on New Data

**1. From CSV file**
```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 predict.py new_data_sample.csv
```

Create a CSV with same columns as `student-mat.csv` (semicolon `;` separator). Use `new_data_sample.csv` as a template.

**2. Web UI (browser form)**
```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 predict_app.py
```
Then open http://localhost:5001 — fill the form and click **Predict**.

## Deploy

**Prerequisites:** Run `python run_all.py` first so `model/` has trained artifacts and `results.html` exists.

### Docker (works on Render, Railway, Fly.io, AWS, etc.)

```bash
docker build -t student-predict .
docker run -p 5001:5001 student-predict
```

### Render.com (free tier)

1. Push repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect repo, select **Docker** as environment
4. Deploy (uses `Dockerfile` and `render.yaml`)

### Railway

1. Install [Railway CLI](https://docs.railway.app/develop/cli) or use dashboard
2. `railway init` → `railway up` (uses Dockerfile)
