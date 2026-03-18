# Student Performance Prediction - Flask app
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model artifacts (run run_all.py first to generate model/)
COPY predict_app.py .
COPY model/ model/
COPY results.html .

EXPOSE 5001

# Render sets PORT at runtime; gunicorn needed for production
ENV PORT=5001
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5001} --workers 1 predict_app:app"]
