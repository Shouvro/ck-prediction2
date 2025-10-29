# CKD Prediction Deployment

This folder contains a small Flask application and supporting assets to serve the provided `pipeline.joblib` model for Chronic Kidney Disease prediction.

## Prerequisites

- Python 3.10+
- The `pipeline.joblib` file placed in the project root (or set the `MODEL_PATH` environment variable to point to it).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Open your browser at http://127.0.0.1:5000/ to access the HTML form. Fill in the clinical measurements and submit to view the model output and predicted probability (if available).

### JSON API

The same model is exposed via a JSON endpoint:

```bash
curl -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{ \"age\": 55, \"blood_pressure\": 80, ... }"
```

Ensure all 24 features listed in `app.py` are supplied. Numeric values should be numbers, categorical values must match the options in the HTML form (e.g. `yes/no`, `normal/abnormal`).

## Project Structure

- `app.py`: Flask application that loads the trained pipeline and handles predictions.
- `templates/index.html`: Jinja template for the web form UI.
- `static/css/styles.css`: Minimal styling for the form.
- `requirements.txt`: Python dependencies pinned for compatibility with the trained model.

## Notes

- The model loads once when the server starts. Restart the server after replacing `pipeline.joblib`.
- If you move the model file, set `MODEL_PATH` before launching: `set MODEL_PATH=C:\path\to\pipeline.joblib`.
- The current environment pins `xgboost==1.4.2` to match the serialized model. Do not upgrade unless you re-export the model with the newer version.
