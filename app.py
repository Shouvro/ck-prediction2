import os
from typing import Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "pipeline.joblib")


def _load_model():
    try:
        from xgboost.sklearn import XGBModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required to load the trained CKD model. "
            "Install the dependencies from requirements.txt."
        ) from exc

    try:
        loaded_model = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Unable to find model file at {MODEL_PATH}") from exc

    legacy_attrs = {
        "callbacks": None,
        "early_stopping_rounds": None,
        "eval_metric": None,
        "eval_set": None,
        "enable_categorical": False,
        "grow_policy": "depthwise",
        "max_bin": 256,
        "max_cat_to_onehot": 4,
    }

    def _ensure_legacy_attrs(obj):
        if isinstance(obj, XGBModel):
            for attr, value in legacy_attrs.items():
                if not hasattr(obj, attr):
                    setattr(obj, attr, value)

        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                _ensure_legacy_attrs(step)
        elif hasattr(obj, "steps"):
            for _, step in obj.steps:
                _ensure_legacy_attrs(step)

    _ensure_legacy_attrs(loaded_model)

    return loaded_model


model = _load_model()


FEATURE_ORDER: List[str] = [
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "red_blood_cells",
    "pus_cell",
    "pus_cell_clumps",
    "bacteria",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "haemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
    "hypertension",
    "diabetes_mellitus",
    "coronary_artery_disease",
    "appetite",
    "peda_edema",
    "aanemia",
]


NUMERIC_FEATURES = {
    "age",
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "sugar",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "sodium",
    "potassium",
    "haemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
}

CHOICES: Dict[str, List[str]] = {
    "red_blood_cells": ["abnormal", "normal"],
    "pus_cell": ["abnormal", "normal"],
    "pus_cell_clumps": ["notpresent", "present"],
    "bacteria": ["notpresent", "present"],
    "hypertension": ["no", "yes"],
    "diabetes_mellitus": ["no", "yes"],
    "coronary_artery_disease": ["no", "yes"],
    "appetite": ["good", "poor"],
    "peda_edema": ["no", "yes"],
    "aanemia": ["no", "yes"],
}

CATEGORY_ENCODINGS: Dict[str, Dict[str, int]] = {
    feature: {value: index for index, value in enumerate(options)}
    for feature, options in CHOICES.items()
}


def prepare_features(form_payload: Dict[str, str]) -> pd.DataFrame:
    record = {}
    errors = []

    for feature in FEATURE_ORDER:
        raw_value = (form_payload.get(feature) or "").strip()
        if raw_value == "":
            errors.append(f"Missing value for '{feature}'.")
            continue

        if feature in NUMERIC_FEATURES:
            try:
                record[feature] = float(raw_value)
            except ValueError:
                errors.append(f"'{feature}' must be numeric.")
        elif feature in CHOICES:
            normalized = raw_value.lower()
            if normalized not in CATEGORY_ENCODINGS[feature]:
                errors.append(
                    f"Invalid choice '{raw_value}' for '{feature}'. "
                    f"Expected one of {CHOICES[feature]}."
                )
            else:
                record[feature] = CATEGORY_ENCODINGS[feature][normalized]
        else:
            record[feature] = raw_value

    if errors:
        raise ValueError(" | ".join(errors))

    return pd.DataFrame([record], columns=FEATURE_ORDER)


def make_prediction(features: pd.DataFrame) -> Dict[str, float]:
    prediction_raw = model.predict(features)[0]
    prediction = (
        prediction_raw.item()
        if hasattr(prediction_raw, "item")
        else prediction_raw
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        positive_probability = float(probabilities[1])
    else:
        positive_probability = None

    return {
        "label": prediction,
        "probability": positive_probability,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = None

    if request.method == "POST":
        try:
            features = prepare_features(request.form)
            prediction = make_prediction(features)
            result = {
                "prediction": prediction["label"],
                "probability": prediction["probability"],
            }
        except ValueError as exc:
            errors = str(exc).split(" | ")

    return render_template(
        "index.html",
        categories=CHOICES,
        numeric_fields=NUMERIC_FEATURES,
        feature_order=FEATURE_ORDER,
        defaults=request.form,
        result=result,
        errors=errors,
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True) or {}
    try:
        features = prepare_features(payload)
        prediction = make_prediction(features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "prediction": prediction["label"],
            "probability": prediction["probability"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
