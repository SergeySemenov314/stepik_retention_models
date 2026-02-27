"""
Stepik Retention Model - Inference API
Загружает лучшую XGBoost модель (XGB Best ROC-AUC) из notebook.
Использует 19 признаков: 13 базовых + 6 полиномиальных.
"""
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Stepik Retention Model API")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Порядок признаков как в X_train_final (base + selected_poly)
BASE_FEATURES = [
    'days', 'steps_tried', 'correct', 'wrong', 'correct_ratio', 'viewed', 'passed',
    'view_to_pass_ratio', 'first_try_ratio', 'active_hours', 'last_sub_correct',
    'attempts_per_step', 'first_day_ratio'
]
SELECTED_POLY_FEATURES = [
    'view_to_pass_ratio active_hours', 'days first_try_ratio', 'wrong viewed',
    'days wrong', 'wrong^2', 'steps_tried viewed'
]
FEATURE_COLUMNS = BASE_FEATURES + SELECTED_POLY_FEATURES

model = None


def load_model():
    global model
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found. Run train_model.py first. Expected: {model_path}"
        )
    model = joblib.load(model_path)


@app.on_event("startup")
def startup():
    load_model()


class PredictRequest(BaseModel):
    features: dict  # 19 признаков: 13 базовых + 6 полиномиальных


@app.post("/predict")
def predict(req: PredictRequest):
    """Предсказание по лучшей XGBoost модели (19 признаков)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        values = [float(req.features.get(k, 0)) for k in FEATURE_COLUMNS]
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    X = np.array([values]).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {
        "will_complete": bool(pred),
        "probability": round(proba, 4),
        "prediction": "Пройдёт курс" if pred else "Не пройдёт курс"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
