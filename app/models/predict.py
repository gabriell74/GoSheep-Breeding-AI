"""
Prediction module — prediksi EBV untuk satu domba.
Dipanggil saat domba didaftarkan
atau saat cron update sheep_features.
"""

import pandas as pd
import numpy as np
import joblib
from app.models.train import FEATURE_COLS, TARGET_COLS

def load_model(model_path: str):
    """
    Load model dari file .pkl
    """
    return joblib.load(model_path)

def predict_one(model, features: dict) -> dict:
    """
    Prediksi EBV untuk satu domba.

    Args:
        model    : model yang sudah di-load dari .pkl
        features : dict fitur domba dari Laravel
                   harus mengandung semua FEATURE_COLS

    Returns:
        dict EBV hasil prediksi:
        {
            "EBV_Bobot"    : float,
            "EBV_ADG"      : float,
            "EBV_Kesehatan": float,
        }
    """
    X = pd.DataFrame([features])[FEATURE_COLS]

    y_pred = model.predict(X)

    return {
        "EBV_Bobot"    : round(float(y_pred[0][0]), 4),
        "EBV_ADG"      : round(float(y_pred[0][1]), 4),
        "EBV_Kesehatan": round(float(y_pred[0][2]), 4),
    }
