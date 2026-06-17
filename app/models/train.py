"""
Training module untuk Random Forest EBV prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

FEATURE_COLS = [
    "gender_enc",
    "breed_id",
    "weight_birth",
    "weight_weaning",
    "ADG_0_90",
    "health_score",
]

TARGET_COLS = [
    "EBV_Bobot",
    "EBV_ADG",
    "EBV_Kesehatan",
]

def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load training_data.csv hasil run_pipeline.py
    """
    df = pd.read_csv(filepath)
    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pisah DataFrame menjadi fitur (X) dan target (y).

    X → fitur yang dipakai model untuk belajar
    y → EBV yang diprediksi model

    Kolom seperti weight_180d, weight_365d, ADG_90_180
    tidak dipakai karena terlalu banyak NULL.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    return X, y

def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Pisah data menjadi training set dan testing set.

    Args:
        X            : fitur
        y            : target EBV
        test_size    : proporsi data test (default 20%)
        random_state : seed untuk reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
) -> MultiOutputRegressor:
    """
    Train Random Forest untuk prediksi EBV.

    Args:
        X_train      : fitur training
        y_train      : target EBV training
        n_estimators : jumlah pohon (default 100)
        random_state : seed untuk reproducibility

    Returns:
        model yang sudah ditraining
    """
    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    return model

def evaluate_model(
    model: MultiOutputRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict:
    """
    Evaluasi model menggunakan data test.

    Metrik:
        MAE → rata-rata error prediksi
        R²  → seberapa baik model fit ke data
    """
    y_pred = model.predict(X_test)

    results = {}
    for i, target in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred[:, i])
        results[target] = {"MAE": round(mae, 4), "R2": round(r2, 4)}

    return results


def save_model(model: MultiOutputRegressor, output_path: str) -> None:
    """
    Simpan model ke file .pkl menggunakan joblib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)

def run_training(
    training_path : str,
    model_path    : str,
) -> dict:
    """
    Jalankan full training pipeline.

    Args:
        training_path : path ke training_data.csv
        model_path    : path output model.pkl

    Returns:
        dict metrics hasil evaluasi
    """
    df   = load_training_data(training_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model   = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    save_model(model, model_path)

    return metrics
