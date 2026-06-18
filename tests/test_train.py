import os
import pandas as pd

from app.models.train import (
    prepare_features,
    split_data,
    train_model,
    evaluate_model,
    save_model,
)


def sample_df():
    return pd.DataFrame({
        "gender_enc": [0, 1, 0, 1, 0],
        "breed_id": [1, 1, 2, 2, 1],
        "weight_birth": [3.2, 3.5, 3.1, 3.6, 3.3],
        "weight_weaning": [18.0, 20.0, 17.0, 21.0, 19.0],
        "ADG_0_90": [0.20, 0.30, 0.25, 0.28, 0.22],
        "health_score": [0.90, 0.80, 1.00, 0.85, 0.95],

        "EBV_Bobot": [1.2, 1.8, 1.4, 1.9, 1.6],
        "EBV_ADG": [0.10, 0.15, 0.12, 0.16, 0.13],
        "EBV_Kesehatan": [0.95, 0.85, 1.00, 0.90, 0.98],
    })


def test_prepare_features():
    df = sample_df()

    X, y = prepare_features(df)

    assert list(X.columns) == [
        "gender_enc",
        "breed_id",
        "weight_birth",
        "weight_weaning",
        "ADG_0_90",
        "health_score",
    ]

    assert list(y.columns) == [
        "EBV_Bobot",
        "EBV_ADG",
        "EBV_Kesehatan",
    ]


def test_split_data():
    df = sample_df()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_train_model():
    df = sample_df()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    assert model is not None


def test_evaluate_model():
    df = sample_df()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.4,
    )

    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert "EBV_Bobot" in metrics
    assert "EBV_ADG" in metrics
    assert "EBV_Kesehatan" in metrics


def test_save_model():
    df = sample_df()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    output_path = "models/test_model.pkl"

    save_model(model, output_path)

    assert os.path.exists(output_path)