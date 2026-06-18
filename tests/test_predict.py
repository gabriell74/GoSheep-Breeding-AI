import pandas as pd

from app.models.predict import predict_one
from app.models.train import train_model, prepare_features


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


def test_predict_one():
    df = sample_df()

    X, y = prepare_features(df)

    model = train_model(X, y)

    features = {
        "gender_enc": 0,
        "breed_id": 1,
        "weight_birth": 3.2,
        "weight_weaning": 18.0,
        "ADG_0_90": 0.20,
        "health_score": 0.90,
    }

    result = predict_one(model, features)

    assert "EBV_Bobot" in result
    assert "EBV_ADG" in result
    assert "EBV_Kesehatan" in result

    assert isinstance(result["EBV_Bobot"], float)
    assert isinstance(result["EBV_ADG"], float)
    assert isinstance(result["EBV_Kesehatan"], float)