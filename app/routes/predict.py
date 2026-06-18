from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictRequest, PredictResponse
from app.models.predict import load_model, predict_one

router = APIRouter(prefix="/predict", tags=["EBV Prediction"])

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model("models/ebv_model.pkl")
    return _model


@router.post("/ebv", response_model=PredictResponse)
def predict_ebv(request: PredictRequest):
    """
    Prediksi EBV untuk satu domba.

    Dipanggil Laravel dengan data dari sheep + sheep_features.
    Return EBV yang akan disimpan kembali ke sheep_features.
    """
    try:
        model = get_model()

        result = predict_one(model, request.model_dump())

        return PredictResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
