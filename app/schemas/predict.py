"""Request and Response scheme for prediction endpoint"""

from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    gender_enc    : int   = Field(..., ge=0, le=1,
                            description="1=jantan, 0=betina")
    breed_id      : int   = Field(..., ge=1, le=4,
                            description="1=Garut, 2=DEG, 3=DET, 4=Priangan")
    weight_birth  : float = Field(..., gt=0,
                            description="Bobot lahir (kg)")
    weight_weaning: float = Field(..., gt=0,
                            description="Bobot sapih ~90hr (kg)")
    ADG_0_90      : float = Field(..., gt=0,
                            description="Average Daily Gain lahir→sapih (kg/hari)")
    health_score  : float = Field(..., ge=0.0, le=1.0,
                            description="Skor kesehatan 0.0–1.0")

class PredictResponse(BaseModel):
    EBV_Bobot    : float
    EBV_ADG      : float
    EBV_Kesehatan: float
