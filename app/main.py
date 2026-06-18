from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI(title="GoSheep Breeding AI")

app.include_router(predict_router)

@app.get("/")
def root():
  return {"message": "FastAPI Smart Breeding AI"}
