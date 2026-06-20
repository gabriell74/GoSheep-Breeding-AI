from fastapi import FastAPI
from app.routes.predict import router as predict_router
from app.routes.wright  import router as wright_router

app = FastAPI(title="GoSheep Breeding AI")

app.include_router(predict_router)
app.include_router(wright_router)

@app.get("/")
def root():
    return {"message": "FastAPI Smart Breeding AI"}
