from fastapi import FastAPI

app = FastAPI(title="GoSheep Breeding AI")

@app.get("/")
def root():
  return {"message": "FastAPI Smart Breeding AI"}
