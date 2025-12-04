import os
import json
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from predictor import SinkholePredictor

app = FastAPI()

MODEL_PATH = "./models/final_sinkhole_model.keras"
SCALER_PATH = "./models/pretrained_encoder_complete_scaler.pkl"

# โหลดโมเดลตอนบู๊ต
predictor = SinkholePredictor(MODEL_PATH, SCALER_PATH)

# โฟลเดอร์เก็บสแกน
DAILY_DIR = "./storage/daily"
os.makedirs(DAILY_DIR, exist_ok=True)

# -------------------- Request Schema --------------------
class ScanRequest(BaseModel):
    lat: float
    lon: float
    date: str
    radius_km: float = 0.5
    step_km: float = 0.05

# -------------------- Default route --------------------
@app.get("/")
def home():
    return {"status": "sinkhole AI server running"}

# -------------------- Main scanning endpoint --------------------
@app.post("/scan-area")
def scan_area(req: ScanRequest):
    results = predictor.scan_grid(
        center_lat=req.lat,
        center_lon=req.lon,
        date=req.date,
        radius_km=req.radius_km,
        step_km=req.step_km
    )
    return results.to_dict(orient="records")

# -------------------- Get latest cached scan --------------------
@app.get("/latest-map")
def latest_map():
    files = sorted(glob.glob(f"{DAILY_DIR}/*.json"))

    if not files:
        return {"error": "No cached scan found yet."}

    latest = files[-1]

    with open(latest, "r") as f:
        data = json.load(f)

    return {
        "date": os.path.basename(latest),
        "data": data
    }
