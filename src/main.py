import os
import json
import glob
from fastapi import FastAPI
from pydantic import BaseModel
from predictor import SinkholePredictor
from predictor import get_complete_satellite_data
from fastapi import HTTPException
import math
from typing import List
from datetime import datetime, timedelta

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
# รับพิกัด แล้วสแกนพื้นที่ เรียกใช้โดยไฟล์ scheduler.py
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
# เลือกไฟล์ล่าสุด แล้วโหลด JSON
@app.get("/latest-map")
def latest_map():
    files = sorted(glob.glob(f"{DAILY_DIR}/*.json"), key=os.path.getmtime)
    if not files:
        return {"error": "No cached scan found yet."}

    latest = files[-1]

    with open(latest, "r") as f:
        data = json.load(f)

    return {
        "date": os.path.basename(latest),
        "data": data
    }

# -------------------- Get latest scan as GeoJSON --------------------
# เลือกไฟล์ล่าสุด แล้วแปลงเป็น GeoJSON
def scan_to_geojson(points):
    """
    points: list[dict] มี key: lat, lon, risk
    """
    features = []
    for p in points:
        try:
            lat = float(p["lat"])
            lon = float(p["lon"])
            risk = float(p["risk"])
        except (KeyError, ValueError, TypeError):
            continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # ⚠ GeoJSON = [lon, lat]
            },
            "properties": {
                "risk": risk,
                "line": p.get("line"),
                "color": p.get("color"),
                "point_type": p.get("point_type"),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }

@app.get("/latest-geojson")
def latest_geojson():
    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.json")))
    if not files:
        raise HTTPException(status_code=404, detail="No cached scan found.")

    latest = files[-1]

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ถ้าไฟล์เป็น {"date": "...", "data": [...]}
    if isinstance(data, dict) and "data" in data:
        points = data["data"]
    else:
        points = data

    geojson = scan_to_geojson(points)

    return {
        "file": os.path.basename(latest),
        "geojson": geojson,
    }


# -------------------- Get point risk history --------------------
# ดึงประวัติความเสี่ยงของจุดละติจูด-ลองจิจูด โดยดูจากไฟล์สแกนที่เก็บไว้ (คลิกที่จุด -> ส่ง lat, lon มา -> ดึงความเสี่ยงย้อนหลัง)
@app.get("/point-history")
def point_history(lat: float, lon: float):
    files = glob.glob(f"{DAILY_DIR}/*.json")

    # วันปัจจุบัน - 30 วัน
    cutoff = datetime.now() - timedelta(days=30)

    valid_files = []
    for f in files:
        fname = os.path.basename(f)
        dt_str = fname.split(".")[0]      # "2025-12-04_13-03"
        date_part = dt_str.split("_")[0]  # "2025-12-04"
        file_date = datetime.strptime(date_part, "%Y-%m-%d")

        if file_date >= cutoff:
            valid_files.append(f)

    # sort ตามวันที่ เก่า → ใหม่
    valid_files = sorted(valid_files)

    history = []
    for f in valid_files:
        with open(f, "r") as fp:
            raw = json.load(fp)

        # handle nested JSON wrapper
        if isinstance(raw, dict) and "data" in raw:
            data = raw["data"]
        else:
            data = raw

        # หา closest point
        closest = min(
            data,
            key=lambda p: (p["lat"] - lat)**2 + (p["lon"] - lon)**2
        )

        history.append({
            "date": os.path.basename(f).split(".")[0],
            "risk": closest["risk"]
        })

    return {"point": {"lat": lat, "lon": lon}, "history": history}

# -------------------- Get point satellite features --------------------
# ดึงข้อมูลดาวเทียมย้อนหลัง สำหรับจุดละติจูด-ลองจิจูด ที่รับมา
@app.get("/point-features")
def point_features(
    lat: float,
    lon: float,
    end_date: str,
    months: int = 12,
):
    df = get_complete_satellite_data(lat, lon, end_date, seq_len=months)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No satellite data available.")

    df_out = df.reset_index()
    df_out.rename(columns={"ts": "timestamp"}, inplace=True)
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%d")

    # ---- choose features here ----
    ndvi_cols = [c for c in df_out.columns if "NDVI" in c]
    ndwi_cols = [c for c in df_out.columns if "NDWI" in c]
    lst_cols = [c for c in df_out.columns if "LST" in c]
    radar_cols = [c for c in df_out.columns if c in ["VV", "VH", "VV_VH_ratio", "VV_VH_diff"]]

    selected = ["timestamp"] + ndvi_cols + ndwi_cols + lst_cols + radar_cols

    return {
        "point": {"lat": lat, "lon": lon},
        "features": df_out[selected].to_dict(orient="records"),
        "columns": selected,
    }
