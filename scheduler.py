from apscheduler.schedulers.blocking import BlockingScheduler
from predictor import SinkholePredictor
import json
import datetime
import os

MODEL_PATH = "./models/final_sinkhole_model.keras"
SCALER_PATH = "./models/pretrained_encoder_complete_scaler.pkl"

POINTS_FILE = "./data/mrt_resampled_1500m.geojson"
DAILY_DIR = "./storage/daily"

os.makedirs(DAILY_DIR, exist_ok=True)

predictor = SinkholePredictor(MODEL_PATH, SCALER_PATH)
scheduler = BlockingScheduler()


def load_points():
    """โหลดจุด MRT ที่ preprocess แล้ว"""
    with open(POINTS_FILE, "r", encoding="utf-8") as f:
        geo = json.load(f)

    points = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] != "Point":
            continue
        lon, lat = feat["geometry"]["coordinates"]
        props = feat.get("properties", {})
        points.append({
            "lat": lat,
            "lon": lon,
            "line": props.get("line"),
            "color": props.get("color"),
            "point_type": props.get("type"),
        })

    return points


def run_scan():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"[JOB] MRT risk scan @ {timestamp}")

    points = load_points()
    results = []

    for i, p in enumerate(points):
        print(f"  → ({i+1}/{len(points)}) {p['lat']}, {p['lon']}")

        prob = predictor.predict(
            lat=p["lat"],
            lon=p["lon"],
            date=date_str
        )

        if prob is None:
            continue

        results.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "risk": float(prob),
            "line": p["line"],
            "color": p["color"],
            "point_type": p["point_type"],
        })

    if not results:
        print("⚠ No prediction results")
        return

    out_path = f"{DAILY_DIR}/{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"✅ Saved {len(results)} points → {out_path}")


# 🔥 วันละ 3 รอบ
scheduler.add_job(run_scan, "cron", hour=8, minute=0)  #8:00 น.
scheduler.add_job(run_scan, "cron", hour=14, minute=0) #14:00 น.
scheduler.add_job(run_scan, "cron", hour=19, minute=0) #19:00 น.

if __name__ == "__main__":
    print("📅 MRT Scheduler started")
    scheduler.start()
