from apscheduler.schedulers.blocking import BlockingScheduler
from predictor import SinkholePredictor
import json
import datetime
import os

MODEL_PATH = "./models/final_sinkhole_model.keras"
SCALER_PATH = "./models/pretrained_encoder_complete_scaler.pkl"
DAILY_DIR = "./storage/daily"

# พื้นที่บางมด
CENTER_LAT = 13.651736
CENTER_LON = 100.492564
RADIUS_KM = 0.6
STEP_KM = 0.05

predictor = SinkholePredictor(MODEL_PATH, SCALER_PATH)

scheduler = BlockingScheduler()

def run_scan():
    now_with_hour = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"[JOB] Running scheduled scan: {now}")

    df = predictor.scan_grid(
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        date=now,
        radius_km=RADIUS_KM,
        step_km=STEP_KM
    )

    if df is None or df.empty:
        print("⚠ Scan failed, no data.")
        return

    os.makedirs(DAILY_DIR, exist_ok=True)

    file_path = f"{DAILY_DIR}/{now_with_hour}.json"
    df.to_json(file_path, orient="records")
    print(f"✅ Saved: {file_path}")


# 🔥 ตั้งเวลา 3 รอบต่อวัน
scheduler.add_job(run_scan, "cron", hour=8, minute=0)    # 8:00
scheduler.add_job(run_scan, "cron", hour=14, minute=0)   # 14:00
scheduler.add_job(run_scan, "cron", hour=19, minute=0)   # 20:00

# เริ่มงาน
if __name__ == "__main__":
    print("📅 Scheduler started...")
    scheduler.start()
