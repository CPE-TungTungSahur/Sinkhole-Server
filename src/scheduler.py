from apscheduler.schedulers.blocking import BlockingScheduler
from .predictor import SinkholePredictor
import json
import datetime
import os
from pathlib import Path

# Get the src directory path
SRC_DIR = Path(__file__).parent
MODEL_PATH = str(SRC_DIR / "models" / "final_sinkhole_model.keras")
SCALER_PATH = str(SRC_DIR / "models" / "pretrained_encoder_complete_scaler.pkl")
DAILY_DIR = str(SRC_DIR / "storage" / "daily")

# พื้นที่กรุงเทพ
CENTER_LAT = 13.7563
CENTER_LON = 100.5018
RADIUS_KM = 0.5
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
