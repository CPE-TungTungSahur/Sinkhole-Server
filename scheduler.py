# ==============================================================================
# 📅 MRT SINKHOLE RISK SCHEDULER
# ==============================================================================

from apscheduler.schedulers.blocking import BlockingScheduler
from predictor_presto import SinkholePredictor

import json
import datetime
import os

# ------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------
POINTS_FILE = "./data/new/mrt_200m.geojson"
DAILY_DIR = "./storage/daily"
FEATURE_DIR = "./storage/features"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# INIT PREDICTOR
# ------------------------------------------------------------------------------
# ⚠ predictor จัดการ model เองแล้ว ไม่ต้องส่ง path
predictor = SinkholePredictor()
scheduler = BlockingScheduler()


# ------------------------------------------------------------------------------
# LOAD MRT POINTS
# ------------------------------------------------------------------------------
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
            "route_name": props.get("route_name"),
            "route_color": props.get("route_color"),
            "pt_index": props.get("pt_index"),
            "source_feature_index": props.get("source_feature_index"),
        })

    return points


def find_best_date(lat, lon):
    for days in [5, 8, 12, 16]:
        d = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        if predictor.has_data(lat, lon, d):
            return d
    return None

# ------------------------------------------------------------------------------
# MAIN SCAN JOB
# ------------------------------------------------------------------------------
def run_scan():
    # ⚠ Sentinel-2 delay ~10–15 days
    #safe_date = (
    #    datetime.datetime.now() - datetime.timedelta(days=15)
    #).strftime("%Y-%m-%d")
    safe_date = datetime.datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"[JOB] MRT risk scan @ {timestamp} (using date={safe_date})")

    points = load_points()
    results = []
    features_out = []

    for i, p in enumerate(points):
        print(f"  → ({i+1}/{len(points)}) {p['lat']}, {p['lon']}")

        result = predictor.predict(
            lat=p["lat"],
            lon=p["lon"],
            date=safe_date
        )

        # ไม่มี satellite data → ข้าม
        if "error" in result:
            continue

        results.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "risk": float(result["sinkhole_probability"]),
            "line": p["route_name"],
            "color": p["route_color"],
            "pt_index": p["pt_index"],
            "source_feature_index": p["source_feature_index"],
        })
        features_out.append({
            "lat": p["lat"],
            "lon": p["lon"],
            "date": safe_date,
            "line": p["route_name"],
            "color": p["route_color"],
            "features": result["user_features"],
        })

    if not results:
        print("⚠ No prediction results")
        return

    out_path = f"{DAILY_DIR}/{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"{FEATURE_DIR}/{timestamp}_features.json", "w", encoding="utf-8") as f:
        json.dump(features_out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(results)} points → {out_path}")


# ------------------------------------------------------------------------------
# SCHEDULE (Daily)
# ------------------------------------------------------------------------------
# 🔥 วันละหลายรอบ (รองรับ server timezone)
scheduler.add_job(run_scan, "cron", hour=16, minute=25, misfire_grace_time=120)

# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("📅 MRT Scheduler started")
    scheduler.start()
