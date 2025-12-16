# ==============================================================================
# 🛰️ SINKHOLE PREDICTOR (NO RETRAIN)
# ==============================================================================

import ee
import joblib
import numpy as np
import pandas as pd

# ✅ import class จากไฟล์ train
from presto_model_train import GEESinkholeLoader, PrestoFeatureExtractor

# ------------------------------------------------------------------------------
# 1. GEE INIT
# ------------------------------------------------------------------------------
try:
    ee.Initialize(project='useful-tempest-477706-j9')
except Exception:
    ee.Authenticate()
    ee.Initialize()

MODEL_PATH = "models/default_model.pt"
SVM_PATH = "models/sinkhole_svm_pipeline.pkl"


def array_to_timeseries(data, band_names):
    """
    data: dict from GEESinkholeLoader.get_data
    """
    arr = data["array"]
    months = data["months"]

    df = pd.DataFrame(arr, columns=band_names)
    df["month"] = months + 1   # 0–11 → 1–12
    return df

def filter_user_features(df):
    df = df.copy()

    # NDVI
    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 1e-8)

    # เลือกเฉพาะที่จะแสดง
    keep_cols = [
        "month",
        "NDVI",
        "VV",
        "VH",
        "elevation",
        "slope",
    ]

    return df[keep_cols]


# ------------------------------------------------------------------------------
# 2. Predictor Class
# ------------------------------------------------------------------------------
class SinkholePredictor:
    def __init__(self, svm_path=SVM_PATH):
        self.loader = GEESinkholeLoader()
        self.encoder = PrestoFeatureExtractor()
        self.clf = joblib.load(svm_path)

    def predict(self, lat, lon, date):
        data = self.loader.get_data(lat, lon, date)
        if data is None:
            return {"error": "No satellite data available"}

        # -------- RAW FEATURES --------
        df_ts = array_to_timeseries(data, self.loader.band_names)
        df_user = filter_user_features(df_ts)

        # -------- PRESTO FEATURES --------
        feats = self.encoder.get_features([data])


        # --- normalize shape ---
        if feats.ndim == 3:
            # (B, T, D) → (B, D)
            feats_2d = feats.mean(axis=1)
        elif feats.ndim == 2:
            # (B, D) → OK
            feats_2d = feats
        else:
            raise ValueError(f"Unexpected feature shape: {feats.shape}")

        # ensure 2D for sklearn
        feats_2d = np.atleast_2d(feats_2d)

        # -------- PREDICTION --------
        prob = float(self.clf.predict_proba(feats_2d)[0, 1])

        return {
            "point": {"lat": lat, "lon": lon},
            "sinkhole_probability": prob,
            "user_features": df_user.to_dict(orient="records"),
            "user_feature_columns": list(df_user.columns),
        }

# ------------------------------------------------------------------------------
# 3. CLI TEST
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = SinkholePredictor()

    result = predictor.predict(
        lat=13.780522,
        lon=100.509275,
        date="2025-12-16"
    )

    print("\n🧪 Prediction Result")
    for k, v in result.items():
        print(f"{k}: {v}")
