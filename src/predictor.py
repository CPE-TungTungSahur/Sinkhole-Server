import ee
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
# import matplotlib.pyplot as plt   # ❌ REMOVED
import os
from tqdm.auto import tqdm

# ==============================================================================
# 1. SETUP & AUTHENTICATION
# ==============================================================================
# Defer Earth Engine initialization to avoid blocking server startup
_ee_initialized = False

def initialize_earth_engine():
    """Initialize Earth Engine with proper error handling."""
    global _ee_initialized
    if _ee_initialized:
        return True
    
    try:
        # Try to initialize with the project ID
        ee.Initialize(project='upheld-shield-330108')
        _ee_initialized = True
        print("✅ Earth Engine Initialized with project.")
        return True
    except Exception as e:
        print(f"⚠️ Earth Engine initialization failed: {e}")
        print("⚠️ Continuing without Earth Engine. Some features may be unavailable.")
        return False

# ==============================================================================
# 2. CUSTOM OBJECTS (จำเป็นต้องแปะไว้เพื่อให้ Keras รู้จักตอนโหลดโมเดล)
# ==============================================================================
class SimCLRLoss(keras.losses.Loss):
    def __init__(self, temperature=0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    def call(self, z_i, z_j):
        return tf.reduce_mean(tf.zeros((1,))) # Dummy call for inference loading

# ==============================================================================
# 3. DATA EXTRACTION FUNCTIONS (ต้องเหมือนตอนเทรนเท่านั้น)
# ==============================================================================

def get_complete_satellite_data(lat, lon, end_date, seq_len=12, buffer=10):
    try:
        # Initialize Earth Engine if not already done
        if not initialize_earth_engine():
            raise Exception("Earth Engine initialization failed. Please authenticate first.")
        
        evt_dt = pd.to_datetime(end_date)
        start_date = (evt_dt - pd.DateOffset(months=seq_len)).strftime('%Y-%m-%d')
        end_date_str = evt_dt.strftime('%Y-%m-%d')
        point = ee.Geometry.Point([lon, lat])

        roi_small = point.buffer(buffer)
        roi_medium = point.buffer(max(buffer, 30))
        roi_large = point.buffer(max(buffer, 250))

        all_data = []

        # SENTINEL-2
        def maskS2(img):
            qa = img.select('QA60')
            return img.updateMask(qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0)))

        def addS2Indices(img):
            ndvi = img.normalizedDifference(['B8','B4']).rename('S2_NDVI')
            ndwi = img.normalizedDifference(['B3','B8']).rename('S2_NDWI')
            evi = img.expression('2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
                {'NIR':img.select('B8'),'RED':img.select('B4'),'BLUE':img.select('B2')}).rename('S2_EVI')
            ndbi = img.normalizedDifference(['B11','B8']).rename('S2_NDBI')
            bsi = img.expression('((SWIR+RED)-(NIR+BLUE))/((SWIR+RED)+(NIR+BLUE))',
                {'SWIR':img.select('B11'),'RED':img.select('B4'),'NIR':img.select('B8'),'BLUE':img.select('B2')}).rename('S2_BSI')
            return img.addBands([ndvi, ndwi, evi, ndbi, bsi])

        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(roi_small).filterDate(start_date, end_date_str)
              .map(maskS2).map(addS2Indices))

        if s2.size().getInfo() > 0:
            s2_bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12',
                       'S2_NDVI','S2_NDWI','S2_EVI','S2_NDBI','S2_BSI']
            def extract_s2(img):
                s = img.select(s2_bands).reduceRegion(
                    reducer=ee.Reducer.median().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=roi_small, scale=20, maxPixels=1e9)
                return ee.Feature(None, s.set('ts', img.date().millis()))
            s2_data = s2.map(extract_s2).aggregate_array('.all').getInfo()
            all_data.append(('S2', s2_data))

        # SENTINEL-1
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
              .filter(ee.Filter.eq('instrumentMode','IW'))
              .filterBounds(roi_small).filterDate(start_date, end_date_str))

        if s1.size().getInfo() > 0:
            def extract_s1(img):
                s = img.select(['VV','VH']).reduceRegion(
                    reducer=ee.Reducer.median(), geometry=roi_small, scale=10, maxPixels=1e9)
                ratio = img.expression('VV/(VH+1e-8)',{'VV':img.select('VV'),'VH':img.select('VH')}).rename('VV_VH_ratio')
                diff = img.expression('VV-VH',{'VV':img.select('VV'),'VH':img.select('VH')}).rename('VV_VH_diff')
                r = ratio.addBands(diff).reduceRegion(reducer=ee.Reducer.median(), geometry=roi_small, scale=10)
                return ee.Feature(None, s.combine(r).set('ts', img.date().millis()))
            s1_data = s1.map(extract_s1).aggregate_array('.all').getInfo()
            all_data.append(('S1', s1_data))

        # LANDSAT
        def maskL8(img):
            qa = img.select('QA_PIXEL')
            return img.updateMask(qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0)))

        def addL8Indices(img):
            ndvi = img.normalizedDifference(['SR_B5','SR_B4']).rename('L8_NDVI')
            ndwi = img.normalizedDifference(['SR_B3','SR_B5']).rename('L8_NDWI')
            lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('L8_LST')
            return img.addBands([ndvi, ndwi, lst])

        l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterBounds(roi_medium).filterDate(start_date, end_date_str)
              .map(maskL8).map(addL8Indices))
        l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
              .filterBounds(roi_medium).filterDate(start_date, end_date_str)
              .map(maskL8).map(addL8Indices))
        landsat = l8.merge(l9)

        if landsat.size().getInfo() > 0:
            l_bands = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                      'L8_NDVI','L8_NDWI','L8_LST']
            def extract_landsat(img):
                s = img.select(l_bands).reduceRegion(
                    reducer=ee.Reducer.median(), geometry=roi_medium, scale=30, maxPixels=1e9)
                return ee.Feature(None, s.set('ts', img.date().millis()))
            landsat_data = landsat.map(extract_landsat).aggregate_array('.all').getInfo()
            all_data.append(('Landsat', landsat_data))

        # MODIS
        modis_sr = (ee.ImageCollection('MODIS/061/MOD09A1')
                    .filterBounds(roi_large).filterDate(start_date, end_date_str))
        modis_lst = (ee.ImageCollection('MODIS/061/MOD11A2')
                     .filterBounds(roi_large).filterDate(start_date, end_date_str))

        if modis_sr.size().getInfo() > 0:
            def extract_modis_sr(img):
                ndvi = img.normalizedDifference(['sur_refl_b02','sur_refl_b01']).rename('MODIS_NDVI')
                evi = img.expression('2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
                    {'NIR':img.select('sur_refl_b02'),'RED':img.select('sur_refl_b01'),
                     'BLUE':img.select('sur_refl_b03')}).rename('MODIS_EVI')
                s = ndvi.addBands(evi).reduceRegion(
                    reducer=ee.Reducer.median(), geometry=roi_large, scale=500, maxPixels=1e9)
                return ee.Feature(None, s.set('ts', img.date().millis()))
            modis_sr_data = modis_sr.map(extract_modis_sr).aggregate_array('.all').getInfo()
            all_data.append(('MODIS_SR', modis_sr_data))

        if modis_lst.size().getInfo() > 0:
            def extract_modis_lst(img):
                day = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('MODIS_LST_Day')
                night = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('MODIS_LST_Night')
                s = day.addBands(night).reduceRegion(
                    reducer=ee.Reducer.median(), geometry=roi_large, scale=1000, maxPixels=1e9)
                return ee.Feature(None, s.set('ts', img.date().millis()))

            modis_lst_data = modis_lst.map(extract_modis_lst).aggregate_array('.all').getInfo()
            all_data.append(('MODIS_LST', modis_lst_data))

        # DEM
        dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        hillshade = ee.Terrain.hillshade(dem)
        dem_stats = dem.addBands([slope, aspect, hillshade]).reduceRegion(
            reducer=ee.Reducer.median().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=roi_medium, scale=30, maxPixels=1e9).getInfo()

        # MERGE & PROCESS
        if not all_data: return None

        dfs = []
        for name, data_list in all_data:
            if data_list:
                try:
                    records = []
                    for record in data_list:
                        if isinstance(record, dict) and 'properties' in record:
                            records.append(record['properties'])
                        elif isinstance(record, dict):
                            records.append(record)
                    if not records: continue
                    df = pd.DataFrame(records)
                    if 'ts' not in df.columns: continue
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms', errors='coerce')
                    df = df.dropna(subset=['ts']).set_index('ts')
                    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(axis=1, how='all')
                    if not df.empty: dfs.append(df)
                except:
                    continue

        if not dfs: return None

        df_merged = pd.DataFrame()
        for df in dfs:
            df_merged = df_merged.join(df, how='outer') if not df_merged.empty else df

        if df_merged.empty: return None

        # Add DEM
        for k, v in dem_stats.items():
            if v is not None:
                df_merged[f'DEM_{k}'] = v

        # Resample & Pad
        df_final = df_merged.resample('ME').median().interpolate(method='time').ffill().bfill()

        if len(df_final) < max(6, seq_len // 2):
            return None

        if len(df_final) < seq_len:
            last_row = df_final.iloc[-1:]
            while len(df_final) < seq_len:
                df_final = pd.concat([df_final, last_row])

        return df_final.iloc[-seq_len:]

    except Exception as e:
        print(f"Extraction Error: {e}")
        return None

# ==============================================================================
# ต้องมี focal_loss ให้ load_model ใช้ด้วย
# ==============================================================================
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * ce)

# ==============================================================================
# 4. PREDICTOR CLASS
# ==============================================================================
class SinkholePredictor:
    def __init__(self, model_path, scaler_path):
        print(f"Loading Model: {model_path}")

        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                'SimCLRLoss': SimCLRLoss,
                'focal_loss': focal_loss
            },
            compile=False
        )

        print(f"Loading Scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("✅ System Ready.")

    def predict(self, lat, lon, date, seq_len=12):
        print(f"fetching data for {lat}, {lon}...")

        ts_data = get_complete_satellite_data(lat, lon, date, seq_len=seq_len)
        if ts_data is None:
            return None

        X = ts_data.values
        X_input = X.reshape(1, seq_len, -1)

        n_features = X_input.shape[2]
        try:
            X_scaled = self.scaler.transform(X_input.reshape(-1, n_features)).reshape(X_input.shape)
        except:
            print("⚠️ Feature mismatch! Training feature set must match extraction function.")
            return None

        prob = self.model.predict(X_scaled, verbose=0)[0][0]
        return prob

    def scan_grid(self, center_lat, center_lon, date, radius_km=1.0, step_km=0.1):
        print(f"Scanning {radius_km}km radius around {center_lat}, {center_lon}")

        d_lat = radius_km / 111.0
        d_lon = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        lats = np.arange(center_lat - d_lat, center_lat + d_lat, step_km / 111.0)
        lons = np.arange(center_lon - d_lon, center_lon + d_lon, step_km / (111.0 * np.cos(np.radians(center_lat))))

        results = []
        for lat in tqdm(lats, desc="Scanning"):
            for lon in lons:
                prob = self.predict(lat, lon, date)
                if prob is not None:
                    results.append({'lat': lat, 'lon': lon, 'risk': prob})

        return pd.DataFrame(results)

# ==============================================================================
# 5. MAIN EXECUTION (NO MATPLOTLIB)
# ==============================================================================
if __name__ == "__main__":

    MODEL_FILE = '/content/final_sinkhole_model (1).keras'
    SCALER_FILE = '/content/pretrained_encoder_complete_scaler (4).pkl'

    CENTER_LAT = 13.780522
    CENTER_LON = 100.509275
    DATE_NOW = '2025-11-27'

    RADIUS_KM = 0.5
    STEP_KM = 0.05

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):

        predictor = SinkholePredictor(MODEL_FILE, SCALER_FILE)

        df_results = predictor.scan_grid(
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
            date=DATE_NOW,
            radius_km=RADIUS_KM,
            step_km=STEP_KM
        )

        if not df_results.empty:
            print("\n✅ Scan Complete!")
            print(df_results.head())

            # Removed block: Plot & Heatmap with matplotlib
            # (plt.figure ...)
            # (plt.scatter ...)
            # (plt.show ...)

            high_risk = df_results[df_results['risk'] > 0.8]
            if not high_risk.empty:
                print("\n⚠️ HIGH RISK ZONES FOUND:")
                print(high_risk)

        else:
            print("❌ No data retrieved (Cloud cover or GEE connection issue).")

    else:
        print("❌ Model or Scaler file not found!")
