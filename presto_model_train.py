# ==============================================================================
# 🚀 PRESTO SINKHOLE: 5-FOLD CROSS VALIDATION (NO SEED LOCK)
# ==============================================================================
import os
import sys
import subprocess
import requests
import warnings
import torch
import numpy as np
import pandas as pd
import random
import time
import joblib 
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. SETUP & INSTALL ---
print("⚙️ INSTALLING DEPENDENCIES...")
if not os.path.exists('presto_source'):
    subprocess.run(['git', 'clone', 'https://github.com/nasaharvest/presto.git', 'presto_source'], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "einops", "xarray", "tqdm", "earthengine-api", "seaborn", "matplotlib", "scikit-learn", "joblib"], check=True)

# --- 2. DOWNLOAD PRESTO MODEL ---
MODEL_URL = "https://github.com/nasaharvest/presto/raw/main/data/default_model.pt"
MODEL_PATH = "data/default_model.pt"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print(f"⬇️ Downloading Presto Model...")
    os.makedirs("data", exist_ok=True)
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

sys.path.insert(0, os.path.abspath('presto_source'))

# Monkey Patch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool

# --- 3. IMPORTS ---
import ee
from single_file_presto import Presto
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 4. GEE AUTHENTICATION ---
try:
    ee.Initialize(project='useful-tempest-477706-j9')
    print("✅ Earth Engine Connected.")
except:
    ee.Authenticate()
    ee.Initialize(project='useful-tempest-477706-j9')
# ==============================================================================
# 5. DATA AUGMENTATION (Random Mode - No Lock)
# ==============================================================================
def get_augmented_data():
    # หมายเหตุ: ปล่อยให้สุ่มตามธรรมชาติ ไม่มีการล็อค Seed
    raw_positives = [
        [13.780522, 100.509275, '2025-09-22'], [37.568800, 126.936300, '2024-08-29'],
        [3.152500, 101.696500, '2024-08-23'], [38.917930, -90.153620, '2024-06-26'],
        [41.861600, 12.551900, '2024-03-28'], [-33.955500, 151.145000, '2024-03-01'],
        [7.365000, 126.032000, '2024-02-06'], [28.771650, -82.695657, '2024-01-01'],
        [-9.638000, -35.752000, '2023-12-10'], [27.963000, -81.893000, '2023-09-25'],
        [27.983000, -82.265300, '2023-07-11'], [14.529000, -90.585000, '2022-09-24'],
        [-27.465300, -70.275400, '2022-07-30'], [24.928000, 106.563000, '2022-05-06'],
        [19.086000, 72.908900, '2021-06-13'], [37.713000, 33.553000, '2021-06-01'],
        [19.125700, -98.373800, '2021-05-29'], [40.863000, 14.227000, '2021-01-08']
    ]

    final_points = []
    final_labels = []

    print(f"🧬 Augmenting Data (Random)...")
    for lat, lon, date in raw_positives:
        # Original
        final_points.append([lat, lon, date]); final_labels.append(1)
        # Jitter
        offsets = [(0.0001, 0.0001), (-0.0001, -0.0001), (0.0001, -0.0001), (-0.0001, 0.0001)]
        for off_lat, off_lon in offsets:
            final_points.append([lat + off_lat, lon + off_lon, date]); final_labels.append(1)
        # Hard Negative
        angle = random.uniform(0, 6.28); dist = 0.005 
        final_points.append([lat + dist * np.cos(angle), lon + dist * np.sin(angle), date]); final_labels.append(0)
    
    # Global Negatives
    num_global_neg = len(raw_positives) * 4
    for _ in range(num_global_neg):
        final_points.append([random.uniform(-60, 60), random.uniform(-180, 180), '2023-01-01']); final_labels.append(0)

    print(f"   ✅ Data Points: {len(final_points)} (Pos: {final_labels.count(1)} / Neg: {final_labels.count(0)})")
    return final_points, final_labels

# ==============================================================================
# 6. FEATURE EXTRACTION
# ==============================================================================
class GEESinkholeLoader:
    def __init__(self, seq_len=12):
        self.seq_len = seq_len
        self.band_names = ['B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12', 'VV', 'VH', 'elevation', 'slope']
    def get_data(self, lat, lon, end_date):
        try:
            point = ee.Geometry.Point([lon, lat]); roi = point.buffer(20); end_dt = pd.to_datetime(end_date); start_dt = end_dt - pd.DateOffset(months=self.seq_len)
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start_dt, end_dt).map(lambda img: img.updateMask(img.select('QA60').lt(1))).select(['B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12'])
            s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(start_dt, end_dt).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW')).select(['VV', 'VH'])
            srtm = ee.Image('NASA/NASADEM_HGT/001'); topo = srtm.select(['elevation']).addBands(ee.Terrain.slope(srtm.select('elevation')))
            def extract(img):
                date = img.date(); s1_img = s1.filterDate(date.advance(-15, 'day'), date.advance(15, 'day')).mosaic(); full = img.divide(10000.0).addBands(s1_img).addBands(topo)
                stat = full.reduceRegion(reducer=ee.Reducer.median(), geometry=roi, scale=10, maxPixels=1e9); return ee.Feature(None, stat.set('system:time_start', date.millis()))
            feats = s2.map(extract).filter(ee.Filter.notNull(self.band_names)).getInfo()['features']
            if not feats: return None
            records = [{**f['properties'], 'ts': f['properties']['system:time_start']} for f in feats]; df = pd.DataFrame(records); df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.sort_values('ts').set_index('ts').resample('M').median().interpolate().ffill().bfill()
            if len(df) > self.seq_len: df = df.iloc[-self.seq_len:]
            elif len(df) < self.seq_len: pad = pd.DataFrame(0, index=range(self.seq_len - len(df)), columns=df.columns); df = pd.concat([pad, df])
            return {'array': df[self.band_names].values, 'lat': lat, 'lon': lon, 'months': df.index.month.values - 1}
        except: return None

class PrestoFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Presto.construct()
        self.model.load_state_dict(torch.load("models/default_model.pt", map_location=self.device)); self.model.eval(); self.model.to(self.device)
    def get_features(self, data_list):
        if not data_list: return np.array([])
        batch_x, batch_mask, batch_dw, batch_latlons, batch_months = [], [], [], [], []
        for d in data_list:
            arr = d['array']; x = np.zeros((12, 17)); mask = np.ones((12, 17))
            x[:, 0:2] = arr[:, 7:9]; mask[:, 0:2] = 0; x[:, 2:5] = arr[:, 0:3]; mask[:, 2:5] = 0; x[:, 8] = arr[:, 3]; mask[:, 8] = 0
            x[:, 9] = arr[:, 4]; mask[:, 9] = 0; x[:, 10:12] = arr[:, 5:7]; mask[:, 10:12] = 0; x[:, 14:16] = arr[:, 9:11]; mask[:, 14:16] = 0
            b8, b4 = arr[:, 3], arr[:, 2]; ndvi = (b8 - b4) / (b8 + b4 +  1e-8); x[:, 16] = ndvi; mask[:, 16] = 0
            batch_x.append(x); batch_mask.append(mask); batch_dw.append(np.zeros(12)); batch_latlons.append([d['lat'], d['lon']]); batch_months.append(d['months'])
        to_t = lambda a, t: torch.tensor(np.array(a), dtype=t).to(self.device)
        with torch.no_grad():
            enc = self.model.encoder(x=to_t(batch_x, torch.float32), dynamic_world=to_t(batch_dw, torch.long), mask=to_t(batch_mask, torch.float32), latlons=to_t(batch_latlons, torch.float32), month=to_t(batch_months, torch.long))
            return enc.cpu().numpy()

# ==============================================================================
# 7. MAIN (5-FOLD CV)
# ==============================================================================
def main():
    # 1. Prepare Data  
    points, labels = get_augmented_data()
    loader = GEESinkholeLoader()
    presto = PrestoFeatureExtractor()
    
    X_features, y_labels = [], []
    print("\n🚀 Processing Data Batches...")
    
    BATCH_SIZE = 10
    for i in range(0, len(points), BATCH_SIZE):
        batch_pts = points[i:i  +BATCH_SIZE]; batch_lbls = labels[i:i+BATCH_SIZE]
        batch_data_ee = []; labels_temp = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(loader.get_data, p[0], p[1], p[2]): l for p, l in zip(batch_pts, batch_lbls)}
            for f in as_completed(futures):
                res = f.result(); 
                if res: batch_data_ee.append(res); labels_temp.append(futures[f])
        if batch_data_ee:
            feats = presto.get_features(batch_data_ee)
            if len(feats.shape) == 3: feats = feats.mean(axis=1)
            X_features.append(feats); y_labels.extend(labels_temp)
            print(f"   Processed batch {i//BATCH_SIZE + 1}...")

    X = np.vstack(X_features)
    y = np.array(y_labels)

    # 2. Setup SVM
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1.5, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'))
    ])

    # 3. Run 5-Fold Cross Validation
    print("\n" + "="*50)
    print("📊 RUNNING 5-FOLD CROSS VALIDATION")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None) # No seed lock
    
    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')
    auc_scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')

    print(f"✅ Average Accuracy:  {acc_scores.mean():.4f}  (Range: {acc_scores.min():.2f} - {acc_scores.max():.2f})")
    print(f"✅ Average F1-Score:  {f1_scores.mean():.4f}")
    print(f"✅ Average ROC-AUC:   {auc_scores.mean():.4f}")
    print("-" * 50)

    # 4. Final Train & Save (ใช้ข้อมูลทั้งหมดเทรนเพื่อเก็บไว้ใช้งานจริง)
    print("\n⚔️ Training Final Model on ALL Data...")
    clf.fit(X, y)
    
    print("💾 Saving Model...")
    joblib.dump(clf, 'sinkhole_svm_pipeline.pkl')
    print("✅ Model saved to: sinkhole_svm_pipeline.pkl")

    # 5. Visualize (Full Fit - เพื่อดูแนวโน้ม)
    y_proba = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    
    print("\n🎨 Generating Visualizations...")
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    plt.rcParams.update({'font.size': 11})

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
    ax[0].set_title('Confusion Matrix (Full Data Fit)')

    try:
        fpr, tpr, _ = roc_curve(y, y_proba)
        ax[1].plot(fpr, tpr, color='#2ca02c', lw=3, label=f'Full AUC={roc_auc_score(y, y_proba):.3f}')
        ax[1].plot([0, 1], [0, 1], 'k--', alpha=0.3); ax[1].legend()
        ax[1].set_title('ROC Curve')
    except: pass

    sns.histplot(y_proba[y == 0], color='blue', label='Normal', kde=True, ax=ax[2], alpha=0.4)
    sns.histplot(y_proba[y == 1], color='red', label='Sinkhole', kde=True, ax=ax[2], alpha=0.4)
    ax[2].set_title('Confidence Separation'); ax[2].legend()

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()