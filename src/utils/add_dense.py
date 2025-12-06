import json
import math

# ค่าคงที่: ระยะห่างระหว่างจุดทำนาย (เมตร)
INTERVAL_METERS = 100 

def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทาง (เมตร)"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_point_at_fraction(p1, p2, fraction):
    """หาพิกัดจุดที่อยู่ระหว่าง p1, p2 ตามสัดส่วน (0.0 - 1.0)"""
    lon1, lat1 = p1
    lon2, lat2 = p2
    new_lon = lon1 + (lon2 - lon1) * fraction
    new_lat = lat1 + (lat2 - lat1) * fraction
    return [new_lon, new_lat]

# --- ส่วนหลักของโปรแกรม ---

input_file = './data/input/Mrt_1_BL.geojson' # ตรวจสอบ path ไฟล์ให้ถูกต้อง
output_geojson = f'mrt_resampled_{INTERVAL_METERS}m.geojson'
output_csv = f'./data/csv/mrt_resampled_{INTERVAL_METERS}m.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

final_features = []
csv_rows = ["latitude,longitude,line_name,color"]

print(f"กำลัง Resample ข้อมูลทุกๆ {INTERVAL_METERS} เมตร...")

for feature in data['features']:
    geom = feature.get('geometry')
    props = feature.get('properties', {})
    
    if not geom or geom['type'] != 'LineString': 
        continue
    
    line_name = props.get('name:en', props.get('name', 'Unknown Line'))
    color = props.get('colour', 'gray')
    coords = geom['coordinates']
    
    # --- Logic ใหม่: เดินสะสมระยะทาง ---
    
    # 1. เก็บจุดเริ่มต้นเสมอ
    current_dist_accumulator = 0  # ระยะทางที่เดินผ่านมาแล้วใน segment นี้
    target_dist = 0               # เป้าหมายระยะทางถัดไป (0, 1500, 3000...)
    
    # เพิ่มจุดแรก (ระยะ 0)
    start_p = coords[0]
    final_features.append({
        "type": "Feature",
        "properties": {"line": line_name, "color": color, "type": "start_node"},
        "geometry": {"type": "Point", "coordinates": start_p}
    })
    csv_rows.append(f"{start_p[1]},{start_p[0]},{line_name},{color}")
    
    target_dist += INTERVAL_METERS # เป้าหมายถัดไปคือ 1500m
    
    # วนลูปเดินไปตามรางทีละท่อน
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        
        # ความยาวของท่อนรางนี้
        segment_dist = haversine_distance(p1[1], p1[0], p2[1], p2[0])
        
        # ตรวจสอบว่าในท่อนนี้ มีจุดเป้าหมายตกอยู่ไหม (อาจมีหลายจุดถ้าท่อนยาวมาก)
        while current_dist_accumulator + segment_dist >= target_dist:
            # คำนวณว่าจุดเป้าหมายอยู่ตรงไหนในท่อนนี้ (ระยะที่ขาด / ความยาวท่อน)
            remaining_dist = target_dist - current_dist_accumulator
            fraction = remaining_dist / segment_dist if segment_dist > 0 else 0
            
            new_p = get_point_at_fraction(p1, p2, fraction)
            
            # บันทึกจุด Resample
            final_features.append({
                "type": "Feature",
                "properties": {"line": line_name, "color": color, "type": "resampled"},
                "geometry": {"type": "Point", "coordinates": new_p}
            })
            csv_rows.append(f"{new_p[1]},{new_p[0]},{line_name},{color}")
            
            # ขยับเป้าหมายไปอีก 1500m
            target_dist += INTERVAL_METERS
            
        # จบท่อนนี้ บวกระยะสะสมเพิ่ม
        current_dist_accumulator += segment_dist

# บันทึกไฟล์
out_data = {"type": "FeatureCollection", "features": final_features}

with open(output_geojson, 'w', encoding='utf-8') as f:
    json.dump(out_data, f, ensure_ascii=False)

with open(output_csv, 'w', encoding='utf-8') as f:
    f.write("\n".join(csv_rows))

print(f"เสร็จสิ้น! บันทึกไฟล์ {output_geojson} เรียบร้อย (จำนวน {len(final_features)} จุด)")