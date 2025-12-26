import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta
import datetime
from io import BytesIO
from typing import Dict, Any, List
import math
import hashlib
import re
import boto3
import pandas.core.common as pdc # Import for robust date offset handling
from dotenv import load_dotenv

load_dotenv()

import tensorflow as tf
load_model = tf.keras.models.load_model  

BASE_DIR = os.getcwd() # หรือระบุ path หลักของโปรเจกต์
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    
# # --- 🆕 S3 Helper Configuration ---
# S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
# S3_REGION = os.environ.get('AWS_REGION', 'ap-southeast-1')

# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
#     region_name=S3_REGION
# )

# def load_s3_bytes(file_key: str) -> BytesIO:
#     """ดึงไฟล์จาก S3 และส่งคืนเป็น BytesIO"""
#     try:
#         obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
#         return BytesIO(obj['Body'].read())
#     except Exception as e:
#         print(f"Error loading {file_key} from S3: {e}")
#         return None

# 🆕 กำหนดกรอบเวลาตายตัวสำหรับกราฟ
ACTUAL_MONTHS = 12 
PREDICT_MONTHS = 4

def load_model_system() -> Dict[str, Any]:
    """Load artifacts if present. Caller should handle exceptions."""
    artifacts = {
        'lstm_model': None,
        'xgb_classifier': None,
        'scaler_x': None,
        'scaler_y': None,
        'metadata': {}
    }

    # 1. load LSTM model (Keras จำเป็นต้องโหลดจากไฟล์จริงในดิสก์)
    try:
        model_path = os.path.join(MODEL_DIR, 'inventory_lstm_model.h5')
        if os.path.exists(model_path):
            artifacts['lstm_model'] = load_model(model_path)
            print(f"✅ Loaded LSTM model from {model_path}")
    except Exception as e:
        print(f"LSTM Load Error: {e}")

    # 2. load pickled artifacts (โหลดตรงจาก Memory ได้เลย)
    # try:
    #     xgb_bytes = load_s3_bytes('lead_time_classifier.pkl')
    #     if xgb_bytes: artifacts['xgb_classifier'] = joblib.load(xgb_bytes)
        
    #     sx_bytes = load_s3_bytes('scaler_x.pkl')
    #     if sx_bytes: artifacts['scaler_x'] = joblib.load(sx_bytes)
        
    #     sy_bytes = load_s3_bytes('scaler_y.pkl')
    #     if sy_bytes: artifacts['scaler_y'] = joblib.load(sy_bytes)
        
    #     meta_bytes = load_s3_bytes('model_metadata.pkl')
    #     if meta_bytes: artifacts['metadata'] = joblib.load(meta_bytes)
        
    #     print("✅ All artifacts loaded successfully from S3")
    # except Exception as e:
    #     print(f"Pickle Load Error: {e}")
    
    try:
        files_to_load = {
            'xgb_classifier': 'lead_time_classifier.pkl',
            'scaler_x': 'scaler_x.pkl',
            'scaler_y': 'scaler_y.pkl',
            'metadata': 'model_metadata.pkl'
        }

        for key, filename in files_to_load.items():
            file_path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(file_path):
                artifacts[key] = joblib.load(file_path)
                print(f"✅ Loaded {filename}")
        
        print("✅ All artifacts loaded successfully from local storage")
    except Exception as e:
        print(f"Pickle Load Error: {e}")

    return artifacts

# 🆕 ฟังก์ชันใหม่: ดึงข้อมูลอ้างอิงจากไฟล์ Excel บน Server
# def get_reference_data() -> bytes:
#     """Read the byte content of the reference Excel file."""
#     if not os.path.exists(REFERENCE_EXCEL_PATH):
#         raise FileNotFoundError(f"Reference Excel file not found at: {REFERENCE_EXCEL_PATH}")
    
#     with open(REFERENCE_EXCEL_PATH, 'rb') as f:
#         return f.read()

def get_reference_data() -> bytes:
    file_path = os.path.join(DATA_DIR, 'Training_Data_Final.xlsx')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference Excel file not found at: {file_path}")
    
    with open(file_path, 'rb') as f:
        return f.read()

def _stable_seed_from_sku(sku: Any, offset: int = 0) -> int:
    """Deterministic seed from SKU + offset using MD5 (stable across runs)."""
    key = str(sku).encode('utf-8')
    h = hashlib.md5(key).hexdigest()
    num = int(h[:8], 16)
    return (num + offset) % (2**32)


def prepare_lstm_input(df_latest: pd.DataFrame, sku: Any, metadata: Dict[str, Any], scaler_x, scaler_y, lt_category_pred):
    """
    Build LSTM input sequence from snapshot row.
    NOTE: In the recursive prediction, this function's output must be carefully updated.
    For standard single prediction, it repeats latest Usage_Qty as fallback.
    """
    time_steps = int(metadata.get('time_steps', 12))
    features = list(metadata.get('valid_features_lstm', []))

    row = df_latest[df_latest['SKU'] == sku].iloc[0]
    try:
        current_usage = float(row.get('Usage_Qty', 0) or 0)
    except Exception:
        current_usage = 0.0

    seq_data = []
    for _ in range(time_steps):
        month_record = {}
        for f in features:
            if f == 'LT_CATEGORY_PRED':
                month_record[f] = lt_category_pred
            elif f == 'Safety_Stock_Qty':
                month_record[f] = row.get('Safety_Stock_Qty', 0)
            elif f == 'Total_SKU_Usage':
                month_record[f] = row.get('Total_SKU_Usage', 0)
            elif f == 'Patient_Count':
                month_record[f] = row.get('Patient_Count', 1500)
            elif f in ['Patient_E', 'Patient_I', 'Patient_O']:
                month_record[f] = row.get(f, 0)
            elif f in ['month_num', 'year_num', 'day_index']:
                month_record[f] = 0
            else:
                month_record[f] = row.get(f, 0)
        month_record['Usage_Qty'] = current_usage
        seq_data.append(month_record)

    df_seq = pd.DataFrame(seq_data)
    keep_cols = [c for c in features + ['Usage_Qty'] if c in df_seq.columns]
    df_seq = df_seq[keep_cols]

    # prepare arrays
    x_cols = [c for c in features if c in df_seq.columns]
    x_vals = df_seq[x_cols].values if len(x_cols) > 0 else np.zeros((len(df_seq), 0))
    y_vals = df_seq[['Usage_Qty']].values if 'Usage_Qty' in df_seq.columns else np.zeros((len(df_seq), 1))

    # scale if scalers provided; otherwise return raw arrays (caller must handle)
    if scaler_x is not None and hasattr(scaler_x, 'transform'):
        x_scaled = scaler_x.transform(x_vals)
    else:
        x_scaled = x_vals

    if scaler_y is not None and hasattr(scaler_y, 'transform'):
        # scaler_y expects 2D
        y_scaled = scaler_y.transform(y_vals.reshape(-1, 1))
    else:
        y_scaled = y_vals

    # Combined input array (X_scaled | Y_scaled)
    combined = np.hstack([x_scaled, y_scaled])
    return np.array([combined])

def recursive_predict_monthly(
    sku_row,
    sku,
    lt_cat,
    metadata,
    lstm_model,
    scaler_x,
    scaler_y,
    predict_months,
    df_history_all,
    use_real_history=False
):

    import numpy as np
    import pandas as pd

    time_steps = int(metadata.get("time_steps", 12))
    features = metadata.get("valid_features_lstm", [])

    # -------------------------------
    # 1. LOAD HISTORY
    # -------------------------------
    sku_hist = (
        df_history_all[df_history_all["SKU"] == str(sku).strip()]
        .sort_values("Date")
        .copy()
    )

    if len(sku_hist) < 12:
        return [0] * predict_months

    sku_hist["Date"] = pd.to_datetime(sku_hist["Date"])
    usage = sku_hist["Usage_Qty"].astype(float)

    h_mean = usage.mean()
    h_median = usage.median()
    h_p75 = usage.quantile(0.75)
    h_p90 = usage.quantile(0.90)

    zero_ratio = (usage == 0).mean()
    semi_intermittent = (h_median < 15) and (zero_ratio > 0.2)

    # -------- FLOOR เพื่อกันลง 0 ----------
    min_nonzero = usage[usage > 0].min() if (usage > 0).any() else 0
    base_floor = max(
        1.0,
        h_mean * 0.25 if h_mean > 0 else 0,
        h_median * 0.5 if h_median > 0 else 0,
        min_nonzero * 0.8 if min_nonzero else 0,
    )

    # -------------------------------
    # 2. SEASONAL
    # -------------------------------
    df_history_all = df_history_all.copy()
    df_history_all["Date"] = pd.to_datetime(df_history_all["Date"])
    df_history_all["month"] = df_history_all["Date"].dt.month

    seasonal_p90 = (
        df_history_all[df_history_all["SKU"] == str(sku).strip()]
        .groupby("month")["Usage_Qty"]
        .quantile(0.90)
        .to_dict()
    )

    # -------------------------------
    # 3. SEED
    # -------------------------------
    history = (
        usage.head(time_steps).tolist()
        if use_real_history
        else usage.tail(time_steps).tolist()
    )

    start_date = (
        sku_hist["Date"].iloc[time_steps - 1]
        if use_real_history
        else sku_hist["Date"].max()
    )

    preds = []

    # -------------------------------
    # 4. LOOP
    # -------------------------------
    for step in range(1, predict_months + 1):

        target_month = (start_date + pd.DateOffset(months=step)).month
        month_cap = seasonal_p90.get(target_month, h_p75)

        if semi_intermittent:
            usage_cap = max(h_p75 * 1.1, h_median * 1.25)
        else:
            usage_cap = max(h_p90 * 1.18, h_mean * 1.25)

        usage_cap = min(usage_cap, month_cap * 1.18)

        # --------- Build sequence ---------
        seq = []
        for i in range(time_steps):
            rec = {f: sku_row.get(f, 0) for f in features}
            rec["Usage_Qty"] = history[-time_steps + i]

            for p in ["Patient_E", "Patient_I", "Patient_O"]:
                if p in rec:
                    rec[p] = 0.0

            if "month_num" in features:
                rec["month_num"] = ((target_month - time_steps + i - 1) % 12) + 1

            seq.append(rec)

        df_seq = pd.DataFrame(seq)

        x = scaler_x.transform(df_seq[features].values)
        y = scaler_y.transform(df_seq[["Usage_Qty"]].values)
        model_in = np.array([np.hstack([x, y])])

        raw = lstm_model.predict(model_in, verbose=0)
        pred = scaler_y.inverse_transform(raw)[0][0]

        # -------------------------------
        # 5. STABILIZATION
        # -------------------------------
        anchor = h_median if h_median > 0 else h_mean
        alpha = 0.65 if not semi_intermittent else 0.50
        stabilized = (pred * alpha) + (anchor * (1 - alpha))

        # growth clamp แต่ไม่แรงเกิน
        if len(history) > 0:
            last_real = history[-1]
            max_allowed = last_real * 1.7
            stabilized = min(stabilized, max_allowed)

        # soft cap
        if stabilized > usage_cap:
            stabilized = usage_cap * np.random.uniform(0.93, 0.99)

        # momentum (กันนิ่งตรง)
        if len(preds) >= 2:
            diff = preds[-1] - preds[-2]
            stabilized += diff * 0.35

        # --------- ANTI ZERO ----------
        if stabilized <= base_floor:
            stabilized = base_floor * np.random.uniform(0.95, 1.08)

        stabilized = max(0.0, stabilized)

        # randomness เล็กน้อย
        stabilized *= np.random.uniform(0.98, 1.03)

        preds.append(int(round(stabilized)))

        # history ใช้ float
        history.append(stabilized)

    return preds
# ⭐⭐⭐ END OF RECURSIVE PREDICTION FUNCTION ⭐⭐⭐


def get_monthly_chart_data(sku_list, sku_to_item_name_map, metadata, lstm_model, scaler_x, scaler_y, df_latest, is_upload=False):
    all_chart_data = []
    time_steps = int(metadata.get('time_steps', 12))
    
    
    # ปรับปรุง: ดึงไฟล์ประวัติจาก S3
    # excel_bytes = load_s3_bytes('Training_Data_Final.xlsx')
    # if not excel_bytes: return []
    
    file_path = os.path.join(DATA_DIR, 'Training_Data_Final.xlsx')
    if not os.path.exists(file_path):
        print(f"History file not found at {file_path}")
        return []

    df_history = pd.read_excel(file_path)
    df_history.columns = df_history.columns.str.strip()
    df_history['Date'] = pd.to_datetime(df_history['Date'])
    df_history['SKU'] = df_history['SKU'].astype(str).str.strip()
    df_history['Month_Year'] = df_history['Date'].dt.to_period('M').dt.to_timestamp()
    df_monthly_agg = df_history.groupby(['Month_Year', 'SKU']).agg({'Usage_Qty': 'sum'}).reset_index()
    
    # new predict from upload file
    if is_upload:
        current_month_start = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        for _, upload_row in df_latest.iterrows():
            sku_val = str(upload_row['SKU']).strip()
            new_usage = float(upload_row.get('Usage_Qty', 0))
            
            mask = (df_monthly_agg['SKU'] == sku_val) & (df_monthly_agg['Month_Year'] == current_month_start)
            if mask.any():
                df_monthly_agg.loc[mask, 'Usage_Qty'] = new_usage
            else:
                new_record = pd.DataFrame([{'Month_Year': current_month_start, 'SKU': sku_val, 'Usage_Qty': new_usage}])
                df_monthly_agg = pd.concat([df_monthly_agg, new_record], ignore_index=True)
    
    

    for sku in sku_list:
        sku_clean = str(sku).strip()
        item_name = sku_to_item_name_map.get(sku, f'SKU {sku}')
        sku_data = df_monthly_agg[df_monthly_agg['SKU'] == sku_clean].sort_values('Month_Year')
        
        # กรองเฉพาะ SKU ที่มีประวัติ > 12 เดือน
        if len(sku_data) <= time_steps: continue

        # 1. ACTUAL DATA
        for _, row in sku_data.iterrows():
            all_chart_data.append({
                'date': row['Month_Year'].strftime('%Y-%m-%d'), 
                'usage': float(row['Usage_Qty']), 
                'SKU': sku_clean, 'Item_Name': item_name, 'type': 'actual'
            })

        sku_snap = df_latest[df_latest['SKU'] == sku].iloc[0] if sku in df_latest['SKU'].values else None
        
        if lstm_model and sku_snap is not None:
            try:
                # 2. PREDICT PAST (ทำนายย้อนหลังหลังจากเจอรายการ 12 ครั้งขึ้นไป)
                # ใช้จุดเริ่มจากข้อมูล 12 เดือนแรก เพื่อทายเดือนที่ 13 จนถึงเดือนล่าสุดที่มีในไฟล์
                past_predict_months = len(sku_data) - time_steps
                if past_predict_months > 0:
                    past_vals = recursive_predict_monthly(sku_snap, sku_clean, 0, metadata, lstm_model, scaler_x, scaler_y, past_predict_months, df_history, True)
                    # หา Date ของเดือนที่ 13 เป็นต้นไป
                    past_dates = sku_data['Month_Year'].iloc[time_steps:].tolist()
                    for idx, val in enumerate(past_vals):
                        all_chart_data.append({
                            'date': past_dates[idx].strftime('%Y-%m-%d'), 
                            'usage': val, 'SKU': sku_clean, 'Item_Name': item_name, 'type': 'predict_past'
                        })

                # 3. PREDICT FUTURE (ทำนายล่วงหน้า 12 เดือนจากปัจจุบัน)
                future_vals = recursive_predict_monthly(sku_snap, sku_clean, 0, metadata, lstm_model, scaler_x, scaler_y, 12, df_history, False)
                start_f = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0)
                for m in range(12):
                    all_chart_data.append({
                        'date': (start_f + pd.DateOffset(months=m)).strftime('%Y-%m-%d'), 
                        'usage': future_vals[m], 'SKU': sku_clean, 'Item_Name': item_name, 'type': 'predicted'
                    })
            except Exception as e:
                print(f"Error predicting SKU {sku_clean}: {e}")
        
    return all_chart_data


def predict_inventory_usage(system_artifacts: Dict[str, Any], file_content: bytes, forecast_days: int, is_upload: bool = False) -> Dict[str, Any]:
    """
    ฟังก์ชันพยากรณ์และวางแผนสต็อกที่ทำงานสัมพันธ์กับกราฟและช่วงเวลาที่เลือก
    โดยใช้ค่าเฉลี่ยคนไข้จากอดีต 3 ปีมาคำนวณโดยอัตโนมัติ
    """
    lstm_model = system_artifacts.get('lstm_model')
    xgb_classifier = system_artifacts.get('xgb_classifier')
    scaler_x = system_artifacts.get('scaler_x')
    scaler_y = system_artifacts.get('scaler_y')
    metadata = system_artifacts.get('metadata', {})

    # 1. โหลดข้อมูลและจัดการชื่อคอลัมน์ให้ยืดหยุ่น
    df_latest = pd.read_excel(BytesIO(file_content), sheet_name=0)
    df_latest.columns = df_latest.columns.str.strip()
    
    RENAME_MAP = {
        'Latest Usage Qty': 'Usage_Qty', 'Latest_Usage_Qty': 'Usage_Qty',
        'Min Stock': 'Safety_Stock_Qty', 'Min_Stock': 'Safety_Stock_Qty',
        'Current Stock': 'Stock_on_Hand_Qty', 'Current_Stock': 'Stock_on_Hand_Qty',
        'Max Stock': 'Max_Stock_Qty', 'Max_Stock': 'Max_Stock_Qty',
        'Unit Cost': 'Unit_Cost', 'Cost_Per_Unit': 'Unit_Cost',
        'Lead Time Days': 'Lead_Time_Days', 'Item Name': 'Item_Name'
    }
    df_latest.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df_latest.columns}, inplace=True)
    df_latest['SKU'] = df_latest['SKU'].astype(str).str.strip()
    df_latest.drop_duplicates(subset=['SKU'], keep='last', inplace=True)

    # จัดการค่าเริ่มต้นและแปลงเป็นตัวเลข
    numeric_defaults = {
        'Stock_on_Hand_Qty': 0, 'Safety_Stock_Qty': 0, 
        'Unit_Cost': 0, 'Lead_Time_Days': 14, 'Max_Stock_Qty': 0
    }
    for col, default in numeric_defaults.items():
        if col not in df_latest.columns: df_latest[col] = default
        df_latest[col] = pd.to_numeric(df_latest[col], errors='coerce').fillna(default)

    # 2. โหลดประวัติข้อมูล 3 ปี เพื่อหา Seasonal Patient Map (ค่าเฉลี่ยคนไข้รายเดือน)
    file_path_hist = os.path.join(DATA_DIR, 'Training_Data_Final.xlsx')
    df_history_all = pd.read_excel(file_path_hist) if os.path.exists(file_path_hist) else pd.DataFrame()
    
    patient_seasonal_avg = {}
    if not df_history_all.empty:
        df_history_all['SKU'] = df_history_all['SKU'].astype(str).str.strip()
        df_history_all['Date'] = pd.to_datetime(df_history_all['Date'])
        # ⭐ คำนวณค่าเฉลี่ยคนไข้แยกรายเดือนจากข้อมูลย้อนหลัง
        if 'Patient_Count' in df_history_all.columns:
            patient_seasonal_avg = df_history_all.groupby(df_history_all['Date'].dt.month)['Patient_Count'].mean().to_dict()

    all_actions = []
    unique_skus = df_latest['SKU'].unique().tolist()
    sku_to_item_name_map = df_latest.set_index('SKU')['Item_Name'].to_dict()

    # 3. ลูปคำนวณราย SKU เพื่อหา Action Items
    for _, row in df_latest.iterrows():
        sku = row['SKU']
        up_usage = float(row.get('Usage_Qty', 0))

        # 3.1 พยากรณ์รายเดือนด้วย LSTM (สัมพันธ์กับจำนวนวันที่เลือก)
        future_vals = []
        months_to_predict = max(1, math.ceil(forecast_days / 30) + 1)
        
        if lstm_model and not df_history_all.empty:
            try:
                # ⭐ ดึงค่าเฉลี่ยคนไข้ตามเดือนปัจจุบันมาใช้แทนการกรอกมือ
                current_month = datetime.datetime.now().month
                row_with_patient = row.copy()
                row_with_patient['Patient_Count'] = patient_seasonal_avg.get(current_month, 1500)
                
                future_vals = recursive_predict_monthly(
                    row_with_patient, sku, 0, metadata, lstm_model, 
                    scaler_x, scaler_y, months_to_predict, df_history_all, False
                )
            except: 
                future_vals = [up_usage] * months_to_predict

        # 3.2 คำนวณความต้องการสะสม (Dynamic Demand Sum) ตาม Forecast Days จริง
        predicted_demand_sum = 0.0
        days_rem = forecast_days
        if future_vals:
            for m_val in future_vals:
                if days_rem <= 0: break
                if days_rem >= 30:
                    predicted_demand_sum += m_val
                    days_rem -= 30
                else:
                    predicted_demand_sum += (m_val / 30.0) * days_rem
                    days_rem = 0
            base_daily_rate = future_vals[0] / 30.0
        else:
            base_daily_rate = up_usage / 30.0
            predicted_demand_sum = base_daily_rate * forecast_days

        # 3.3 วิเคราะห์ความสำคัญ (Priority) และยอดสั่งซื้อ
        SOH = row['Stock_on_Hand_Qty']
        Min_Stock = row['Safety_Stock_Qty']
        Max_Stock = row['Max_Stock_Qty']
        LT_Days = row['Lead_Time_Days']

        rop_threshold = (base_daily_rate * LT_Days) + Min_Stock
        needed_for_period = predicted_demand_sum + Min_Stock

        if SOH < rop_threshold: priority = 'High Priority'
        elif SOH < needed_for_period: priority = 'Medium Priority'
        else: priority = 'Low Priority'

        reorder_qty = 0
        if priority != 'Low Priority':
            target = min(Max_Stock, needed_for_period) if Max_Stock > 0 else needed_for_period
            reorder_qty = max(0, int(math.ceil(target - SOH)))

        all_actions.append({
            'sku': sku,
            'item_name': row.get('Item_Name', f'SKU {sku}'),
            'priority': priority,
            'recommended_qty': reorder_qty,
            'reorder_cost': round(reorder_qty * row['Unit_Cost'], 2),
            'current_soh': int(round(SOH)),
            'rop_threshold': int(round(rop_threshold)),
            'max_stock_policy': int(Max_Stock),
            'demand_for_period': int(round(predicted_demand_sum)),
            'lead_time_days': int(LT_Days)
        })

    # 4. ดึงข้อมูลกราฟรายเดือน
    monthly_chart_data = get_monthly_chart_data(
        unique_skus, sku_to_item_name_map, metadata, 
        lstm_model, scaler_x, scaler_y, df_latest, is_upload
    )

    return {
        "Monthly_Chart_Data": monthly_chart_data,
        "metrics": {
            'total_skus': int(len(df_latest)),
            'high_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'High Priority'],
            'medium_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'Medium Priority'],
            'reorder_cost_total': round(sum(item['reorder_cost'] for item in all_actions), 2),
            'action_items': all_actions,
        }
    }
