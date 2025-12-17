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
import pandas.core.common as pdc # Import for robust date offset handling

# optional: tensorflow import only if model exists in runtime
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None  # safe fallback for environments without TF

# Paths (update as needed)
MODEL_LSTM_PATH = 'models/inventory_lstm_model.h5'
MODEL_XGB_CLASS_PATH = 'models/lead_time_classifier.pkl'
SCALER_X_PATH = 'models/scaler_x.pkl'
SCALER_Y_PATH = 'models/scaler_y.pkl'
METADATA_PATH = 'models/model_metadata.pkl'

# 🆕 กำหนดกรอบเวลาตายตัวสำหรับกราฟ
ACTUAL_MONTHS = 12 
PREDICT_MONTHS = 4

# 🆕 Path สำหรับไฟล์ Excel อ้างอิงที่ใช้ในการ Initial Load
REFERENCE_EXCEL_PATH = 'data/Training_Data_Final.xlsx' 

def load_model_system() -> Dict[str, Any]:
    """Load artifacts if present. Caller should handle exceptions."""
    artifacts = {
        'lstm_model': None,
        'xgb_classifier': None,
        'scaler_x': None,
        'scaler_y': None,
        'metadata': {}
    }

    # load LSTM model if available and tensorflow present
    try:
        if load_model and os.path.exists(MODEL_LSTM_PATH):
            artifacts['lstm_model'] = load_model(MODEL_LSTM_PATH)
    except Exception:
        artifacts['lstm_model'] = None

    # load pickled artifacts if exist
    try:
        if os.path.exists(MODEL_XGB_CLASS_PATH):
            artifacts['xgb_classifier'] = joblib.load(MODEL_XGB_CLASS_PATH)
    except Exception:
        artifacts['xgb_classifier'] = None

    try:
        if os.path.exists(SCALER_X_PATH):
            artifacts['scaler_x'] = joblib.load(SCALER_X_PATH)
    except Exception:
        artifacts['scaler_x'] = None

    try:
        if os.path.exists(SCALER_Y_PATH):
            artifacts['scaler_y'] = joblib.load(SCALER_Y_PATH)
    except Exception:
        artifacts['scaler_y'] = None

    try:
        if os.path.exists(METADATA_PATH):
            artifacts['metadata'] = joblib.load(METADATA_PATH)
    except Exception:
        artifacts['metadata'] = {}

    return artifacts

# 🆕 ฟังก์ชันใหม่: ดึงข้อมูลอ้างอิงจากไฟล์ Excel บน Server
def get_reference_data() -> bytes:
    """Read the byte content of the reference Excel file."""
    if not os.path.exists(REFERENCE_EXCEL_PATH):
        raise FileNotFoundError(f"Reference Excel file not found at: {REFERENCE_EXCEL_PATH}")
    
    with open(REFERENCE_EXCEL_PATH, 'rb') as f:
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

    usage = sku_hist["Usage_Qty"].astype(float)

    h_mean = usage.mean()
    h_median = usage.median()
    h_p75 = usage.quantile(0.75)
    h_p90 = usage.quantile(0.90)

    zero_ratio = (usage == 0).mean()

    semi_intermittent = (h_median < 15) and (zero_ratio > 0.2)

    # -------------------------------
    # 2. SEASONAL CAP (ROBUST)
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
    # 4. FORECAST
    # -------------------------------
    for step in range(1, predict_months + 1):

        target_month = (start_date + pd.DateOffset(months=step)).month

        month_cap = seasonal_p90.get(target_month, h_p75)

        if semi_intermittent:
            usage_cap = max(h_p75 * 1.1, h_median * 1.3)
            alpha = 0.45
        else:
            usage_cap = max(h_p90 * 1.15, h_mean * 1.2)
            alpha = min(0.7, 0.55 + step * 0.03)

        seq = []
        for i in range(time_steps):
            rec = {f: sku_row.get(f, 0) for f in features}
            rec["Usage_Qty"] = history[-time_steps + i]

            # ❌ ignore patient for chart
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
        # 5. STABILIZATION (FIX FLAT LINE)
        # -------------------------------

        # ใช้ median เป็น anchor
        anchor = h_median if h_median > 0 else h_mean

        # alpha ลดลงเมื่อเริ่มนิ่ง
        alpha = 0.55 if semi_intermittent else min(0.65, 0.55 + step * 0.02)

        stabilized = (pred * alpha) + (anchor * (1 - alpha))

        # ---- SOFT CAP (สำคัญมาก) ----
        if stabilized > usage_cap:
            stabilized = usage_cap * 0.95 + np.random.uniform(-1.0, 1.0)

        # ---- FLOOR ----
        stabilized = max(0.0, stabilized)

        # ---- BREAK CONSTANT LOOP ----
        if len(preds) >= 2:
            if abs(preds[-1] - stabilized) < 0.5:
                stabilized *= np.random.uniform(0.96, 1.04)

        preds.append(float(stabilized))
        history.append(stabilized)

    return preds



# ⭐⭐⭐ END OF RECURSIVE PREDICTION FUNCTION ⭐⭐⭐


def get_monthly_chart_data(sku_list, sku_to_item_name_map, metadata, lstm_model, scaler_x, scaler_y, df_latest):
    all_chart_data = []
    time_steps = int(metadata.get('time_steps', 12))
    
    if not os.path.exists(REFERENCE_EXCEL_PATH): return []

    df_history = pd.read_csv(REFERENCE_EXCEL_PATH) if REFERENCE_EXCEL_PATH.endswith('.csv') else pd.read_excel(REFERENCE_EXCEL_PATH)
    df_history.columns = df_history.columns.str.strip()
    df_history['Date'] = pd.to_datetime(df_history['Date'])
    df_history['SKU'] = df_history['SKU'].astype(str).str.strip()
    df_history['Month_Year'] = df_history['Date'].dt.to_period('M').dt.to_timestamp()
    
    df_monthly_agg = df_history.groupby(['Month_Year', 'SKU']).agg({'Usage_Qty': 'sum'}).reset_index()

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


def predict_inventory_usage(system_artifacts: Dict[str, Any], file_content: bytes, forecast_days: int) -> Dict[str, Any]:
    """
    Main function: reads Excel bytes, classifies lead-time risk, forecasts (LSTM if available),
    creates rule-based recommendations, and returns forecast + metrics, including Monthly Chart data.
    """
    lstm_model = system_artifacts.get('lstm_model')
    xgb_classifier = system_artifacts.get('xgb_classifier')
    scaler_x = system_artifacts.get('scaler_x')
    scaler_y = system_artifacts.get('scaler_y')
    metadata = system_artifacts.get('metadata', {})

    # Load input snapshot (first sheet)
    df_latest = pd.read_excel(BytesIO(file_content), sheet_name=0)
    df_latest.columns = df_latest.columns.str.strip()

    # Renaming map (extend if needed)
    RENAME_MAP = {
        'Next_Patient_Count_M1': 'Patient_Count',
        'Latest_Usage_Qty': 'Usage_Qty',
        'Min_Stock': 'Safety_Stock_Qty',
        'Current_Stock': 'Stock_on_Hand_Qty',
        'Cost_Per_Unit': 'Unit_Cost',
        'Item Name': 'Item_Name',
        'ItemName': 'Item_Name',
        'Product Name': 'Item_Name',
        'Description': 'Item_Name',
        'Max_Stock': 'Max_Stock_Qty',
        'Max Stock': 'Max_Stock_Qty',
        'Next_Patient_E_M1': 'Patient_E',
        'Next_Patient_I_M1': 'Patient_I',
        'Next_Patient_O_M1': 'Patient_O',
    }
    columns_to_rename = {k: v for k, v in RENAME_MAP.items() if k in df_latest.columns}
    if columns_to_rename:
        df_latest.rename(columns=columns_to_rename, inplace=True)

    # Ensure required columns exist and types are numeric where appropriate
    if 'Item_Name' not in df_latest.columns:
        df_latest['Item_Name'] = None
    if 'Max_Stock_Qty' not in df_latest.columns:
        df_latest['Max_Stock_Qty'] = 0
    if 'Unit_Cost' not in df_latest.columns:
        df_latest['Unit_Cost'] = 0
    if 'SKU' not in df_latest.columns:
        raise KeyError('Input Excel must contain a SKU column')
    
    # Normalize SKU to string for stable operations
    df_latest['SKU'] = df_latest['SKU'].astype(str)

    # 🔥 FIX 1: Remove duplicate SKUs from the input DataFrame.
    df_latest.drop_duplicates(subset=['SKU'], keep='last', inplace=True)

    # Map common column variants
    if 'Usage_Qty' in df_latest.columns:
        df_latest['Usage_Qty'] = pd.to_numeric(df_latest['Usage_Qty'], errors='coerce').fillna(0)
    else:
        df_latest['Usage_Qty'] = 0

    # Ensure numeric columns exist and are filled
    numeric_cols_defaults = {
        'Stock_on_Hand_Qty': 0,
        'Safety_Stock_Qty': 0,
        'Unit_Cost': 0,
        'Lead_Time_Days': 14,
        'Max_Stock_Qty': 0,
        'Conversion_Factor': 1
    }
    for col, default in numeric_cols_defaults.items():
        if col not in df_latest.columns:
            df_latest[col] = default
        else:
            df_latest[col] = pd.to_numeric(df_latest[col], errors='coerce').fillna(default)

    # Apply conversion factor if given (to standardize units)
    if 'Conversion_Factor' in df_latest.columns:
        conv = df_latest['Conversion_Factor'].fillna(1)
        df_latest['Usage_Qty'] = df_latest['Usage_Qty'] * conv
        df_latest['Stock_on_Hand_Qty'] = df_latest['Stock_on_Hand_Qty'] * conv

    # Classifier features fallback
    class_features = metadata.get('class_features', ['Unit_Cost', 'Safety_Stock_Qty'])
    for f in class_features:
        if f not in df_latest.columns:
            df_latest[f] = 0

    # Predict LT category if classifier present
    if xgb_classifier is not None:
        try:
            X_class_input = df_latest[class_features].fillna(0)
            df_latest['LT_CATEGORY_PRED'] = xgb_classifier.predict(X_class_input)
        except Exception:
            df_latest['LT_CATEGORY_PRED'] = 0
    else:
        df_latest['LT_CATEGORY_PRED'] = 0

    # Outputs for Action Items
    forecast_result_rows: List[Dict[str, Any]] = []
    all_actions: List[Dict[str, Any]] = []
    sku_policy_map = metadata.get('sku_policy_map', {})

    # Forecast start date = tomorrow (midnight + 1 day) - Used for Action Items only
    current_time = datetime.datetime.now()
    today_midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today_midnight + timedelta(days=1)
    
    # Collect all unique SKUs for processing
    unique_skus = df_latest['SKU'].unique().tolist() 
    sku_to_item_name_map = df_latest.set_index('SKU')['Item_Name'].to_dict()

    # Iterate SKUs to calculate Action Items (using existing daily forecast logic)
    for _, row in df_latest.iterrows():
        sku = row['SKU']
        lt_cat = row.get('LT_CATEGORY_PRED', 0)

        item_name_raw = row.get('Item_Name')
        item_name = str(item_name_raw).strip() if not pd.isna(item_name_raw) and str(item_name_raw).strip() != '' else f'SKU {sku}'


        # ---------- Forecasting (Re-run for Action Item Daily Demand Sum) ----------
        predicted_usage_monthly = float(row.get('Usage_Qty', 0) or 0) 
        
        # Get U_hat for ROP/Coverage calculation 
        if lstm_model and scaler_x and scaler_y:
            try:
                input_seq = prepare_lstm_input(df_latest, sku, metadata, scaler_x, scaler_y, lt_cat)
                scaled_pred = lstm_model.predict(input_seq, verbose=0)
                if np.ndim(scaled_pred) == 2 and scaled_pred.shape[1] >= 1:
                    actual_pred = scaler_y.inverse_transform(scaled_pred)[0][0]
                else:
                    actual_pred = float(np.mean(scaler_y.inverse_transform(scaled_pred.reshape(-1, 1))))
                predicted_usage_monthly = float(max(0.0, actual_pred))
            except Exception:
                pass # Use fallback

        if predicted_usage_monthly == 0:
            predicted_usage_monthly = float(row.get('Usage_Qty', 0) or 0)

        # Average daily usage (used only as base rate for daily noise and ROP)
        base_daily_usage_rate = predicted_usage_monthly / 30.0 if predicted_usage_monthly > 0 else 0.0

        predicted_demand_sum = 0.0
        
        # Generate daily forecast for Action Item period (forecast_days)
        for i in range(int(forecast_days)):
            seed = _stable_seed_from_sku(sku, i)
            np.random.seed(seed)
            random_factor = np.random.uniform(0.95, 1.05)
            predicted_daily_usage_float = base_daily_usage_rate * random_factor
            predicted_daily_usage = max(0.0, float(predicted_daily_usage_float))
            
            # Summing up demand over the entire forecast period
            predicted_demand_sum += predicted_daily_usage 
            
            forecast_date = start_date + timedelta(days=i)
            # Keeping this daily data, although main chart uses monthly data now
            forecast_result_rows.append({ 
                'date': forecast_date.isoformat(),
                'predicted_usage': predicted_daily_usage,
                'SKU': sku,
                'Item_Name': item_name
            })

        # ---------- Rule-based recommendation (DYNAMICALLY LINKED TO forecast_days) ----------
        Max_Stock = int(row.get('Max_Stock_Qty', 0) or 0)
        Min_Stock = float(row.get('Safety_Stock_Qty', 0) or 0)
        UC = float(row.get('Unit_Cost', 0) or 0)
        LT_Days = int(row.get('Lead_Time_Days', 14) or 14)
        SOH = float(row.get('Stock_on_Hand_Qty', 0) or 0)

        
        # ROP Threshold (Static part based on Lead Time)
        rop_threshold = (base_daily_usage_rate * LT_Days) + Min_Stock
        
        # Dynamic Coverage Threshold (Based on Forecast Days)
        needed_for_forecast = predicted_demand_sum + Min_Stock

        stock_out_date = None
        reorder_qty_float = 0.0

        # Determine priority based on ROP and Forecast Coverage (DYNAMIC)
        is_high_risk = (SOH < rop_threshold) 
        is_medium_risk = (not is_high_risk) and (SOH < needed_for_forecast) 
        
        if is_high_risk:
            priority_group = 'High Priority'
        elif is_medium_risk:
            priority_group = 'Medium Priority'
        else:
            priority_group = 'Low Priority'

        # Reorder Calculation based on Dynamic Policy
        if priority_group in ['High Priority', 'Medium Priority']:
            if Max_Stock > 0:
                final_target_stock = min(Max_Stock, needed_for_forecast)
                reorder_qty_float = max(0.0, final_target_stock - SOH)
            else:
                reorder_qty_float = max(0.0, predicted_demand_sum + Min_Stock - SOH)
        
        reorder_qty = int(math.ceil(reorder_qty_float))
        reorder_cost = reorder_qty * UC
        
        # Stock out date calculation
        if base_daily_usage_rate > 0 and SOH < (base_daily_usage_rate * LT_Days):
            days_until_out = int(max(0, math.floor(SOH / base_daily_usage_rate)))
            stock_out_date = (start_date + timedelta(days=days_until_out)).strftime('%Y-%m-%d')

        all_actions.append({
            'sku': sku,
            'item_name': item_name,
            'priority': priority_group, 
            'stock_out_date': stock_out_date if stock_out_date else 'N/A',
            'recommended_qty': int(reorder_qty), 
            'reorder_cost': round(reorder_cost, 2), 
            'current_soh': int(round(SOH)),
            'rop_threshold': int(round(rop_threshold)),
            'max_stock_policy': int(Max_Stock),
            'lead_time_days': LT_Days
        })

    # Sum reorder_cost across all items (DYNAMIC)
    reorder_cost_total = round(sum(item['reorder_cost'] for item in all_actions), 2)
    
    # ⭐⭐⭐ GET MONTHLY CHART DATA (12 Actual + 4 Predict) ⭐⭐⭐
    monthly_chart_data = get_monthly_chart_data(
        sku_list=unique_skus, 
        sku_to_item_name_map=sku_to_item_name_map,
        metadata=metadata,
        lstm_model=lstm_model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        df_latest=df_latest
    )
    # ⭐⭐⭐ END MONTHLY CHART DATA STEP ⭐⭐⭐

    return {
        "Daily_Forecast_Data": forecast_result_rows, 
        
        "Monthly_Chart_Data": monthly_chart_data, # 🆕 New Key for the chart
        
        "metrics": {
            'total_skus': int(len(df_latest)), 
            'high_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'High Priority'], 
            'medium_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'Medium Priority'], 
            'reorder_cost_total': reorder_cost_total, 
            'action_items': all_actions, 
        }
    }