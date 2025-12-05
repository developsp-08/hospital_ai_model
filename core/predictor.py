# core/predictor.py (FINAL VERSION - HYBRID INFERENCE & RECOMMENDATION)

import pandas as pd
import numpy as np
import os
import joblib
from datetime import timedelta
import datetime
from io import BytesIO
from typing import Dict, Any, List

from tensorflow.keras.models import load_model

# Paths
MODEL_LSTM_PATH = 'models/inventory_lstm_model.h5'
MODEL_XGB_CLASS_PATH = 'models/lead_time_classifier.pkl'
SCALER_X_PATH = 'models/scaler_x.pkl'
SCALER_Y_PATH = 'models/scaler_y.pkl'
METADATA_PATH = 'models/model_metadata.pkl'

def load_model_system() -> Dict[str, Any]:
    if not os.path.exists(MODEL_LSTM_PATH):
        raise FileNotFoundError("Model files missing. Run training script first.")
    
    return {
        'lstm_model': load_model(MODEL_LSTM_PATH),
        'xgb_classifier': joblib.load(MODEL_XGB_CLASS_PATH),
        'scaler_x': joblib.load(SCALER_X_PATH),
        'scaler_y': joblib.load(SCALER_Y_PATH),
        'metadata': joblib.load(METADATA_PATH)
    }

def prepare_lstm_input(df_latest, sku, metadata, scaler_x, scaler_y, lt_category_pred):
    """สร้าง Input Sequence สำหรับ LSTM จากข้อมูล Snapshot"""
    time_steps = metadata['time_steps']
    features = metadata['valid_features_lstm']
    
    row = df_latest[df_latest['SKU'] == sku].iloc[0]
    current_usage = row.get('Usage_Qty', 0)
    
    seq_data = []
    for i in range(time_steps):
        month_record = {}
        for f in features:
            # ใส่ค่า Feature ต่างๆ ลงใน Sequence
            if f == 'LT_CATEGORY_PRED':
                month_record[f] = lt_category_pred # ใช้ค่าที่ XGBoost ทำนายได้
            elif f == 'Safety_Stock_Qty':
                month_record[f] = row.get('Safety_Stock_Qty', 0) # Min Stock
            elif f == 'Total_SKU_Usage':
                 month_record[f] = row.get('Total_SKU_Usage', 0)
            elif f == 'Patient_Count':
                month_record[f] = row.get('Patient_Count', 1500)
            elif f in ['Patient_E', 'Patient_I', 'Patient_O']:
                month_record[f] = row.get(f, 0) # Patient Breakdown
            elif f in ['month_num', 'year_num', 'day_index']:
                month_record[f] = 0 # Placeholder
            else:
                month_record[f] = row.get(f, 0)
        
        month_record['Usage_Qty'] = current_usage # สมมติว่าประวัติคงที่ (ใน Snapshot เราไม่มี History)
        seq_data.append(month_record)
        
    df_seq = pd.DataFrame(seq_data)
    df_seq = df_seq[features + ['Usage_Qty']] # Reorder
    
    x_vals = df_seq[features].values
    y_vals = df_seq[['Usage_Qty']].values
    
    x_scaled = scaler_x.transform(x_vals)
    y_scaled = scaler_y.transform(y_vals)
    
    return np.array([np.hstack([x_scaled, y_scaled])])

def predict_inventory_usage(system_artifacts: Dict[str, Any], file_content: bytes, forecast_days: int) -> Dict[str, Any]:
    
    lstm_model = system_artifacts['lstm_model']
    xgb_classifier = system_artifacts['xgb_classifier']
    scaler_x = system_artifacts['scaler_x']
    scaler_y = system_artifacts['scaler_y']
    metadata = system_artifacts['metadata']
    
    # 1. Load Snapshot Input
    df_latest = pd.read_excel(BytesIO(file_content), sheet_name=0) 
    df_latest.columns = df_latest.columns.str.strip()
    
    # Map Columns (ให้ตรงกับ Training)
    df_latest.rename(columns={
        'Next_Patient_Count_M1': 'Patient_Count', 
        'Latest_Usage_Qty': 'Usage_Qty',
        'Min_Stock': 'Safety_Stock_Qty', 
        # เพิ่ม Mapping Patient Breakdown จาก Input (ถ้ามี)
        'Next_Patient_E_M1': 'Patient_E',
        'Next_Patient_I_M1': 'Patient_I',
        'Next_Patient_O_M1': 'Patient_O',
    }, inplace=True)
    
    # Standardize Unit
    if 'Conversion_Factor' in df_latest.columns:
        df_latest['Usage_Qty'] = df_latest['Usage_Qty'] * df_latest['Conversion_Factor'].fillna(1)

    # --- STEP 1: CLASSIFY LEAD TIME RISK (XGBoost) ---
    class_features = metadata.get('class_features', ['Unit_Cost', 'Safety_Stock_Qty'])
    # Ensure features exist
    for f in class_features:
        if f not in df_latest.columns: df_latest[f] = 0
            
    X_class_input = df_latest[class_features].fillna(0)
    df_latest['LT_CATEGORY_PRED'] = xgb_classifier.predict(X_class_input)
    
    # --- STEP 2: FORECASTING (LSTM) ---
    forecast_result_rows = []
    all_actions = []
    sku_policy_map = metadata.get('sku_policy_map', {})
    
    try:
        start_date = metadata['max_date'] + pd.DateOffset(months=1)
    except:
        start_date = datetime.datetime.now()
    
    for index, row in df_latest.iterrows():
        sku = row['SKU']
        lt_cat = row['LT_CATEGORY_PRED']
        
        # Forecast
        try:
            input_seq = prepare_lstm_input(df_latest, sku, metadata, scaler_x, scaler_y, lt_cat)
            scaled_pred = lstm_model.predict(input_seq, verbose=0)
            actual_pred = scaler_y.inverse_transform(scaled_pred)[0][0]
            predicted_usage = max(0, round(actual_pred))
        except:
            predicted_usage = row.get('Usage_Qty', 0) # Fallback

        # Generate Output Rows
        for i in range(forecast_days):
            forecast_result_rows.append({
                'date': start_date + pd.DateOffset(months=i),
                'predicted_usage': int(predicted_usage),
                'SKU': sku
            })
            
        # --- STEP 3: RECOMMENDATION (Rule-Based) ---
        # 1. ดึง Policy (Min/Max Stock)
        policy = sku_policy_map.get(sku, {})
        Max_Stock = row.get('Max_Stock', policy.get('Max_Stock', 0)) # Input > Training
        Min_Stock = row.get('Safety_Stock_Qty', policy.get('Safety_Stock_Qty', 0))
        
        LT_Days = row.get('Lead_Time_Days', 14)
        SOH = row.get('Stock_on_Hand_Qty', 0)
        UC = row.get('Unit_Cost', 0)
        
        # 2. Calculate ROP
        avg_daily_forecast = predicted_usage / 30
        rop_threshold = (avg_daily_forecast * LT_Days) + Min_Stock
        
        # 3. Determine Priority & Qty
        stock_out_date = None
        if SOH < predicted_usage: # Risk of stockout this month
             days = (SOH / avg_daily_forecast) if avg_daily_forecast > 0 else 30
             stock_out_date = (start_date + timedelta(days=int(days))).strftime('%Y-%m-%d')
             
        is_high_risk = (SOH < rop_threshold)
        reorder_qty = 0
        priority_group = 'Low Priority'

        if is_high_risk:
            priority_group = 'High Priority'
            # เติมของให้เต็ม Max (ถ้ามี) หรือตาม Forecast + Min
            target_level = predicted_usage + Min_Stock
            if Max_Stock > 0: 
                target_level = min(target_level, Max_Stock) # ห้ามเกิน Max Stock
            
            reorder_qty = max(0, target_level - SOH)
            
        elif SOH < (rop_threshold * 1.5):
            priority_group = 'Medium Priority'
            target_level = rop_threshold * 1.5
            if Max_Stock > 0: target_level = min(target_level, Max_Stock)
            reorder_qty = max(0, target_level - SOH)
            
        reorder_cost = reorder_qty * UC
        
        all_actions.append({
            'sku': sku,
            'priority': priority_group,
            'stock_out_date': stock_out_date if stock_out_date else 'N/A',
            'recommended_qty': int(reorder_qty),
            'reorder_cost': round(reorder_cost, 2),
            'current_soh': int(SOH),
            'rop_threshold': int(rop_threshold),
            'max_stock_policy': int(Max_Stock)
        })

    return {
        "forecast": forecast_result_rows,
        "metrics": {
            'total_skus': len(df_latest),
            'high_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'High Priority'],
            'medium_priority_items': [i['sku'] for i in all_actions if i['priority'] == 'Medium Priority'],
            'reorder_cost_total': round(sum(i['reorder_cost'] for i in all_actions), 2),
            'action_items': all_actions,
        }
    }