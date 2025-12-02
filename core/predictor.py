import pandas as pd
from joblib import load
import numpy as np
import os
from datetime import timedelta
import datetime
from io import BytesIO
from typing import Dict, Any, List
import warnings
from tensorflow.keras.models import load_model
# FIX: Import mean_squared_error จาก keras.losses เพื่อแก้ปัญหา ImportError ใน Uvicorn
from keras.losses import mean_squared_error 

warnings.filterwarnings('ignore')

MODEL_PATH = 'models/inventory_model_lstm.pkl'
TRAINING_DATA_PATH = 'data/Training_Data_Final.xlsx'

def load_model_metadata(path: str = MODEL_PATH) -> Dict[str, Any]:
    """โหลดโมเดลและ Metadata รวมถึง Keras Model Weights"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}.")
    metadata = load(path)
    
    # โหลด Keras Model Weights
    try:
        # FIX: ระบุ custom_objects เพื่อให้ Keras รู้จัก 'mse'
        metadata['model'] = load_model(
            metadata['model_path'],
            custom_objects={'mse': mean_squared_error}
        )
    except Exception as e:
        # หากยัง Error ให้แสดง Error ที่เกี่ยวข้องกับการโหลดโมเดลออกมา
        raise Exception(f"Failed to load Keras model weights from {metadata['model_path']}: {e}")
        
    return metadata

def get_start_date_from_file(metadata: Dict) -> datetime.datetime:
    """กำหนดวันเริ่มต้นทำนายเป็นวันแรกของเดือนถัดไปจาก Max Training Date"""
    try:
        max_training_date = metadata['max_date']
        start_date = max_training_date + pd.DateOffset(months=1)
        return start_date.replace(day=1)
    except Exception as e:
        raise Exception(f"Failed to infer start date from model metadata: {e}.")


def calculate_inventory_actions(df_latest_data: pd.DataFrame, usage_forecast: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """คำนวณ Stock-out Risk, ROP, และจัดลำดับความสำคัญ (Rule-Based Logic)"""
    
    total_skus = len(metadata['item_list'])
    all_actions = []

    # 1. GroupBy เพื่อหาค่าเฉลี่ย/ค่าล่าสุดของ Inventory Params
    df_latest_data.columns = df_latest_data.columns.str.strip().str.replace(' ', '_')
    sku_groups = df_latest_data.groupby('SKU').agg({
        'Stock_on_Hand_Qty': 'mean',
        'Lead_Time_Days': 'mean',
        'Safety_Stock_Qty': 'mean',
        'Unit_Cost': 'mean',
        'Latest_Usage_Qty': 'mean' 
    }).reset_index()
    
    # 2. คำนวณค่าเฉลี่ยการใช้งานต่อวัน (Avg_Daily_Usage) จากข้อมูล Training ทั้งหมด
    try:
        df_train = pd.read_excel(TRAINING_DATA_PATH)
        df_train.columns = df_train.columns.str.strip().str.replace(' ', '_')
        df_train['Usage_Qty'] = pd.to_numeric(df_train['Usage_Qty'], errors='coerce')
        df_train = df_train.dropna(subset=['Date', 'Usage_Qty'])
        total_days_trained = (metadata['max_date'] - metadata['base_date']).days + 1
        avg_daily_usage_all_data = df_train.groupby('SKU')['Usage_Qty'].sum() / total_days_trained
    except Exception:
        avg_daily_usage_all_data = pd.Series(1.0, index=metadata['item_list'])


    # 3. คำนวณ ROP, Stock-out Date และ Priority
    for index, sku_row in sku_groups.iterrows():
        sku = sku_row['SKU']
        forecast_sku = usage_forecast[usage_forecast['SKU'] == sku].copy()

        LT_Days = sku_row.get('Lead_Time_Days', 14) 
        SS_Qty = sku_row.get('Safety_Stock_Qty', 50)
        SOH = sku_row.get('Stock_on_Hand_Qty', 200) 
        UC = sku_row.get('Unit_Cost', 10)
        
        # 3.1 คำนวณ ROP Threshold
        avg_daily_usage = avg_daily_usage_all_data.get(sku, 1)
        rop_threshold = (avg_daily_usage * LT_Days) + SS_Qty
        rop_threshold = max(3, rop_threshold)

        # 3.2 หาเดือนที่ Stock จะหมด
        stock_current = SOH
        stock_out_date = None

        for _, row in forecast_sku.iterrows():
            monthly_predicted_usage = row['predicted_usage']
            
            if stock_current >= monthly_predicted_usage:
                stock_current -= monthly_predicted_usage
            else:
                remaining_stock = stock_current
                # ใช้ pd.Period เพื่อหาจำนวนวันในเดือน
                days_in_month = pd.Period(row['date'], freq='M').days_in_month
                daily_usage = monthly_predicted_usage / days_in_month if monthly_predicted_usage > 0 else 0.1
                
                if daily_usage > 0:
                    days_to_stock_out = (remaining_stock / daily_usage)
                    stock_out_date = (row['date'] + timedelta(days=days_to_stock_out)).strftime('%Y-%m-%d')
                else:
                    stock_out_date = 'N/A' 
                    
                stock_current = 0
                break

        # 3.3 จัดกลุ่ม Priority และคำนวณ Reorder Qty
        is_high_risk = (SOH < rop_threshold) and (stock_out_date is not None) and (stock_out_date != 'N/A')
        
        if is_high_risk:
            priority_group = 'High Priority'
            reorder_qty = rop_threshold 
        elif SOH < (rop_threshold * 1.5):
            priority_group = 'Medium Priority'
            reorder_qty = int(np.round(rop_threshold * 1.5))
        else:
            priority_group = 'Low Priority'
            reorder_qty = 0
            
        reorder_cost = max(0, reorder_qty * UC)

        all_actions.append({
            'sku': sku,
            'priority': priority_group,
            'stock_out_date': stock_out_date if stock_out_date else 'N/A',
            'recommended_qty': int(np.round(reorder_qty)),
            'reorder_cost': round(reorder_cost, 2),
            'current_soh': int(SOH),
            'rop_threshold': int(np.round(rop_threshold))
        })

    # 4. สรุปผลลัพธ์
    return {
        'total_skus': total_skus,
        'high_priority_items': [item['sku'] for item in all_actions if item['priority'] == 'High Priority'],
        'medium_priority_items': [item['sku'] for item in all_actions if item['priority'] == 'Medium Priority'],
        'reorder_cost_total': round(sum(item['reorder_cost'] for item in all_actions), 2),
        'action_items': all_actions,
    }


def predict_inventory_usage(
    metadata: Dict[str, Any], 
    file_content: bytes,
    forecast_days: int
) -> Dict[str, Any]:
    
    # 1. โหลดไฟล์ล่าสุด (Snapshot Input) และทำความสะอาด
    df_latest = pd.read_excel(BytesIO(file_content), sheet_name=0) 
    df_latest.columns = df_latest.columns.str.strip().str.replace(' ', '_')
    
    # 2. กำหนด Start Date และ Metadata Components
    start_date = get_start_date_from_file(metadata)
    base_date = metadata['base_date']
    features = metadata['features']
    sku_list = metadata['item_list']
    timesteps = metadata['timesteps']
    scaler = metadata['scaler']
    lstm_model = metadata['model'] # Keras Model Object

    # 3. เตรียมข้อมูลย้อนหลังสำหรับ Lag (History Window)
    df_latest['Date'] = pd.to_datetime(df_latest['Date'], errors='coerce')
    df_latest = df_latest.dropna(subset=['Date'])
    df_latest = df_latest.sort_values(by='Date').reset_index(drop=True)
    
    # ต้องสร้าง Visit_Campus_Num และ Time Features
    try:
        df_latest['Visit_Campus_Num'] = df_latest['Visit_Campus'].astype('category').cat.codes
    except KeyError:
        df_latest['Visit_Campus_Num'] = 0
        
    df_latest['day_index'] = (df_latest['Date'] - base_date).dt.days
    df_latest['month_num'] = df_latest['Date'].dt.month
    df_latest['year_num'] = df_latest['Date'].dt.year

    # 3.1 ดึง Contextual Features ล่าสุดจาก History Window
    # (ใช้ค่าเฉลี่ยของข้อมูลล่าสุดที่ได้รับมาเพื่อการทำนาย)
    last_context = df_latest.iloc[-1].to_dict()
    
    try:
        avg_patient_e = df_latest['Patient_E'].mean()
        avg_patient_i = df_latest['Patient_I'].mean()
        avg_patient_o = df_latest['Patient_O'].mean()
    except KeyError:
        # Default ค่าถ้าคอลัมน์หาย
        avg_patient_e, avg_patient_i, avg_patient_o = 500, 500, 500
        
    try:
        avg_total_usage = df_latest['Total_SKU_Usage'].mean()
    except KeyError:
        avg_total_usage = 1500 
    
    try:
        avg_visit_campus_num = df_latest['Visit_Campus_Num'].mean()
    except KeyError:
        avg_visit_campus_num = 0

    # 4. Iterative Prediction (การทำนายแบบวนซ้ำ)
    df_forecast_list = []
    
    for sku in sku_list:
        
        # 4.1. เตรียม History Window (ย้อนหลัง TIMESTEPS)
        df_history = df_latest[df_latest['SKU'] == sku].tail(timesteps).copy()
        
        if len(df_history) < timesteps:
            warnings.warn(f"SKU {sku} has insufficient history for LSTM (need {timesteps}, found {len(df_history)}). Skipping prediction for this SKU.")
            continue 

        # 4.2. เตรียมตารางทำนาย
        df_predict_dates = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=forecast_days, freq='MS')})
        df_predict_temp = df_predict_dates.copy()
        df_predict_temp['SKU'] = sku
        
        # Map Contextual Features สำหรับอนาคต (ใช้ค่าเฉลี่ย)
        df_predict_temp['Visit_Campus_Num'] = avg_visit_campus_num
        df_predict_temp['Patient_E'] = avg_patient_e
        df_predict_temp['Patient_I'] = avg_patient_i
        df_predict_temp['Patient_O'] = avg_patient_o
        df_predict_temp['Total_SKU_Usage'] = avg_total_usage 
        
        df_predict_temp['Usage_Qty'] = 0.0

        current_window = df_history[features].copy()
        
        # 4.3. Loop ทำนายทีละเดือน
        for i in range(forecast_days):
            # i) เตรียม Input Window (3D Tensor)
            X_input_scaled = scaler.transform(current_window.values)
            X_input_reshaped = X_input_scaled[np.newaxis, :, :] # (1, TIMESTEPS, FEATURES)

            # ii) ทำนาย
            prediction_scaled = lstm_model.predict(X_input_reshaped, verbose=0)[0][0]
            
            # iii) Inverse Transform เพื่อให้ได้ยอดใช้จริง
            dummy_row = np.zeros(len(features))
            dummy_row[features.index('Usage_Qty')] = prediction_scaled
            
            prediction_actual = scaler.inverse_transform(dummy_row.reshape(1, -1))[0][features.index('Usage_Qty')]
            predicted_usage = max(0, int(np.round(prediction_actual)))
            
            # iv) บันทึกผลทำนายใน DataFrame ชั่วคราว
            df_predict_temp.loc[i, 'Usage_Qty'] = predicted_usage
            
            # v) เตรียม Window ใหม่สำหรับรอบถัดไป (Shifting Window)
            next_row_context = df_predict_temp.loc[i, features].values
            next_row_context[features.index('Usage_Qty')] = predicted_usage
            
            current_window = pd.DataFrame(
                np.vstack([current_window.iloc[1:], next_row_context]),
                columns=features
            )

        # 4.4. บันทึกผลทำนาย
        df_forecast_list.append(df_predict_temp[['Date', 'SKU', 'Usage_Qty']].rename(columns={'Usage_Qty': 'predicted_usage'}))

    df_forecast_result = pd.concat(df_forecast_list).reset_index(drop=True)
    
    # 5. คำนวณ Actions และ Priority
    action_metrics = calculate_inventory_actions(df_latest, df_forecast_result, metadata)
    
    return {
        "forecast": df_forecast_result.to_dict('records'),
        "metrics": action_metrics
    }