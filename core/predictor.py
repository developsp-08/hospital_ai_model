import pandas as pd
from joblib import load
import numpy as np
import os
from datetime import timedelta
import datetime
from io import BytesIO
from typing import Dict, Any, List
import warnings

# ปิดคำเตือน (warnings)
warnings.filterwarnings('ignore')

MODEL_PATH = 'models/inventory_model.pkl'

def load_model_metadata(path: str = MODEL_PATH) -> Dict[str, Any]:
    """โหลดโมเดลและ Metadata"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}.")
    metadata = load(path)
    return metadata

def get_start_date_from_file(metadata: Dict) -> datetime.datetime:
    """
    กำหนดวันเริ่มต้นทำนายโดยใช้ Max Date จากข้อมูล Training 
    """
    
    try:
        max_training_date = metadata['max_date']
        
        # วันเริ่มต้นทำนายคือวันแรกของเดือนถัดไปจาก Max Training Date
        start_date = max_training_date + pd.DateOffset(months=1)
        return start_date.replace(day=1)

    except Exception as e:
        # หากเกิดข้อผิดพลาดในการอ่าน metadata (เช่น max_date ไม่มี)
        raise Exception(f"Failed to infer start date from model metadata: {e}.")


def calculate_inventory_actions(df_latest_data: pd.DataFrame, usage_forecast: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """คำนวณ Stock-out Risk, ROP, และจัดลำดับความสำคัญ (Rule-Based Logic)"""
    
    total_skus = len(metadata['item_list'])
    all_actions = []

    # 1. GroupBy เพื่อหาค่าเฉลี่ย/ค่าล่าสุดของ Inventory Params ใน Input Snapshot
    sku_groups = df_latest_data.groupby('SKU').agg({
        'Stock_on_Hand_Qty': 'mean',
        'Lead_Time_Days': 'mean',
        'Safety_Stock_Qty': 'mean',
        'Unit_Cost': 'mean',
        'Latest_Usage_Qty': 'mean' 
    }).reset_index()
    
    # 2. คำนวณค่าเฉลี่ยการใช้งานต่อวัน (Avg_Daily_Usage) จากข้อมูล Training ทั้งหมด
    
    # *** สมมติฐาน: ไฟล์ Training_Data_Final.xlsx ต้องอยู่ใน path 'data/' ***
    try:
        df_train = pd.read_excel('data/Training_Data_Final.xlsx')
        df_train['Usage_Qty'] = pd.to_numeric(df_train['Usage_Qty'], errors='coerce')
        df_train = df_train.dropna(subset=['Date', 'Usage_Qty'])
        total_days_trained = (metadata['max_date'] - metadata['base_date']).days + 1
        avg_daily_usage_all_data = df_train.groupby('SKU')['Usage_Qty'].sum() / total_days_trained
    except Exception:
        # ใช้ค่า Default หากไม่พบไฟล์ Training หรือมีข้อผิดพลาด
        avg_daily_usage_all_data = pd.Series(1.0, index=metadata['item_list'])


    # 3. คำนวณ ROP, Stock-out Date และ Priority
    for index, sku_row in sku_groups.iterrows():
        sku = sku_row['SKU']
        
        # กรองผลทำนายสำหรับ SKU นี้
        forecast_sku = usage_forecast[usage_forecast['SKU'] == sku].copy()

        # ดึง Inventory Parameters
        LT_Days = sku_row.get('Lead_Time_Days', 14) 
        SS_Qty = sku_row.get('Safety_Stock_Qty', 50)
        SOH = sku_row.get('Stock_on_Hand_Qty', 200) 
        UC = sku_row.get('Unit_Cost', 10)
        
        # 3.1 คำนวณ ROP Threshold (ใช้ค่าเฉลี่ยการใช้งานรายวันจากข้อมูล Training ทั้งหมด)
        avg_daily_usage = avg_daily_usage_all_data.get(sku, 1)
        rop_threshold = (avg_daily_usage * LT_Days) + SS_Qty
        rop_threshold = max(3, rop_threshold) # กำหนดค่าขั้นต่ำเป็น 3 

        # 3.2 หาเดือนที่ Stock จะหมด (ใช้ Predicted Usage)
        stock_current = SOH
        stock_out_date = None

        for _, row in forecast_sku.iterrows():
            # ใช้จำนวนวันจริงในเดือนนั้นๆ ในการคำนวณการใช้ (เพื่อความแม่นยำ)
            # แต่ในกรณีนี้ Predicted Usage เป็นยอดรวมต่อเดือนแล้ว
            monthly_predicted_usage = row['predicted_usage']
            
            if stock_current >= monthly_predicted_usage:
                stock_current -= monthly_predicted_usage
            else:
                # คำนวณวันที่ขาดแคลน
                remaining_stock = stock_current
                # สมมติการใช้ต่อวันในเดือนนี้เท่ากับ Predicted Usage / จำนวนวันในเดือน
                days_in_month = pd.Period(row['date'], freq='M').days_in_month
                daily_usage = monthly_predicted_usage / days_in_month if monthly_predicted_usage > 0 else 0.1
                
                if daily_usage > 0:
                    days_to_stock_out = (remaining_stock / daily_usage)
                    stock_out_date = (row['date'] + timedelta(days=days_to_stock_out)).strftime('%Y-%m-%d')
                else:
                    stock_out_date = 'N/A' # ไม่มีวันหมดเพราะไม่ถูกใช้
                    
                stock_current = 0
                break

        # 3.3 จัดกลุ่ม Priority และคำนวณ Reorder Qty (Rule-Based Logic)
        is_high_risk = (SOH < rop_threshold) and (stock_out_date is not None) and (stock_out_date != 'N/A')
        
        if is_high_risk:
            priority_group = 'High Priority'
            # สั่งซื้อเพื่อให้มีสต็อกเพียงพอไปอีก Lead Time 
            # และบวก Safety Stock (ROP)
            reorder_qty = rop_threshold 
        elif SOH < (rop_threshold * 1.5):
            priority_group = 'Medium Priority'
            # สั่งซื้อเพื่อให้มีสต็อกเหลือเฟือเล็กน้อย
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
    df_latest.columns = df_latest.columns.str.strip()
    
    # 2. กำหนด Start Date 
    start_date = get_start_date_from_file(metadata)
    
    # 3. สร้าง Input Features สำหรับ XGBoost
    base_date = metadata['base_date']
    features = metadata['features']
    sku_list = metadata['item_list']
    
    # ดึงค่า Contextual Features ล่าสุดจากไฟล์ที่อัปโหลด
    last_known_context = df_latest.iloc[0].to_dict()
    
    # สร้างตารางทำนาย
    df_predict_dates = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=forecast_days, freq='MS')})
    
    # --- FIX: แก้ปัญหา KeyError สำหรับ Patient_Count/Total_SKU_Usage ---
    try:
        avg_patient_count = df_latest['Patient_Count'].mean()
    except KeyError:
        # ใช้ค่า Default หากไม่พบคอลัมน์ Patient_Count
        avg_patient_count = 1500 
    
    try:
        avg_total_usage = df_latest['Total_SKU_Usage'].mean()
    except KeyError:
        # ใช้ค่า Default หากไม่พบคอลัมน์ Total_SKU_Usage
        avg_total_usage = 1500 

    # 4. สร้าง DataFrame สำหรับทำนาย (Expansion)
    df_predict = pd.DataFrame({
        'Date': np.repeat(df_predict_dates['Date'], len(sku_list)),
        'SKU': np.tile(sku_list, len(df_predict_dates)),
    }).sort_values(by='Date').reset_index(drop=True)
    
    # 5. Map Time Features และ Contextual Features (แบบคงที่/เฉลี่ย)
    df_predict['day_index'] = (df_predict['Date'] - base_date).dt.days
    df_predict['month_num'] = df_predict['Date'].dt.month
    df_predict['year_num'] = df_predict['Date'].dt.year
    
    # Contextual Features ที่ใช้ในการทำนาย จะใช้ค่าเฉลี่ยของข้อมูลล่าสุดที่ได้รับมา
    df_predict['Patient_Count'] = avg_patient_count
    df_predict['Total_SKU_Usage'] = avg_total_usage
    
    # 6. Iterative Prediction (การทำนายแบบวนซ้ำเพื่อให้ Lag1 เป็น Dynamic)
    df_predict['predicted_usage'] = 0 # สร้างคอลัมน์เปล่า
    
    for i in range(len(df_predict_dates)):
        current_date = df_predict_dates.iloc[i]['Date']
        df_current_month = df_predict[df_predict['Date'] == current_date].copy()
        
        # 6.1. กำหนด Lag1:
        if i == 0:
            # เดือนแรก: ใช้ค่า Latest_Usage_Qty จาก Snapshot
            try:
                lag_values = df_latest.groupby('SKU')['Latest_Usage_Qty'].mean().to_dict()
                df_current_month['lag1'] = df_current_month['SKU'].map(lag_values).fillna(30)
            except KeyError:
                # FIX: หากคอลัมน์ Latest_Usage_Qty หายไป ให้ใช้ค่า Default 30
                df_current_month['lag1'] = 30
        else:
            # เดือนถัดไป: ใช้ผลทำนายของเดือนก่อน (i-1) เป็น Lag1
            df_prev_month = df_predict[df_predict['Date'] == df_predict_dates.iloc[i-1]['Date']]
            lag_values = df_prev_month.set_index('SKU')['predicted_usage'].to_dict()
            df_current_month['lag1'] = df_current_month['SKU'].map(lag_values).fillna(30)
            
        
        # 6.2. ทำนายผล:
        X_predict = df_current_month[features]
        predictions = metadata['model'].predict(X_predict)
        
        # 6.3. บันทึกผลทำนายกลับเข้าไปใน DataFrame หลัก
        df_predict.loc[df_predict['Date'] == current_date, 'predicted_usage'] = np.maximum(0, predictions).round().astype(int)

    
    # 7. จัดรูปแบบผลลัพธ์
    df_forecast_result = df_predict[['Date', 'predicted_usage', 'SKU']].rename(columns={'Date': 'date'})
    
    # 8. คำนวณ Actions และ Priority
    action_metrics = calculate_inventory_actions(df_latest, df_forecast_result, metadata)
    
    return {
        "forecast": df_forecast_result.to_dict('records'),
        "metrics": action_metrics
    }