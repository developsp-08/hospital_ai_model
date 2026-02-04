import sys
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks
from io import BytesIO
from datetime import datetime
import pandas as pd
import json
import numpy as np # จำเป็นสำหรับ Custom JSON Encoder
from core.retrainer import run_retrain_process

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.predictor import load_model_system, predict_inventory_usage, get_reference_data 
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INVENTORY_SYSTEM: Dict[str, Any] = {}
DATA_DIR_PATH = os.path.join(parent_dir, 'data') # Path สำหรับเก็บ Cache

# 🆕 Custom Encoder เพื่อให้ save numpy data ลง json ได้ไม่ error
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_model_metadata() -> Dict[str, Any]:
    global INVENTORY_SYSTEM
    
    if not INVENTORY_SYSTEM: 
        try:
            INVENTORY_SYSTEM = load_model_system()
            logger.info("Hybrid AI System loaded.")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            INVENTORY_SYSTEM = {} 
    return INVENTORY_SYSTEM


def manual_column_mapping(df_columns):
    """จับคู่ชื่อหัวตารางจาก User เข้ากับมาตรฐานของระบบ (Hard-coded)"""
    mapping_dict = {
        'รหัสสินค้า': 'SKU', 
        'SKU ID': 'SKU', 
        'Item No': 'SKU', 
        'ชื่อรายการ': 'Item_Name', 
        'ชื่อสินค้า': 'Item_Name',
        'จำนวนเบิก': 'Usage_Qty', 
        'เบิกจ่าย': 'Usage_Qty', 
        'Latest_Usage_Qty': 'Usage_Qty',
        'จำนวนคนไข้': 'Visit_Campus', 
        'Current_Patient_Count': 'Visit_Campus',
        'Lead_Time_Days': 'Lead_Time_Days',
        'Unit_Cost': 'Unit_Cost',
        'Min_Stock': 'Min_Stock',
        'Max_Stock': 'Max_Stock',
        'Conversion_Factor': 'Conversion_Factor'
    }
    
    return {col: mapping_dict[col] for col in df_columns if col in mapping_dict}

app = FastAPI(title="Hybrid Inventory AI", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#ENDPOINT 
@app.post("/predict")
async def predict_inventory_from_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    forecast_days: int = Form(7), 
    metadata: Dict[str, Any] = Depends(get_model_metadata)
):
    if not metadata:
        raise HTTPException(status_code=503, detail="AI Model not ready.")
    
    file_content = await file.read()
    
    try:
        last_train_metrics = metadata.get('last_metrics', {
            "accuracy": 0, "mae": 0, "variance": 0
        })

        df_uploaded = pd.read_excel(BytesIO(file_content))
        
        current_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        df_uploaded['Date'] = current_date
        
        mapping = manual_column_mapping(df_uploaded.columns.tolist())
        df_mapped = df_uploaded.rename(columns=mapping)

        # Update Master File
        from core.data_master import update_master_file
        is_updated = update_master_file(df_mapped)
        
        if is_updated:
            logger.info("🚀 ตรวจพบข้อมูลใหม่...")
            
            # --- Logic เตรียมข้อมูล ---
            if 'Current_Patient_Count' in df_uploaded.columns:
                total_vists = pd.to_numeric(df_uploaded['Current_Patient_Count'], errors='coerce').fillna(0)
                df_uploaded['Visit_Campus'] = total_vists
                df_uploaded['Patient_E'] = (total_vists * 0.15).astype(int)
                df_uploaded['Patient_I'] = (total_vists * 0.25).astype(int)
                df_uploaded['Patient_O'] = (total_vists * 0.60).astype(int)

            if 'Latest_Usage_Qty' in df_uploaded.columns:
                df_uploaded['Usage_Qty'] = df_uploaded['Latest_Usage_Qty']
            if 'Conversion_Factor' in df_uploaded.columns:
                df_uploaded['Usage_Qty'] = pd.to_numeric(df_uploaded['Usage_Qty'], errors='coerce') * \
                                          pd.to_numeric(df_uploaded['Conversion_Factor'], errors='coerce').fillna(1)
            
            # ⚡ CLEAR CACHE: ข้อมูลเปลี่ยนแล้ว ต้องลบ Cache เก่าทิ้ง เพื่อให้ Initial Load ครั้งหน้าคำนวณใหม่
            try:
                for f in os.listdir(DATA_DIR_PATH):
                    if f.startswith("dashboard_cache_"):
                        os.remove(os.path.join(DATA_DIR_PATH, f))
                logger.info("🧹 Cleared old dashboard cache.")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")

            # ⚡ Background Retrain
            logger.info("⏳ Scheduling background retraining task...")
            background_tasks.add_task(run_retrain_process) 
            
            logger.info("ℹ️ Using current model while retraining runs in background.")

        else:
            logger.info(f"ℹ️ ไฟล์เดิม เปลี่ยน forecast_days เป็น {forecast_days}: ข้ามขั้นตอนเทรน")

        # --- step 5: predict ---
        results = predict_inventory_usage(metadata, file_content, forecast_days, is_upload=True)
        
        return {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            "Monthly_Time_Series_Data": results['Monthly_Chart_Data'], 
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            },
            "Message": "Data uploaded successfully. Model retraining started in background." if is_updated else "Prediction updated."
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# ENDPOINT 
@app.get("/initial_forecast")
async def initial_forecast(
    forecast_days: int = Query(7, ge=1), 
    metadata: Dict[str, Any] = Depends(get_model_metadata)
):
    if not metadata:
        raise HTTPException(status_code=503, detail="AI Model not ready.")
        
    # 🆕 CACHE KEY: แยกไฟล์ Cache ตามจำนวนวันที่ Forecast
    CACHE_FILE = os.path.join(DATA_DIR_PATH, f'dashboard_cache_{forecast_days}.json')

    # 1. ลองอ่านจาก Cache ก่อน (Fast Path 🚀)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.info(f"✅ Loaded initial data from Cache ({forecast_days} days)")
            return cached_data
        except Exception as e:
            logger.warning(f"Cache read error (will re-compute): {e}")

    # 2. ถ้าไม่มี Cache ต้องคำนวณใหม่ (Slow Path)
    try:
        logger.info("Computing initial forecast (No Cache found)...")
        file_content = get_reference_data() 
        
        results = predict_inventory_usage(metadata, file_content, forecast_days, is_upload=False)
        
        response_data = {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            "Monthly_Time_Series_Data": results['Monthly_Chart_Data'], 
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            }
        }

        # 🆕 บันทึก Cache ไว้ใช้รอบหน้า
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, cls=NpEncoder, ensure_ascii=False)
            logger.info("💾 Saved new dashboard cache.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return response_data

    except FileNotFoundError as e:
        logger.error(f"Initial Load Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: Reference Excel file not found. Please ensure exists.")
    except Exception as e:
        logger.error(f"Prediction failed during initial load: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")