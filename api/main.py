import sys
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from io import BytesIO
from datetime import datetime
import pandas as pd
from core.retrainer import run_retrain_process

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.predictor import load_model_system, predict_inventory_usage, get_reference_data 
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any
# import entrypoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INVENTORY_SYSTEM: Dict[str, Any] = {}

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

# origins = ["http://localhost:3000", "http://127.0.0.1:3000","http://localhost:5173","http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# async def startup_event():
#     entrypoint.prepare_environment()

#ENDPOINT 
@app.post("/predict")
async def predict_inventory_from_file(
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

        
        from core.data_master import update_master_file
        is_updated = update_master_file(df_mapped)
        
        
        if is_updated:
            logger.info("🚀 ตรวจพบข้อมูลใหม่ เริ่มกระบวนการประมวลผลและ Re-train...")
            
            # --- Step 2: Patient Breakdown ---
            # current_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # df_uploaded['Date'] = current_date
            
            if 'Current_Patient_Count' in df_uploaded.columns:
                total_vists = pd.to_numeric(df_uploaded['Current_Patient_Count'], errors='coerce').fillna(0)
                df_uploaded['Visit_Campus'] = total_vists
                ratio_E, ratio_I, ratio_O = 0.15, 0.25, 0.60
                df_uploaded['Patient_E'] = (total_vists * ratio_E).astype(int)
                df_uploaded['Patient_I'] = (total_vists * ratio_I).astype(int)
                df_uploaded['Patient_O'] = (total_vists * ratio_O).astype(int)

            # --- Step 3: Conversion Factor ---
            if 'Latest_Usage_Qty' in df_uploaded.columns:
                df_uploaded['Usage_Qty'] = df_uploaded['Latest_Usage_Qty']
            if 'Conversion_Factor' in df_uploaded.columns:
                df_uploaded['Usage_Qty'] = pd.to_numeric(df_uploaded['Usage_Qty'], errors='coerce') * \
                                          pd.to_numeric(df_uploaded['Conversion_Factor'], errors='coerce').fillna(1)

            # Re-train
            training_result = run_retrain_process()
            
            if training_result and isinstance(training_result, dict) and training_result.get("success"):
                global INVENTORY_SYSTEM
                INVENTORY_SYSTEM = load_model_system()
                metadata = INVENTORY_SYSTEM 
                
                last_train_metrics["accuracy"] = training_result.get("accuracy", 0)
                last_train_metrics["mae"] = training_result.get("mae", 0)
                last_train_metrics["variance"] = training_result.get("variance", 0)
                logger.info(" Re-train และโหลดโมเดลใหม่สำเร็จ")
        else:
            # change forecast_days
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
            # "Training_Metrics": last_train_metrics
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
        
    try:
        
        file_content = get_reference_data() 
        
        
        results = predict_inventory_usage(metadata, file_content, forecast_days, is_upload=False)
        
        
        train_metrics = metadata.get('last_metrics', {
            "accuracy": "0", 
            "mae": "0", 
            "variance": "0"
        })

    
        return {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            
            "Monthly_Time_Series_Data": results['Monthly_Chart_Data'], 
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            },
            # "Training_Metrics": train_metrics
        }
    except FileNotFoundError as e:
        logger.error(f"Initial Load Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: Reference Excel file not found. Please ensure exists.")
    except Exception as e:
        logger.error(f"Prediction failed during initial load: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
   
    
    
    