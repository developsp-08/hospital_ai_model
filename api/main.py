import sys
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.predictor import load_model_system, predict_inventory_usage, get_reference_data 
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any
import entrypoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ใช้ global variable เพื่อเก็บ model/metadata
INVENTORY_SYSTEM: Dict[str, Any] = {}

def get_model_metadata() -> Dict[str, Any]:
    global INVENTORY_SYSTEM
    # ใช้ len(INVENTORY_SYSTEM) แทนการเช็คด้วย globals()
    if not INVENTORY_SYSTEM: 
        try:
            INVENTORY_SYSTEM = load_model_system()
            logger.info("Hybrid AI System loaded.")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            INVENTORY_SYSTEM = {} # ตั้งเป็น dict เปล่า
    return INVENTORY_SYSTEM

app = FastAPI(title="Hybrid Inventory AI", version="5.0")

origins = ["http://localhost:3000", "http://127.0.0.1:3000","http://localhost:5173","http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    entrypoint.prepare_environment()

# 🛑 ENDPOINT เดิม (สำหรับ Upload Excel)
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
        results = predict_inventory_usage(metadata, file_content, forecast_days)
        
        # Format for Dashboard
        return {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            "Monthly_Time_Series_Data": results['Monthly_Chart_Data'], 
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# 🆕 ENDPOINT ใหม่ (สำหรับ Initial Load)
@app.get("/initial_forecast")
async def initial_forecast(
    forecast_days: int = Query(7, ge=1), # 7 วันเป็นค่า default
    metadata: Dict[str, Any] = Depends(get_model_metadata)
):
    if not metadata:
        raise HTTPException(status_code=503, detail="AI Model not ready.")
        
    try:
        # 1. ดึงไฟล์ Excel อ้างอิงจาก Server
        file_content = get_reference_data() 
        
        # 2. รัน prediction logic
        results = predict_inventory_usage(metadata, file_content, forecast_days)

        # 3. Format และคืนค่าผลลัพธ์ (ใช้โครงสร้างเดียวกับ /predict)
        return {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            # 🆕 ใช้คีย์ใหม่ที่รวมข้อมูล Actual (12M) และ Predicted (4M) รายเดือนเข้าด้วยกัน
            "Monthly_Time_Series_Data": results['Monthly_Chart_Data'], 
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            }
        }
    except FileNotFoundError as e:
        logger.error(f"Initial Load Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: Reference Excel file not found. Please ensure exists.")
    except Exception as e:
        logger.error(f"Prediction failed during initial load: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    # version 3