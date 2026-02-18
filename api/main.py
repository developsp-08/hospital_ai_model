import sys
import os
# 🆕 1. เพิ่ม dotenv เพื่อให้รันในเครื่องแล้วเจอ Database
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env ทันที
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks
from io import BytesIO
# 🆕 เพิ่ม timedelta เพื่อใช้บวกเวลา 7 ชั่วโมง
from datetime import datetime, timedelta
import pandas as pd
import json
import numpy as np # จำเป็นสำหรับ Custom JSON Encoder
# 🆕 เพิ่ม SQLAlchemy สำหรับเชื่อมต่อ Database
from sqlalchemy import create_engine, text
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

# 🆕 รับค่า DATABASE_URL จาก Environment Variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# 🆕 สร้าง Database Engine (Connection Pool)
db_engine = None
if DATABASE_URL:
    try:
        # pool_size=5: เปิด connection ค้างไว้ 5 อัน
        # max_overflow=10: ถ้าคนใช้เยอะยอมให้เกินได้อีก 10 อัน
        db_engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
        logger.info("✅ Database Engine Created (Connected to Neon DB)")
    except Exception as e:
        logger.error(f"❌ Failed to create DB engine: {e}")


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

# 🆕 ฟังก์ชันสร้างตาราง (แยกออกมาเพื่อให้เรียกใช้ตอนเปิด Server ได้เลย)
def init_db_table():
    if not db_engine:
        return
    try:
        with db_engine.connect() as conn:
            # สร้างตาราง upload_logs ถ้ายังไม่มี
            create_table_sql = text("""
            CREATE TABLE IF NOT EXISTS upload_logs (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP,
                forecast_days INTEGER,
                status TEXT
            );
            """)
            conn.execute(create_table_sql)
            conn.commit()
            logger.info("✅ Database Table 'upload_logs' checked/created successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database table: {e}")

# 🆕 ฟังก์ชันสำหรับบันทึก Log ลง Neon Database (ปรับแก้ให้มีแค่ 1 Record และเวลาไทย)
def log_upload_to_neon(filename: str, status: str, forecast_days: int):
    if not db_engine:
        logger.warning("⚠️ No Database Engine found. Skipping DB logging.")
        return

    try:
        # 🕒 คำนวณเวลาประเทศไทย (UTC + 7 ชั่วโมง)
        thai_time = datetime.utcnow() + timedelta(hours=7)

        # ใช้ Connection จาก Engine (จัดการเปิด-ปิดให้อัตโนมัติ)
        with db_engine.connect() as conn:
            
            # 1. ลบข้อมูลเก่าทั้งหมดทิ้งก่อน (DELETE ALL)
            delete_sql = text("DELETE FROM upload_logs;")
            conn.execute(delete_sql)

            # 2. บันทึกข้อมูลใหม่ พร้อมเวลาไทย (INSERT NEW)
            # ระบุคอลัมน์ upload_time ชัดเจน
            insert_sql = text("""
            INSERT INTO upload_logs (filename, status, forecast_days, upload_time)
            VALUES (:filename, :status, :days, :upload_time);
            """)
            
            # ส่ง parameter แบบ Dictionary
            conn.execute(insert_sql, {
                "filename": filename, 
                "status": status, 
                "days": forecast_days,
                "upload_time": thai_time # ส่งเวลาไทยเข้าไป
            })
            
            conn.commit() # ยืนยันการเปลี่ยนแปลง
            
        logger.info(f"✅ Logged upload activity to Neon DB (Time: {thai_time}): {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log to Neon DB: {e}")


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

# 🆕 สั่งให้สร้างตารางทันทีที่ Server เริ่มทำงาน
@app.on_event("startup")
async def startup_event():
    init_db_table()

# 🆕 ENDPOINT ใหม่: สำหรับดึงข้อมูล Log ล่าสุดจาก Database
@app.get("/latest_upload_log")
async def get_latest_upload_log():
    if not db_engine:
        return None
    try:
        with db_engine.connect() as conn:
            # ดึงข้อมูลแถวแรกสุด
            query = text("SELECT filename, upload_time, forecast_days FROM upload_logs LIMIT 1")
            result = conn.execute(query).fetchone()
            
            if result:
                return {
                    "filename": result[0],
                    "upload_time": result[1], # ส่ง DateTime กลับไป
                    "forecast_days": result[2]
                }
            return None # ถ้าไม่มีข้อมูล
    except Exception as e:
        logger.error(f"Error fetching latest log: {e}")
        return None

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
    
    # อ่านชื่อไฟล์ก่อน await file.read() เพื่อเอาไปเก็บ Log
    filename = file.filename 
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
            
            # 🆕 เพิ่ม Task บันทึก Log ลง Neon DB (ทำงานเบื้องหลัง)
            background_tasks.add_task(log_upload_to_neon, filename, "Updated & Retraining", forecast_days)
            
            logger.info("ℹ️ Using current model while retraining runs in background.")

        else:
            logger.info(f"ℹ️ ไฟล์เดิม เปลี่ยน forecast_days เป็น {forecast_days}: ข้ามขั้นตอนเทรน")
            # 🆕 เพิ่ม Task บันทึก Log กรณีไม่ได้อัปเดต (ไฟล์ซ้ำ)
            background_tasks.add_task(log_upload_to_neon, filename, "No Update (Duplicate)", forecast_days)

        # --- step 5: predict ---
        results = predict_inventory_usage(metadata, file_content, forecast_days, is_upload=True)
        
        # เตรียมข้อมูลสำหรับ Response
        response_data = {
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
        
        # ⚡ Instant Cache Update (บันทึกลง Cache ทันทีเพื่อให้ตอน Refresh หน้าจอได้ข้อมูลชุดเดียวกัน)
        try:
            CACHE_FILE = os.path.join(DATA_DIR_PATH, f'dashboard_cache_{forecast_days}.json')
            
            # ตัด Message ออกก่อนเซฟลง Cache
            cache_data = response_data.copy()
            if "Message" in cache_data: del cache_data["Message"]
            
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, cls=NpEncoder, ensure_ascii=False)
            logger.info(f"💾 Instant Cache Updated for {forecast_days} days.")
        except Exception as e:
            logger.error(f"Failed to update instant cache: {e}")

        return response_data

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # 🆕 บันทึก Log กรณี Error ลง DB ด้วย
        if filename:
             background_tasks.add_task(log_upload_to_neon, filename, f"Error: {str(e)}", forecast_days)
             
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