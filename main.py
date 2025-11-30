# main.py (FINAL VERSION)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from core.predictor import load_model_metadata, predict_inventory_usage
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency: Load Model Metadata ---
def get_model_metadata() -> Dict[str, Any]:
    global INVENTORY_MODEL_METADATA
    if 'INVENTORY_MODEL_METADATA' not in globals():
        try:
            INVENTORY_MODEL_METADATA = load_model_metadata()
            logger.info("Model loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Error loading model: {e}")
            INVENTORY_MODEL_METADATA = None
    return INVENTORY_MODEL_METADATA

# --- Initialize FastAPI ---
app = FastAPI(
    title="Hospital Inventory Predictor API (XGBoost)",
    version="1.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "XGBoost Inventory Predictor API is running."}

@app.post("/predict")
async def predict_inventory_from_file(
    file: UploadFile = File(..., description="ไฟล์ Excel ที่มีข้อมูลการใช้งานล่าสุด (Snapshot)"),
    forecast_days: int = Form(3, description="จำนวนเดือนที่ต้องการทำนายล่วงหน้า"),
    metadata: Dict[str, Any] = Depends(get_model_metadata)
):
    """
    รับไฟล์ Excel ล่าสุดเพื่อทำนายยอด Usage, จัดลำดับความสำคัญ, และคำนวณต้นทุน
    """
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model service unavailable. Please run train_and_save_model.ipynb first.")
    
    # 1. อ่านไฟล์ที่อัปโหลดเป็น bytes
    file_content = await file.read()
    
    # 2. ทำนายและคำนวณ Actions
    try:
        results = predict_inventory_usage(
            metadata=metadata, 
            file_content=file_content,
            forecast_days=forecast_days
        )
        
        # 3. จัดโครงสร้าง Output สำหรับ Dashboard
        return {
            "Total_SKUs_Trained": results['metrics']['total_skus'],
            "Total_Reorder_Cost": results['metrics']['reorder_cost_total'],
            "Forecast_Data": results['forecast'],
            "Priority_Metrics": {
                "High_Priority_Items": results['metrics']['high_priority_items'],
                "Medium_Priority_Items": results['metrics']['medium_priority_items'],
                "Action_Items_Summary": results['metrics']['action_items']
            }
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")