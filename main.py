# main.py (FINAL VERSION)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from core.predictor import load_model_system, predict_inventory_usage
from starlette.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_metadata() -> Dict[str, Any]:
    global INVENTORY_SYSTEM
    if 'INVENTORY_SYSTEM' not in globals():
        try:
            INVENTORY_SYSTEM = load_model_system()
            logger.info("Hybrid AI System loaded.")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            INVENTORY_SYSTEM = None
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

@app.post("/predict")
async def predict_inventory_from_file(
    file: UploadFile = File(...),
    forecast_days: int = Form(3),
    metadata: Dict[str, Any] = Depends(get_model_metadata)
):
    if metadata is None:
        raise HTTPException(status_code=503, detail="AI Model not ready.")
    
    file_content = await file.read()
    try:
        results = predict_inventory_usage(metadata, file_content, forecast_days)
        
        # Format for Dashboard
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
        raise HTTPException(status_code=500, detail=f"Error: {e}")