from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from core.predictor import load_model, make_predictions
import pandas as pd
import io

app = FastAPI(title="AI Inventory Prediction Service")

# Setup CORS: สำคัญมากเพื่อให้ Node.js และ React คุยกันได้
origins = [
    "http://localhost:3000",  # React Frontend
    "http://localhost:3001",  # Node.js Backend
    "*" 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด Model เมื่อ Server เริ่มต้น (ใช้ Depends หรือโหลด Global)
inventory_model = load_model()

# Helper function to get the loaded model instance
def get_model():
    """Provides the pre-loaded AI model instance."""
    return inventory_model

@app.post("/api/predict/")
async def predict_inventory(
    file: UploadFile = File(...),
    model = Depends(get_model)
):
    """
    Endpoint ที่รับไฟล์ Excel จาก Backend (Node.js) และคืนผลการทำนายสต็อก
    """
    if model is None:
        return {"status": "error", "message": "AI Model not initialized. Check server logs."}
        
    try:
        # 1. อ่านไฟล์ Excel ที่อัปโหลดจาก Buffer
        file_content = await file.read()
        # Use io.BytesIO to read the binary content directly with pandas
        df_uploaded = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        
        # 2. ทำนายผลลัพธ์
        predictions_data = make_predictions(model, df_uploaded)
        
        # 3. คืนผลลัพธ์
        return {"status": "success", "predictions": predictions_data}

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

@app.get("/api/health/")
def health_check():
    return {"status": "ok", "message": "AI Predictor is running."}