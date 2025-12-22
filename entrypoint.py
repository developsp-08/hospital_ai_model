import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# 1. โหลดค่าจากไฟล์ .env (สำคัญมากสำหรับการรันในเครื่อง Local)
load_dotenv()

# 2. รายชื่อไฟล์ที่ต้องใช้ (ระบุ Path ให้ตรงกับที่ predictor.py เรียกหา)
REQUIRED_FILES = [
    'inventory_lstm_model.h5',
    'lead_time_classifier.pkl',
    'scaler_x.pkl',
    'scaler_y.pkl',
    'model_metadata.pkl',
    'Training_Data_Final.xlsx'
]

def prepare_environment():
    # ดึงค่า Environment Variables
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_REGION', 'ap-southeast-1')

    # ตรวจสอบว่ามีชื่อ Bucket หรือไม่ (ป้องกัน Error ที่คุณเจอ)
    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME is not set. Please check your .env file or Vercel settings.")
        return

    # สร้าง S3 Client
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )

    print(f"--- Starting Environment Preparation (Bucket: {bucket_name}) ---")

    for file_path in REQUIRED_FILES:
        # แยกชื่อโฟลเดอร์ออกมา (เช่น models หรือ data)
        directory = os.path.dirname(file_path)
        
        # ถ้ามีโฟลเดอร์ ให้สร้างโฟลเดอร์นั้นรอก่อนดาวน์โหลด
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {file_path} from S3...")
        try:
            # ดาวน์โหลดไฟล์จาก S3 มาวางในเครื่องตามโครงสร้างโฟลเดอร์เดิม
            s3.download_file(bucket_name, file_path, file_path)
        except Exception as e:
            print(f"⚠️ Could not download {file_path}: {e}")

    print("✅ Environment Ready! All models and data are in place.")

if __name__ == "__main__":
    prepare_environment()