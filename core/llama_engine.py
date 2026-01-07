import ollama
import json
import logging

logger = logging.getLogger(__name__)

def get_column_mapping(new_columns, target_columns):
    """
    ใช้ Llama 3.2 เพื่อจับคู่ชื่อหัวตารางจากไฟล์อัปโหลดให้ตรงกับ Master
    """
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านข้อมูลโรงพยาบาล (Hospital Data Expert)
    เป้าหมาย: จับคู่ชื่อคอลัมน์จาก 'Uploaded Columns' ให้ตรงกับ 'Master Columns' ตามความหมาย
    
    Master Columns (Target): {target_columns}
    Uploaded Columns (Source): {new_columns}
    
    กฎการทำงาน:
    1. จับคู่คอลัมน์ที่มีความหมายเดียวกัน (เช่น 'Latest_Usage_Qty' คู่กับ 'Usage_Qty')
    2. 'Current_Patient_Count' ต้องคู่กับ 'Visit_Campus' เสมอ
    3. ส่งค่ากลับเป็น JSON object เท่านั้น ห้ามมีคำอธิบาย
    """
    
    try:
        response = ollama.generate(
            model='llama3.2:3b', 
            prompt=prompt, 
            format='json',
            options={'temperature': 0} 
        )
        mapping = json.loads(response['response'])
        logger.info(f"🤖 Llama Mapping Suggestion: {mapping}")
        return mapping
    except Exception as e:
        logger.error(f"❌ Llama Mapping Error: {e}")
        return {}

def get_patient_breakdown_ratios():
    """
    🆕 ใหม่: ใช้ Llama เพื่อทำนายสัดส่วนคนไข้ Emergency, IPD, และ OPD 
    เพื่อใช้กระจายยอดจาก Current_Patient_Count
    """
    prompt = """
    ในฐานะผู้เชี่ยวชาญการจัดการโรงพยาบาล ช่วยประมาณสัดส่วนคนไข้ (Percentage Breakdown) ดังนี้:
    1. Emergency Patients (Patient_E) - คนไข้ฉุกเฉิน
    2. In-patient (Patient_I) - คนไข้ใน (นอนโรงพยาบาล)
    3. Out-patient (Patient_O) - คนไข้นอก (รับยาแล้วกลับ)
    
    กฎ:
    - ผลรวมของทั้ง 3 ส่วนต้องเท่ากับ 1.0 (100%)
    - คืนค่าเป็น JSON object ที่มี key เป็น "E", "I", "O" และ value เป็นทศนิยม
    - ตัวอย่างคำตอบ: {"E": 0.10, "I": 0.20, "O": 0.70}
    """
    
    try:
        response = ollama.generate(
            model='llama3.2:3b', 
            prompt=prompt, 
            format='json',
            options={'temperature': 0}
        )
        ratios = json.loads(response['response'])
        logger.info(f"🤖 Llama Suggested Patient Ratios: {ratios}")
        return ratios
    except Exception as e:
        logger.error(f"❌ Llama Ratio Error: {e}")
        # ค่า Default กรณี AI ขัดข้อง (E 10%, I 20%, O 70%)
        return {"E": 0.10, "I": 0.20, "O": 0.70}