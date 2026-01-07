import pandas as pd
import os
import numpy as np
from datetime import datetime
from core.llama_engine import get_patient_breakdown_ratios

def update_master_file(new_df, file_path='data/Training_Data_Final.xlsx'):
    try:
        
        if not os.path.exists(file_path):
            base_dir = os.getcwd()
            file_path = os.path.join(base_dir, 'data', 'Training_Data_Final.xlsx')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ไม่พบไฟล์ Master ที่: {file_path}")

        
        master_df = pd.read_excel(file_path)
        master_df = master_df.loc[:, ~master_df.columns.duplicated()]

        
        master_columns = [
            'Date', 'Visit_Campus', 'SKU', 'Item_Name', 'Usage_Qty', 
            'Total_SKU_Usage', 'Lead_Time_Days', 'Unit_Cost', 
            'Patient_E', 'Patient_I', 'Patient_O', 
            'Min_Stock', 'Max_Stock', 'UOM', 'Conversion_Factor'
        ]

        
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]
        
        
        if 'Current_Patient_Count' in new_df.columns:
            new_df['Visit_Campus'] = pd.to_numeric(new_df['Current_Patient_Count'], errors='coerce').fillna(0)
        
        if 'Latest_Usage_Qty' in new_df.columns:
            new_df['Usage_Qty'] = pd.to_numeric(new_df['Latest_Usage_Qty'], errors='coerce').fillna(0)

        if 'Conversion_Factor' in new_df.columns and 'Usage_Qty' in new_df.columns:
            new_df['Usage_Qty'] = new_df['Usage_Qty'] * pd.to_numeric(new_df['Conversion_Factor'], errors='coerce').fillna(1)

        
        new_df['Date'] = pd.to_datetime(new_df['Date']).dt.normalize()

       
        new_df_keys = (
            new_df['SKU'].astype(str) + 
            new_df['Date'].dt.strftime('%Y-%m-%d') + 
            new_df['Usage_Qty'].astype(float).astype(str)
        )
        
        master_keys = (
            master_df['SKU'].astype(str) + 
            pd.to_datetime(master_df['Date']).dt.strftime('%Y-%m-%d') + 
            master_df['Usage_Qty'].astype(float).astype(str)
        )

        if new_df_keys.isin(master_keys).all():
            print("ℹ️ ข้อมูลชุดนี้มีอยู่ในระบบแล้วและไม่มีการเปลี่ยนแปลง ข้ามการบันทึกและ Retrain")
            return False 

       
        
        # 3. Patient Breakdown Llama
        # if 'Visit_Campus' in new_df.columns:
        #     ratios = get_patient_breakdown_ratios()
        #     new_df['Patient_E'] = (new_df['Visit_Campus'] * ratios.get('E', 0.1)).astype(int)
        #     new_df['Patient_I'] = (new_df['Visit_Campus'] * ratios.get('I', 0.2)).astype(int)
        #     new_df['Patient_O'] = (new_df['Visit_Campus'] * ratios.get('O', 0.7)).astype(int)
        
        
        if 'Visit_Campus' in new_df.columns:
            total_visits = new_df['Visit_Campus']
            new_df['Patient_E'] = (total_visits * 0.15).astype(int)
            new_df['Patient_I'] = (total_visits * 0.25).astype(int)
            new_df['Patient_O'] = (total_visits * 0.60).astype(int)
            print("📊 Calculated patient breakdown using fixed ratios (15%, 25%, 60%)")

        
        existing_cols = [c for c in master_columns if c in new_df.columns]
        new_df_cleaned = new_df[existing_cols].copy()
        
        combined = pd.concat([master_df, new_df_cleaned], ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
        
        
        combined = combined.drop_duplicates(subset=['Date', 'SKU'], keep='last')

       
        combined['temp_month'] = combined['Date'].dt.to_period('M')
        combined['Total_SKU_Usage'] = combined.groupby('temp_month')['Usage_Qty'].transform('sum')
        combined = combined.drop(columns=['temp_month'])

        
        cols_to_numeric = ['Patient_E', 'Patient_I', 'Patient_O', 'Total_SKU_Usage', 'Visit_Campus', 'Usage_Qty']
        for col in cols_to_numeric:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0)

        
        final_df = combined[master_columns].copy()
        final_df = final_df.sort_values(['SKU', 'Date']).groupby('SKU').tail(60)
        
        
        final_df.to_excel(file_path, index=False)
        print(f"✅ บันทึก/อัปเดตข้อมูลสำเร็จ! (Total rows: {len(final_df)})")
        return True 

    except Exception as e:
        print(f"❌ Data Master Error: {e}")
        import traceback
        print(traceback.format_exc())
        raise e