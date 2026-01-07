import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import logging


BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Training_Data_Final.xlsx')

logger = logging.getLogger(__name__)

def run_retrain_process():
    try:
        logger.info("🚀 Starting Incremental Training (Fine-tuning)...")
        
        if not os.path.exists(DATA_PATH):
            logger.error(f"File not found: {DATA_PATH}")
            return False

        
        df = pd.read_excel(DATA_PATH)
        df.columns = df.columns.str.strip()
        
       
        df = df.drop_duplicates(subset=['Date', 'SKU'], keep='last')
        
        
        df.rename(columns={'Visit_Campus': 'Patient_Count', 'Min_Stock': 'Safety_Stock_Qty'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['month_num'] = df['Date'].dt.month
        
        
        important_cols = [
            'Patient_Count', 'Patient_E', 'Patient_I', 'Patient_O', 
            'Total_SKU_Usage', 'Unit_Cost', 'Safety_Stock_Qty', 'Lead_Time_Days'
        ]
        for col in important_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 3. STEP 1: XGBoost Classification (Lead Time Risk)
        X_class_features = ['Unit_Cost', 'Safety_Stock_Qty']
        df['LT_CLASS_TARGET'] = pd.cut(df['Lead_Time_Days'], bins=[-np.inf, 7, 30, np.inf], labels=[0, 1, 2]).astype(int)
        
        xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_classifier.fit(df[X_class_features], df['LT_CLASS_TARGET'])
        df['LT_CATEGORY_PRED'] = xgb_classifier.predict(df[X_class_features])
        
        # 4. STEP 2: LSTM Preparation
        TIME_STEPS = 12
        
        X_features = [
            'month_num', 'Patient_Count', 'Patient_E', 'Patient_I', 'Patient_O', 
            'Total_SKU_Usage', 'LT_CATEGORY_PRED', 'Safety_Stock_Qty'
        ]
        feature_cols = X_features + ['Usage_Qty']
        
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        scaled_feats = scaler_x.fit_transform(df[X_features])
        scaled_target = scaler_y.fit_transform(df[['Usage_Qty']].fillna(0))
        
        df_scaled = pd.DataFrame(np.hstack([scaled_feats, scaled_target]), columns=feature_cols)
        df_scaled['SKU'] = df['SKU'].values
        df_scaled['Date'] = df['Date'].values

        X_sequences, y_targets = [], []
        
        df_scaled = df_scaled.sort_values(['SKU', 'Date'])
        
        for sku in df_scaled['SKU'].unique():
            sku_data = df_scaled[df_scaled['SKU'] == sku]
            if len(sku_data) <= TIME_STEPS: 
                continue 
            
            values = sku_data[feature_cols].values
            for i in range(TIME_STEPS, len(values)):
                X_sequences.append(values[i-TIME_STEPS:i])
                y_targets.append(values[i, -1])

        if len(X_sequences) == 0:
            logger.warning("⚠️ ข้อมูลใหม่มีไม่พอสร้าง Sequence (ต้องมีอย่างน้อย 13 เดือนต่อ SKU) ข้ามการ Fit โมเดล")
            return True 

        X_train = np.array(X_sequences)
        y_train = np.array(y_targets)

        # 5. Fine-tuning LSTM
        model_path = os.path.join(MODEL_DIR, 'inventory_lstm_model.h5')
        if not os.path.exists(model_path):
            logger.error(f"ไม่พบไฟล์โมเดลที่: {model_path}")
            return False
            
       
        model = tf.keras.models.load_model(model_path, compile=False) 
        model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
        
        logger.info(f"Model re-compiled. Training shape: {X_train.shape}")
        
        
        model.fit(
            X_train, 
            y_train, 
            epochs=100, 
            batch_size=32, 
            verbose=1, 
            shuffle=False
        )
        
        
        y_pred_scaled = model.predict(X_train, verbose=0)
        
        
        y_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_predicted = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        
        errors = y_actual - y_predicted
        mae = np.mean(np.abs(errors)) # Avg. Margin of Error
        variance = np.var(errors)      # Error Variance
        
        # Calculate Accuracy (1 - WAPE)
        sum_actual = np.sum(y_actual)
        if sum_actual > 0:
            accuracy = max(0, (1 - (np.sum(np.abs(errors)) / sum_actual)) * 100)
        else:
            accuracy = 0.0

        
        print(f"\n --- Training Evaluation Metrics ---")
        print(f" Forecast Accuracy: {accuracy:.2f}%")
        print(f" Avg. Margin of Error: {mae:.2f} units")
        print(f" Error Variance: {variance:.4f}")
        print(f"--------------------------------------\n")
        
        
        model.save(model_path, save_format='h5')
        joblib.dump(xgb_classifier, os.path.join(MODEL_DIR, 'lead_time_classifier.pkl'))
        joblib.dump(scaler_x, os.path.join(MODEL_DIR, 'scaler_x.pkl'))
        joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.pkl'))
        
        # metadata = {'time_steps': TIME_STEPS, 'valid_features_lstm': X_features}
        metadata = {
            'time_steps': TIME_STEPS, 
            'valid_features_lstm': X_features,
            'last_metrics': {  
                'accuracy': f"{accuracy:.2f}",
                'mae': f"{mae:.2f}",
                'variance': f"{variance:.4f}"
            }
        }
        joblib.dump(metadata, os.path.join(MODEL_DIR, 'model_metadata.pkl'))
        
        logger.info(" Incremental Training สำเร็จและบันทึกโมเดลแล้ว")
        # return True
    
        return {
            "success": True,
            "accuracy": f"{accuracy:.2f}",
            "mae": f"{mae:.2f}",
            "variance": f"{variance:.4f}"
        }
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Retraining Failed: {traceback.format_exc()}")
        # return False
        return {"success": False, "error": str(e)}