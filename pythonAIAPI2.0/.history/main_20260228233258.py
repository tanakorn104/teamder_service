from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Team AI Predictor & Learning API")

# รายชื่อ Features และ Targets (ต้องตรงกับตอนเทรน)
FEATURES = ['TeamSize', 'BFI_O', 'BFI_C', 'BFI_E', 'BFI_A', 'BFI_N', 'SJT_Lead', 'SJT_Plan', 'SJT_Support']
TARGETS = ['Y1_Safety', 'Y2_Compatibility', 'Y3_Success', 'Y4_Innovation']

# 1. โหลดโมเดลและฐานข้อมูล
MODEL_PATH = 'team_model_master.pkl'
DATA_PATH = 'synthetic_team_data_5000.csv'

try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    print("✅ Model and Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    model = None
    df = None

# --- โครงสร้างข้อมูล (Schemas) ---

class Member(BaseModel):
    O: float; C: float; E: float; A: float; N: float
    Lead: float; Plan: float; Support: float

class TeamRequest(BaseModel):
    members: List[Member]

class FeedbackRequest(BaseModel):
    members: List[Member]
    actual_scores: List[float] # รับเป็น [Y1, Y2, Y3, Y4] เช่น [0.9, 0.8, 0.85, 0.7]

# --- Endpoints ---

@app.post("/predict")
async def predict_team(request: TeamRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        member_data = [[m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support] for m in request.members]
        team_size = len(request.members)
        mean_traits = np.mean(member_data, axis=0)
        
        input_df = pd.DataFrame([[team_size] + mean_traits.tolist()], columns=FEATURES)
        
        raw_preds = model.predict(input_df)
        preds = np.atleast_1d(raw_preds.squeeze())
        
        return {
            "predictions": {
                "Y1_Safety": round(float(preds[0]) * 100, 2),
                "Y2_Compatibility": round(float(preds[1]) * 100, 2),
                "Y3_Success": round(float(preds[2]) * 100, 2),
                "Y4_Innovation": round(float(preds[3]) * 100, 2)
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/feedback")
async def give_feedback(request: FeedbackRequest):
    global model, df
    if model is None or df is None:
        return {"error": "System not ready for feedback"}

    try:
        # 1. เตรียมข้อมูลใหม่ (คำนวณค่าเฉลี่ยทีม)
        member_data = [[m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support] for m in request.members]
        mean_traits = np.mean(member_data, axis=0)
        
        new_entry = {
            'TeamSize': len(request.members),
            'BFI_O': mean_traits[0], 'BFI_C': mean_traits[1], 'BFI_E': mean_traits[2], 
            'BFI_A': mean_traits[3], 'BFI_N': mean_traits[4],
            'SJT_Lead': mean_traits[5], 'SJT_Plan': mean_traits[6], 'SJT_Support': mean_traits[7],
            'Y1_Safety': request.actual_scores[0],
            'Y2_Compatibility': request.actual_scores[1],
            'Y3_Success': request.actual_scores[2],
            'Y4_Innovation': request.actual_scores[3]
        }
        
        # 2. อัปเดต CSV
        new_row_df = pd.DataFrame([new_entry])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        
        # 3. Retrain โมเดลใน Memory
        model.fit(df[FEATURES], df[TARGETS])
        
        # 4. บันทึกโมเดลทับไฟล์เดิม
        joblib.dump(model, MODEL_PATH)
        
        return {
            "status": "Success",
            "message": "AI has learned from this feedback",
            "new_dataset_size": len(df)
        }
    except Exception as e:
        return {"error": f"Feedback update failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)