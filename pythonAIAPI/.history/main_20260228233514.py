from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Team AI Predictor & Learning API")

# --- การตั้งค่าเบื้องต้น ---
FEATURES = ['TeamSize', 'BFI_O', 'BFI_C', 'BFI_E', 'BFI_A', 'BFI_N', 'SJT_Lead', 'SJT_Plan', 'SJT_Support']
TARGETS = ['Y1_Safety', 'Y2_Compatibility', 'Y3_Success', 'Y4_Innovation']
MODEL_PATH = 'team_model_master.pkl'
DATA_PATH = 'synthetic_team_data_5000.csv'

# --- ระบบโหลดข้อมูลเมื่อ Startup ---
@app.on_event("startup")
def load_assets():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
            app.state.model = joblib.load(MODEL_PATH)
            app.state.df = pd.read_csv(DATA_PATH)
            print("✅ [Ready] Model and Database are online.")
        else:
            print("❌ [Error] Model or CSV file missing in the directory.")
            app.state.model = None
            app.state.df = None
    except Exception as e:
        print(f"❌ [Startup Error] {e}")
        app.state.model = None
        app.state.df = None

# --- โครงสร้างข้อมูล (Schemas) ---
class Member(BaseModel):
    O: float; C: float; E: float; A: float; N: float
    Lead: float; Plan: float; Support: float

class TeamRequest(BaseModel):
    members: List[Member]

class FeedbackRequest(BaseModel):
    members: List[Member]
    actual_scores: List[float] # รับเป็น [Y1, Y2, Y3, Y4] เช่น [0.95, 0.88, 0.90, 0.85]

# --- Endpoints ---

@app.post("/predict")
async def predict_team(request: TeamRequest):
    model = app.state.model
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        member_data = [[m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support] for m in request.members]
        team_size = len(request.members)
        mean_traits = np.mean(member_data, axis=0)
        
        # เตรียมข้อมูลในรูปแบบ DataFrame เพื่อส่งเข้าโมเดล
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
async def update_learning(request: FeedbackRequest):
    model = app.state.model
    df = app.state.df
    
    if model is None or df is None:
        return {"error": "System assets missing, cannot learn."}

    try:
        # 1. เตรียมข้อมูลใหม่ (คำนวณค่าเฉลี่ยทีมจากสมาชิกที่ส่งมา)
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
        
        # 2. อัปเดตฐานข้อมูล (CSV)
        new_row_df = pd.DataFrame([new_entry])
        app.state.df = pd.concat([df, new_row_df], ignore_index=True)
        app.state.df.to_csv(DATA_PATH, index=False) # เซฟทับไฟล์เดิม
        
        # 3. Retrain โมเดลใหม่ด้วยข้อมูลล่าสุดทั้งหมด
        print(f"--- 🔄 Retraining started (Total samples: {len(app.state.df)}) ---")
        app.state.model.fit(app.state.df[FEATURES], app.state.df[TARGETS])
        
        # 4. เซฟโมเดลที่ฉลาดขึ้นทับไฟล์ .pkl
        joblib.dump(app.state.model, MODEL_PATH)
        
        return {
            "status": "Learning Complete",
            "message": "AI model has been updated with your feedback!",
            "new_database_size": len(app.state.df)
        }
    except Exception as e:
        return {"error": f"Feedback update failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)