from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Strategic Team AI Predictor & Learning API")

# --- การตั้งค่าที่อัปเดตตามโมเดลใหม่ ---
# ลำดับ Features ต้องตรงกับที่ใช้เทรน (Team-Level)
FEATURES = [
    'TeamSize', 'Env_Type', 'BFI_O', 'BFI_C', 'BFI_E', 
    'BFI_A', 'BFI_N', 'SJT_Lead', 'SJT_Plan', 'SJT_Support', 
    'Skill_Mean', 'Max_Skill'
]
TARGETS = ['Y1_Safety', 'Y2_Compatibility', 'Y3_Success', 'Y4_Innovation']
MODEL_PATH = 'team_model_master.pkl'
DATA_PATH = 'team_data_10000.csv' # แนะนำให้ใช้ไฟล์ที่ผ่านการ groupby มาแล้วเพื่อความแม่นยำ

# --- ระบบโหลดข้อมูลเมื่อ Startup ---
@app.on_event("startup")
def load_assets():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
            app.state.model = joblib.load(MODEL_PATH)
            app.state.df = pd.read_csv(DATA_PATH)
            print(f"✅ [Ready] Model and Aggregated Database ({len(app.state.df)} teams) online.")
        else:
            print("❌ [Warning] Assets missing. Ensure pkl and aggregated csv exist for high accuracy.")
            app.state.model = None
            app.state.df = None
    except Exception as e:
        print(f"❌ [Startup Error] {e}")

# --- โครงสร้างข้อมูล (Schemas) ---
class Member(BaseModel):
    # ข้อมูล 10 ค่าตามที่ตกลงกัน
    O: float; C: float; E: float; A: float; N: float
    Lead: float; Plan: float; Support: float
    Skill: float; Env_Type: int # 0=Student, 1=Pro

class TeamRequest(BaseModel):
    members: List[Member]

class FeedbackRequest(BaseModel):
    members: List[Member]
    actual_scores: List[float] # [Y1, Y2, Y3, Y4]

# --- Helper Function: สรุปข้อมูลสมาชิกให้เป็นข้อมูลทีม ---
def aggregate_team_data(members: List[Member]):
    team_size = len(members)
    # แปลงข้อมูลสมาชิกเป็น Matrix เพื่อคำนวณ
    m_data = np.array([[
        m.O, m.C, m.E, m.A, m.N, 
        m.Lead, m.Plan, m.Support, m.Skill
    ] for m in members])
    
    means = np.mean(m_data, axis=0)
    max_skill = np.max(m_data[:, 8]) # ดึงค่าสูงสุดจากคอลัมน์ Skill
    env_type = members[0].Env_Type   # ใช้บริบทของคนแรกเป็นเกณฑ์ทีม
    
    # จัดเรียงตามลำดับ FEATURES เป๊ะๆ
    return [
        team_size, env_type, 
        means[0], means[1], means[2], means[3], means[4], # BFI
        means[5], means[6], means[7],                     # SJT
        means[8], max_skill                               # Skill_Mean, Max_Skill
    ]

# --- Endpoints ---

@app.post("/predict")
async def predict_team(request: TeamRequest):
    model = app.state.model
    if model is None: return {"error": "Model not loaded"}
    
    try:
        # 1. รวมกลุ่มข้อมูลสมาชิกให้กลายเป็น 1 แถวของทีม (Strategic Input)
        team_vector = aggregate_team_data(request.members)
        
        # 2. ทำนายผล
        input_df = pd.DataFrame([team_vector], columns=FEATURES)
        raw_preds = model.predict(input_df)
        preds = np.atleast_1d(raw_preds.squeeze())
        
        return {
            "predictions": {
                "Y1_Safety": round(float(preds[0]) * 100, 2),
                "Y2_Compatibility": round(float(preds[1]) * 100, 2),
                "Y3_Success": round(float(preds[2]) * 100, 2),
                "Y4_Innovation": round(float(preds[3]) * 100, 2)
            },
            "context": "Professional" if request.members[0].Env_Type == 1 else "Student"
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
        # 1. สรุปข้อมูลทีมใหม่จาก Feedback
        team_vector = aggregate_team_data(request.members)
        
        new_entry = dict(zip(FEATURES, team_vector))
        # เพิ่มคะแนนจริงเข้าไปใน Entry
        for i, target in enumerate(TARGETS):
            new_entry[target] = request.actual_scores[i]
        
        # 2. อัปเดตฐานข้อมูลทีม (Aggregated CSV)
        new_row_df = pd.DataFrame([new_entry])
        app.state.df = pd.concat([df, new_row_df], ignore_index=True)
        app.state.df.to_csv(DATA_PATH, index=False)
        
        # 3. Retrain โมเดล Master (เทรนด้วยข้อมูลระดับทีมเพื่อให้ R2 สูง)
        print(f"--- 🔄 Retraining started (Total teams: {len(app.state.df)}) ---")
        app.state.model.fit(app.state.df[FEATURES], app.state.df[TARGETS])
        
        # 4. บันทึกโมเดล
        joblib.dump(app.state.model, MODEL_PATH)
        
        return {
            "status": "Learning Complete",
            "new_database_size": len(app.state.df),
            "info": "AI has learned from this team's strategic performance."
        }
    except Exception as e:
        return {"error": f"Feedback update failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)