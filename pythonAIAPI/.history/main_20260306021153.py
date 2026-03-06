from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Strategic Team AI Predictor & Learning API")

# --- การตั้งค่า Feature (ต้องเรียงลำดับให้เป๊ะตามตอน Train) ---
FEATURES = [
    'TeamSize', 'Env_Type', 'BFI_O', 'BFI_C', 'BFI_E', 
    'BFI_A', 'BFI_N', 'SJT_Lead', 'SJT_Plan', 'SJT_Support', 
    'Skill_Mean', 'Max_Skill'
]
TARGETS = ['Y1_Safety', 'Y2_Compatibility', 'Y3_Success', 'Y4_Innovation']
MODEL_PATH = 'team_model_master.pkl'
DATA_PATH = 'team_data_aggregated.csv'

@app.on_event("startup")
def load_assets():
    try:
        if os.path.exists(MODEL_PATH):
            app.state.model = joblib.load(MODEL_PATH)
            print("✅ [Model] Master Model loaded.")
        if os.path.exists(DATA_PATH):
            app.state.df = pd.read_csv(DATA_PATH)
            print(f"✅ [Data] Aggregated Database ({len(app.state.df)} teams) online.")
    except Exception as e:
        print(f"❌ [Startup Error] {e}")

# --- โครงสร้างข้อมูล (Schemas) ---
class Member(BaseModel):
    O: float; C: float; E: float; A: float; N: float
    Lead: float; Plan: float; Support: float
    Skill: float; Env_Type: int

class TeamRequest(BaseModel):
    members: List[Member]

class FeedbackRequest(BaseModel):
    members: List[Member]
    actual_scores: List[float]

# --- Helper Function: สกัดฟีเจอร์ระดับทีม (หัวใจสำคัญ) ---
def extract_team_features(members: List[Member]):
    if not members:
        return None
    
    team_size = len(members)
    env_type = members[0].Env_Type # ใช้บริบทของคนแรกเป็นหลัก
    
    # ดึงค่าดิบออกมาเป็น Matrix เพื่อคำนวณสถิติ
    data = np.array([[
        m.O, m.C, m.E, m.A, m.N, 
        m.Lead, m.Plan, m.Support, m.Skill
    ] for m in members])
    
    # คำนวณค่าเฉลี่ย (Mean) ของแต่ละ Trait
    means = np.mean(data, axis=0)
    
    # คำนวณค่าสูงสุด (Max Skill) เพื่อให้ AI เห็น "ตัวแบก"
    max_skill = np.max(data[:, 8]) # Index 8 คือ Skill
    
    # สร้าง List ของฟีเจอร์ตามลำดับใน FEATURES
    vector = [
        float(team_size), 
        float(env_type),
        float(means[0]), float(means[1]), float(means[2]), float(means[3]), float(means[4]), # BFI
        float(means[5]), float(means[6]), float(means[7]), # SJT
        float(means[8]), # Skill_Mean
        float(max_skill) # Max_Skill
    ]
    return vector

# --- Endpoints ---

@app.post("/predict")
async def predict_team(request: TeamRequest):
    if not hasattr(app.state, 'model') or app.state.model is None:
        return {"error": "Model not loaded on server"}
    
    try:
        # 1. รวมข้อมูลสมาชิกให้กลายเป็นข้อมูลทีม 1 แถว
        team_vector = extract_team_features(request.members)
        print(f"🔍 DEBUG: Input Vector -> {team_vector}") # ตรวจสอบค่าใน Console
        
        # 2. ทำนายผล (ใช้ DataFrame เพื่อรักษาชื่อ Feature)
        input_df = pd.DataFrame([team_vector], columns=FEATURES)
        raw_preds = app.state.model.predict(input_df)
        
        # 3. จัดการผลลัพธ์ (MultiOutputRegressor คืนค่าเป็น Array 2D)
        preds = raw_preds[0] 
        
        return {
            "status": "success",
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
    if not hasattr(app.state, 'df') or app.state.df is None:
        return {"error": "Database not found"}

    try:
        team_vector = extract_team_features(request.members)
        
        # สร้าง Dictionary และเพิ่ม Target จริงเข้าไป
        new_entry = dict(zip(FEATURES, team_vector))
        for i, target in enumerate(TARGETS):
            new_entry[target] = request.actual_scores[i]
        
        # อัปเดตไฟล์ CSV และโหลดกลับเข้า Memory
        new_row = pd.DataFrame([new_entry])
        app.state.df = pd.concat([app.state.df, new_row], ignore_index=True)
        app.state.df.to_csv(DATA_PATH, index=False)
        
        # Retrain ทันทีเพื่อให้ AI "จำ" เคสนี้
        print(f"🔄 Retraining with {len(app.state.df)} teams...")
        app.state.model.fit(app.state.df[FEATURES], app.state.df[TARGETS])
        joblib.dump(app.state.model, MODEL_PATH)
        
        return {"status": "success", "message": "Model updated with feedback"}
    except Exception as e:
        return {"error": f"Learning failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)