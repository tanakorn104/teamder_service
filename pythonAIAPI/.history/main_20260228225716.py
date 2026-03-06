from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Team AI Predictor API")

# 1. โหลดโมเดล (ตรวจสอบให้แน่ใจว่าไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกัน)
try:
    model = joblib.load('team_model_master.pkl')
    print("Model loaded successfully!")
except:
    print("Error: 'team_model_master.pkl' not found.")

# 2. กำหนดโครงสร้างข้อมูลที่รับจาก NextJS
class Member(BaseModel):
    O: float
    C: float
    E: float
    A: float
    N: float
    Lead: float
    Plan: float
    Support: float

class TeamRequest(BaseModel):
    members: List[Member]

# 3. Endpoint สำหรับการทำนาย
@app.post("/predict")
async def predict_team(request: TeamRequest):
    members = request.members
    team_size = len(members)
    
    # แปลงข้อมูลจาก JSON เป็น List ของ List
    member_data = []
    for m in members:
        member_data.append([m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support])
    
    # คำนวณค่าเฉลี่ยของทีม
    mean_traits = np.mean(member_data, axis=0)
    
    # สร้าง Input Vector: [TeamSize + 8 traits]
    input_vector = np.insert(mean_traits, 0, team_size).reshape(1, -1)
    
    # ทำนายผล (Multi-Output)
    preds = model.predict(input_vector)[0]
    
    # ส่งผลลัพธ์กลับเป็น JSON
    return {
        "team_size": team_size,
        "predictions": {
            "Y1_Safety": round(float(preds[0]) * 100, 2),
            "Y2_Compatibility": round(float(preds[1]) * 100, 2),
            "Y3_Success": round(float(preds[2]) * 100, 2),
            "Y4_Innovation": round(float(preds[3]) * 100, 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)