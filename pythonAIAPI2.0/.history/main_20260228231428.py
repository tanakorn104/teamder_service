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
# แก้ไขเฉพาะฟังก์ชัน predict_team ใน main.py ของคุณ

@app.post("/predict")
async def predict_team(request: TeamRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # 1. เตรียมข้อมูลสมาชิก
        member_data = [[m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support] for m in request.members]
        team_size = len(request.members)
        mean_traits = np.mean(member_data, axis=0)
        
        # 2. สร้าง DataFrame พร้อมระบุชื่อคอลัมน์ (เพื่อแก้ UserWarning)
        features = ['TeamSize', 'BFI_O', 'BFI_C', 'BFI_E', 'BFI_A', 'BFI_N', 'SJT_Lead', 'SJT_Plan', 'SJT_Support']
        input_values = [team_size] + mean_traits.tolist()
        input_df = pd.DataFrame([input_values], columns=features)
        
        # 3. ทำนายผล
        raw_preds = model.predict(input_df)
        
        # ตรวจสอบว่าได้มากี่ค่า (ป้องกัน IndexError)
        # ถ้าเป็น MultiOutput จะได้ Array 2D เช่น [[0.1, 0.2, 0.3, 0.4]]
        # เราจะ Flatten ให้เหลือชั้นเดียว
        preds = np.atleast_1d(raw_preds.squeeze())
        
        # ตรวจสอบว่าโมเดลคืนค่ามากี่ตัว
        if len(preds) < 4:
            return {
                "error": f"Model returned only {len(preds)} values, but 4 were expected. "
                         "Please check if you loaded the MultiOutput Master Model."
            }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)