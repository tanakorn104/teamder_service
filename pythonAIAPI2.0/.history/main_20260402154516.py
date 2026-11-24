from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import os

FEATURES = [
    'TeamSize', 'Env_Type',
    'Tag_Dim_1', 'Tag_Dim_2', 'Tag_Dim_3', 'Tag_Dim_4', 
    'Tag_Dim_5', 'Tag_Dim_6', 'Tag_Dim_7', 'Tag_Dim_8',
    'BFI_O_Mean', 'BFI_C_Mean', 'BFI_E_Mean', 'BFI_A_Mean', 'BFI_N_Mean', 
    'SJT_Lead_Mean', 'SJT_Plan_Mean', 'SJT_Support_Mean', 
    'Skill_Mean', 'Max_Skill'
]
TARGETS = ['Y1_Safety', 'Y2_Compatibility', 'Y3_Success', 'Y4_Innovation']
MODEL_PATH = 'team_model_master.pkl'
DATA_PATH = 'team_data_aggregated.csv'

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if os.path.exists(MODEL_PATH):
            app.state.model = joblib.load(MODEL_PATH)
            print("✅ [Model] Master Model loaded.")
        if os.path.exists(DATA_PATH):
            app.state.df = pd.read_csv(DATA_PATH)
            app.state.df['Env_Type'] = app.state.df['Env_Type'].astype('category')
            print(f"✅ [Data] Database ({len(app.state.df)} teams) online.")
        else:
            app.state.df = pd.DataFrame(columns=FEATURES + TARGETS)
    except Exception as e:
        print(f"❌ [Startup Error] {e}")
    yield

app = FastAPI(title="Strategic Team AI Predictor", lifespan=lifespan)
RETRAIN_BATCH_SIZE = 10  
app.state.new_feedback_count = 0  

class ProjectConfig(BaseModel):
    tags: List[str]
    env_type: str 

class Member(BaseModel):
    O: float; C: float; E: float; A: float; N: float
    Lead: float; Plan: float; Support: float
    Skill: float

class TeamRequest(BaseModel):
    project_cfg: ProjectConfig
    members: List[Member]

class FeedbackRequest(BaseModel):
    project_cfg: ProjectConfig
    members: List[Member]
    actual_scores: List[float]

def get_tag_embeddings(tags: List[str]) -> List[float]:
    if not tags: return [0.0] * 8
    clean_tags = [str(t).strip().lower() for t in tags]
    seed_val = sum(hash(t) for t in clean_tags) % (2**32 - 1)
    np.random.seed(seed_val)
    return np.random.rand(8).tolist()

def extract_team_features(project_cfg: ProjectConfig, members: List[Member]) -> dict:
    if not members: return {}
    env_type_clean = str(project_cfg.env_type).strip().lower()
    tags_vector = get_tag_embeddings(project_cfg.tags)
    data = np.array([[m.O, m.C, m.E, m.A, m.N, m.Lead, m.Plan, m.Support, m.Skill] for m in members])
    means = np.mean(data, axis=0)
    max_skill = np.max(data[:, 8]) 
    return {
        'TeamSize': float(len(members)), 'Env_Type': env_type_clean, 
        'Tag_Dim_1': float(tags_vector[0]), 'Tag_Dim_2': float(tags_vector[1]),
        'Tag_Dim_3': float(tags_vector[2]), 'Tag_Dim_4': float(tags_vector[3]),
        'Tag_Dim_5': float(tags_vector[4]), 'Tag_Dim_6': float(tags_vector[5]),
        'Tag_Dim_7': float(tags_vector[6]), 'Tag_Dim_8': float(tags_vector[7]),
        'BFI_O_Mean': float(means[0]), 'BFI_C_Mean': float(means[1]), 
        'BFI_E_Mean': float(means[2]), 'BFI_A_Mean': float(means[3]), 'BFI_N_Mean': float(means[4]),
        'SJT_Lead_Mean': float(means[5]), 'SJT_Plan_Mean': float(means[6]), 'SJT_Support_Mean': float(means[7]),
        'Skill_Mean': float(means[8]), 'Max_Skill': float(max_skill)
    }

def process_feedback_in_background(team_features: dict, actual_scores: List[float]):
    for i, target in enumerate(TARGETS): team_features[target] = actual_scores[i]
    new_row = pd.DataFrame([team_features])
    app.state.df = pd.concat([app.state.df, new_row], ignore_index=True)
    file_exists = os.path.isfile(DATA_PATH)
    new_row.to_csv(DATA_PATH, mode='a', header=not file_exists, index=False)
    app.state.new_feedback_count += 1
    if app.state.new_feedback_count >= RETRAIN_BATCH_SIZE:
        app.state.df['Env_Type'] = app.state.df['Env_Type'].astype('category')
        if hasattr(app.state, 'model') and app.state.model is not None:
             app.state.model.fit(app.state.df[FEATURES], app.state.df[TARGETS])
             joblib.dump(app.state.model, MODEL_PATH)
        app.state.new_feedback_count = 0

@app.post("/predict")
async def predict_team(request: TeamRequest):
    if not hasattr(app.state, 'model') or app.state.model is None:
        return {"error": "Model not loaded on server"}
    try:
        team_features = extract_team_features(request.project_cfg, request.members)
        input_df = pd.DataFrame([team_features])
        input_df['Env_Type'] = input_df['Env_Type'].astype('category')
        
        preds = app.state.model.predict(input_df)[0]
        
        y1_safety = min(100.0, max(0.0, round(float(preds[0]) * 100, 2)))
        y2_compat = min(100.0, max(0.0, round(float(preds[1]) * 100, 2)))
        y3_success = min(100.0, max(0.0, round(float(preds[2]) * 100, 2)))
        y4_innov = min(100.0, max(0.0, round(float(preds[3]) * 100, 2)))
        overall_score = round((y3_success * 0.40) + (y2_compat * 0.30) + (y1_safety * 0.20) + (y4_innov * 0.10), 2)
        
        return {
            "status": "success",
            "overall_match_percent": overall_score,
            "project_context": {
                "tags_cleaned": [t.strip().lower() for t in request.project_cfg.tags],
                "env_type_cleaned": request.project_cfg.env_type.strip().lower()
            },
            "predictions_detail": {
                "Y1_Safety": y1_safety,
                "Y2_Compatibility": y2_compat,
                "Y3_Success": y3_success,
                "Y4_Innovation": y4_innov
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.post("/feedback")
async def update_learning(request: FeedbackRequest, background_tasks: BackgroundTasks):
    try:
        team_features = extract_team_features(request.project_cfg, request.members)
        background_tasks.add_task(process_feedback_in_background, team_features, request.actual_scores)
        return {"status": "success", "progress_to_next_retrain": f"{app.state.new_feedback_count + 1}/{RETRAIN_BATCH_SIZE}"}
    except Exception as e:
        return {"error": f"Feedback acceptance failed: {str(e)}"}