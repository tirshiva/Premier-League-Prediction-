from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
from typing import List
from src.models.train_model import MatchPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Premier League Match Predictor API")

class MatchPredictionInput(BaseModel):
    home_team: str
    away_team: str

class MatchPredictionOutput(BaseModel):
    home_points: float
    away_points: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

# Global variables
model = None
team_stats = None

@app.on_event("startup")
async def load_model():
    """Load the trained model and team statistics"""
    global model, team_stats
    try:
        # Load team statistics
        team_stats = pd.read_csv('data/processed/team_stats.csv')
        
        # Initialize and load model
        model = MatchPredictor()
        
        # Load scaler parameters
        model.scaler.mean_ = np.load('models/scaler_mean.npy')
        model.scaler.scale_ = np.load('models/scaler_scale.npy')
        
        # Load the model from the latest training
        runs = mlflow.search_runs(order_by=["start_time DESC"])
        if len(runs) > 0:
            latest_run_id = runs.iloc[0].run_id
            model.model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
            logger.info("Model and team stats loaded successfully")
        else:
            raise Exception("No trained model found")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.post("/predict_match", response_model=MatchPredictionOutput)
async def predict_match(input_data: MatchPredictionInput):
    """
    Predict the outcome of a match between two teams
    """
    try:
        # Verify teams exist in our dataset
        if input_data.home_team not in team_stats['Team'].values:
            raise HTTPException(status_code=400, detail=f"Home team '{input_data.home_team}' not found in database")
        if input_data.away_team not in team_stats['Team'].values:
            raise HTTPException(status_code=400, detail=f"Away team '{input_data.away_team}' not found in database")
            
        # Make prediction
        result = model.predict_match(input_data.home_team, input_data.away_team, team_stats)
        
        return MatchPredictionOutput(**result)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams")
async def get_teams():
    """Get list of all teams"""
    try:
        teams = team_stats['Team'].tolist()
        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error getting teams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/team_stats/{team_name}")
async def get_team_stats(team_name: str):
    """Get statistics for a specific team"""
    try:
        if team_name not in team_stats['Team'].values:
            raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
            
        stats = team_stats[team_stats['Team'] == team_name].iloc[0].to_dict()
        return stats
    except Exception as e:
        logger.error(f"Error getting team stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
