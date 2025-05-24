import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def prepare_features(self, match_data, team_stats):
        """
        Prepare features for match prediction
        """
        features = []
        targets = []
        
        for _, match in match_data.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get team stats
            home_stats = team_stats[team_stats['Team'] == home_team].iloc[0]
            away_stats = team_stats[team_stats['Team'] == away_team].iloc[0]
            
            # Create feature vector
            feature_vector = [
                home_stats['GoalsScored'],
                home_stats['GoalsConceded'],
                home_stats['WinRate'],
                home_stats['PointsPerGame'],
                away_stats['GoalsScored'],
                away_stats['GoalsConceded'],
                away_stats['WinRate'],
                away_stats['PointsPerGame']
            ]
            
            # Create target vector [home_points, away_points]
            target_vector = [match['HomePoints'], match['AwayPoints']]
            
            features.append(feature_vector)
            targets.append(target_vector)
        
        return np.array(features), np.array(targets)
    
    def train(self, match_data, team_stats):
        """
        Train the model using match data and team statistics
        """
        try:
            logger.info("Preparing features for training...")
            X, y = self.prepare_features(match_data, team_stats)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Start MLflow run
            with mlflow.start_run():
                # Train model
                self.model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                r2 = self.model.score(X_test_scaled, y_test)
                
                # Log metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                logger.info(f"Model training completed. MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
                # Save scaler
                Path("models").mkdir(exist_ok=True)
                np.save("models/scaler_mean.npy", self.scaler.mean_)
                np.save("models/scaler_scale.npy", self.scaler.scale_)
                
                return mse, rmse, r2
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict_match(self, home_team, away_team, team_stats):
        """
        Predict the outcome of a match between two teams
        """
        try:
            # Get team stats
            home_stats = team_stats[team_stats['Team'] == home_team].iloc[0]
            away_stats = team_stats[team_stats['Team'] == away_team].iloc[0]
            
            # Create feature vector
            feature_vector = np.array([[
                home_stats['GoalsScored'],
                home_stats['GoalsConceded'],
                home_stats['WinRate'],
                home_stats['PointsPerGame'],
                away_stats['GoalsScored'],
                away_stats['GoalsConceded'],
                away_stats['WinRate'],
                away_stats['PointsPerGame']
            ]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            
            # Calculate win probabilities
            home_points, away_points = prediction
            
            result = {
                'home_points': home_points,
                'away_points': away_points,
                'home_win_prob': max(0, min(1, home_points / 3)),
                'draw_prob': max(0, min(1, 1 - abs(home_points - away_points) / 3)),
                'away_win_prob': max(0, min(1, away_points / 3))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in match prediction: {str(e)}")
            raise

if __name__ == "__main__":
    # Test model training
    match_data = pd.read_csv('data/processed/match_data.csv')
    team_stats = pd.read_csv('data/processed/team_stats.csv')
    
    predictor = MatchPredictor()
    mse, rmse, r2 = predictor.train(match_data, team_stats)
    print("Model training completed successfully!")
