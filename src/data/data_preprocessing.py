import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df: pd.DataFrame):
        """
        Preprocess the Champions League data for regression
        """
        try:
            # Drop unnecessary columns and handle missing values
            df = df.dropna()
            
            # Clean up goals column (remove time format)
            df['goals'] = df['goals'].str.split(':').str[0].astype(float)
            
            # Create features for regression (predicting goals)
            features = ['M.', 'Dif']  # Matches played and Goal difference
            target = 'goals'  # Predicting total goals
            
            # Prepare X and y
            X = df[features]
            y = df[target]
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            
            # Save processed data
            np.save('data/processed/X_train.npy', X_train)
            np.save('data/processed/X_test.npy', X_test)
            np.save('data/processed/y_train.npy', y_train)
            np.save('data/processed/y_test.npy', y_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

if __name__ == "__main__":
    # Test preprocessing
    raw_data = pd.read_csv('data/raw/champions_league_data.csv')
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(raw_data)
