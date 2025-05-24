import pytest
import pandas as pd
import numpy as np
from src.data.data_preprocessing import DataPreprocessor
from src.models.train_model import ChampionsLeagueModel

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'date': ['2022-01-01', '2022-02-01', '2023-01-01'],
        'home_score': [2, 1, 3],
        'away_score': [1, 1, 2],
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team D', 'Team E', 'Team F']
    })

def test_data_preprocessor(sample_data):
    """Test data preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(sample_data)
    
    # Check shapes
    assert X_train.shape[1] == 1  # Number of features
    assert len(y_train.shape) == 1  # Target is 1D
    assert len(X_train) + len(X_test) == len(sample_data)  # Split sizes sum to total

def test_model_training():
    """Test model training"""
    # Create dummy data
    X_train = np.array([[2020], [2021], [2022]])
    y_train = np.array([1, 2, 3])
    
    model = ChampionsLeagueModel()
    train_mse, train_r2 = model.train(X_train, y_train)
    
    # Basic checks
    assert isinstance(train_mse, float)
    assert isinstance(train_r2, float)
    assert 0 <= train_r2 <= 1  # R2 should be between 0 and 1

def test_model_evaluation():
    """Test model evaluation"""
    # Create dummy data
    X_train = np.array([[2020], [2021], [2022]])
    y_train = np.array([1, 2, 3])
    X_test = np.array([[2023]])
    y_test = np.array([4])
    
    model = ChampionsLeagueModel()
    model.train(X_train, y_train)
    test_mse, test_r2 = model.evaluate(X_test, y_test)
    
    # Basic checks
    assert isinstance(test_mse, float)
    assert isinstance(test_r2, float)

if __name__ == "__main__":
    pytest.main([__file__])
