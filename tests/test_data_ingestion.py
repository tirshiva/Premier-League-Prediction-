import pytest
import os
import pandas as pd
from src.data.data_ingestion import fetch_champions_league_data
import kagglehub

class TestDataIngestion:
    def test_kaggle_dataset_download(self):
        """Test Kaggle dataset download functionality"""
        try:
            path = kagglehub.dataset_download("fardifaalam170041060/champions-league-dataset-1955-2023")
            assert path is not None
            assert len(path) > 0
            assert os.path.exists(path[0])
        except Exception as e:
            pytest.fail(f"Kaggle dataset download failed: {str(e)}")

    def test_data_loading(self):
        """Test data loading and validation"""
        try:
            df = fetch_champions_league_data()
            
            # Check if DataFrame is created successfully
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            
            # Check required columns
            required_columns = ['date', 'home_score', 'away_score']
            for col in required_columns:
                assert col in df.columns
                
            # Check data types
            assert pd.to_datetime(df['date'], errors='coerce').notnull().all()
            assert df['home_score'].dtype in ['int64', 'float64']
            assert df['away_score'].dtype in ['int64', 'float64']
            
        except Exception as e:
            pytest.fail(f"Data loading and validation failed: {str(e)}")

    def test_data_saving(self):
        """Test if data is saved correctly"""
        try:
            # Fetch data
            df = fetch_champions_league_data()
            
            # Check if raw data directory exists
            assert os.path.exists('data/raw')
            
            # Check if data file was saved
            assert os.path.exists('data/raw/champions_league_data.csv')
            
            # Verify saved data
            saved_df = pd.read_csv('data/raw/champions_league_data.csv')
            assert saved_df.equals(df)
            
        except Exception as e:
            pytest.fail(f"Data saving test failed: {str(e)}")

    def test_error_handling(self):
        """Test error handling in data ingestion"""
        # Test with invalid dataset name
        with pytest.raises(Exception):
            kagglehub.dataset_download("invalid/dataset")

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after tests"""
        yield
        # Clean up test data
        if os.path.exists('data/raw/champions_league_data.csv'):
            os.remove('data/raw/champions_league_data.csv')

if __name__ == "__main__":
    pytest.main([__file__])
