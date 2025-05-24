import pytest
import os
import pandas as pd
import mlflow
from src.data.data_ingestion import fetch_champions_league_data
from src.data.data_preprocessing import DataPreprocessor
from src.models.train_model import ChampionsLeagueModel
from src.main import run_pipeline

class TestIntegration:
    @pytest.fixture(scope="class")
    def setup_test_data(self):
        """Create test data directory and sample data"""
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        # Create sample data
        df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100),
            'home_score': range(100),
            'away_score': range(100),
            'home_team': ['Team A'] * 100,
            'away_team': ['Team B'] * 100
        })
        df.to_csv('data/raw/champions_league_data.csv', index=False)
        return df

    def test_full_pipeline_execution(self, setup_test_data):
        """Test the complete pipeline execution"""
        try:
            # Run the complete pipeline
            results = run_pipeline()
            
            # Check if results contain expected metrics
            assert 'training_mse' in results
            assert 'training_r2' in results
            assert 'test_mse' in results
            assert 'test_r2' in results
            
            # Check if processed files were created
            assert os.path.exists('data/processed/X_train.npy')
            assert os.path.exists('data/processed/X_test.npy')
            assert os.path.exists('data/processed/y_train.npy')
            assert os.path.exists('data/processed/y_test.npy')
            
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {str(e)}")

    def test_mlflow_tracking(self, setup_test_data):
        """Test MLflow experiment tracking"""
        # Load test data
        df = pd.read_csv('data/raw/champions_league_data.csv')
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
        
        # Train model with MLflow tracking
        model = ChampionsLeagueModel()
        with mlflow.start_run() as run:
            train_mse, train_r2 = model.train(X_train, y_train)
            
            # Verify MLflow tracking
            run_id = run.info.run_id
            
            # Check if metrics were logged
            metrics = mlflow.get_run(run_id).data.metrics
            assert 'training_mse' in metrics
            assert 'training_r2' in metrics
            
            # Check if model was logged
            artifacts = mlflow.get_run(run_id).data.tags
            assert 'mlflow.log-model.history' in artifacts

    def test_data_flow(self, setup_test_data):
        """Test data flow between components"""
        # Test data ingestion
        df = fetch_champions_league_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Test preprocessing flow
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
        
        # Verify data shapes and types
        assert X_train.shape[1] == X_test.shape[1]  # Same number of features
        assert len(y_train.shape) == 1  # Target is 1D
        assert len(y_test.shape) == 1
        
        # Test model flow
        model = ChampionsLeagueModel()
        train_mse, train_r2 = model.train(X_train, y_train)
        test_mse, test_r2 = model.evaluate(X_test, y_test)
        
        # Verify metrics
        assert isinstance(train_mse, float)
        assert isinstance(train_r2, float)
        assert isinstance(test_mse, float)
        assert isinstance(test_r2, float)

    def test_error_handling(self, setup_test_data):
        """Test error handling in pipeline components"""
        # Test with invalid data
        invalid_df = pd.DataFrame()
        
        # Test preprocessing with invalid data
        preprocessor = DataPreprocessor()
        with pytest.raises(Exception):
            preprocessor.preprocess_data(invalid_df)
        
        # Test model with invalid data
        model = ChampionsLeagueModel()
        with pytest.raises(Exception):
            model.train(np.array([]), np.array([]))

if __name__ == "__main__":
    pytest.main([__file__])
