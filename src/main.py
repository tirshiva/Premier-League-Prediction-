import logging
from data.data_ingestion import fetch_champions_league_data
from data.data_preprocessing import DataPreprocessor
from models.train_model import ChampionsLeagueModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Run the complete ML pipeline
    """
    try:
        # 1. Data Ingestion
        logger.info("Starting data ingestion...")
        df = fetch_champions_league_data()
        logger.info("Data ingestion completed")

        # 2. Data Preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
        logger.info("Data preprocessing completed")

        # 3. Model Training and Evaluation
        logger.info("Starting model training...")
        model = ChampionsLeagueModel()
        train_mse, train_r2 = model.train(X_train, y_train)
        logger.info(f"Training completed. MSE: {train_mse:.4f}, R2: {train_r2:.4f}")

        # 4. Model Evaluation
        logger.info("Starting model evaluation...")
        test_mse, test_r2 = model.evaluate(X_test, y_test)
        logger.info(f"Evaluation completed. Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

        return {
            "training_mse": train_mse,
            "training_r2": train_r2,
            "test_mse": test_mse,
            "test_r2": test_r2
        }

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
