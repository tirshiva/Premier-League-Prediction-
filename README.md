# Champions League Score Prediction MLOps Project

This project implements a machine learning pipeline for predicting scores in Champions League matches using historical data from 1955-2023.

## Project Structure
```
├── data/
│   ├── raw/         # Raw data downloaded from Kaggle
│   └── processed/   # Processed data for training
├── src/
│   ├── api/         # FastAPI service
│   ├── data/        # Data ingestion and preprocessing
│   ├── models/      # Model training and evaluation
│   └── main.py      # Main pipeline orchestration
├── requirements.txt # Project dependencies
└── README.md
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

To run the complete ML pipeline (data ingestion, preprocessing, training, and evaluation):

```bash
python src/main.py
```

### Starting the API Service

To start the prediction API:

```bash
uvicorn src.api.main:app --reload
```

### API Endpoints

- `POST /predict`: Make predictions
  - Input: JSON with features array
  - Returns: Predicted score

- `GET /health`: Health check endpoint

### MLflow Tracking

The project uses MLflow to track experiments. To view the MLflow UI:

```bash
mlflow ui
```

Then visit `http://localhost:5000`

## Features

- Data ingestion from Kaggle API
- Automated data preprocessing pipeline
- Model training with MLflow experiment tracking
- Model serving via FastAPI
- Logging and error handling
- Scalable project structure

## Model Details

The current implementation uses Linear Regression to predict match scores based on historical data. Features include:
- Year of the match
- (Additional features can be added in the data preprocessing step)

## Future Improvements

1. Add more sophisticated models (e.g., XGBoost, LightGBM)
2. Implement feature importance analysis
3. Add CI/CD pipeline
4. Add model versioning
5. Implement A/B testing
6. Add data validation
7. Implement model monitoring

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
