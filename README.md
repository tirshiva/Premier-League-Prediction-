# Premier League Match Predictor

## 🏟️ Overview
This project is a web application that uses machine learning to predict the outcome of Premier League matches. The application allows users to select two teams and view detailed statistics and predictions for the match.

## 🚀 Features
- **Match Prediction**: Uses a trained machine learning model to predict match outcomes based on selected teams.
- **Team Statistics**: Displays team stats including:
  - Matches played
  - Total points
  - Goals scored & conceded
  - Goal difference
  - Win rate
  - Points per game
- **User Interface**: Built with **Streamlit**, making it easy to select teams and view predictions.

## 🔧 Technical Details
- **Backend**: Built using **FastAPI** to provide a RESTful API.
- **Machine Learning**: 
  - Trained model (`MatchPredictor`) built using `scikit-learn`
  - Trained and managed with `mlflow`
- **Data Source**: Historical Premier League data fetched using the **Kaggle API**
- **Logging**: Integrated for debugging and error tracking

## 📦 Requirements
- Python 3.8+
- FastAPI
- mlflow
- scikit-learn
- pandas
- numpy
- Streamlit
- Kaggle API

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/premier-league-match-predictor.git
   cd premier-league-match-predictor
