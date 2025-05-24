import os
import kagglehub
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_premier_league_data():
    """
    Fetch Premier League dataset from Kaggle
    """
    try:
        # Download latest version of the dataset
        path = kagglehub.dataset_download("panaaaaa/english-premier-league-and-championship-full-dataset")
        logger.info(f"Dataset downloaded to path: {path}")
        
        # Find and load the CSV files
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise ValueError("No CSV files found in the downloaded dataset")
            
        # Load and combine all CSV files
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            # Filter for Premier League matches only
            premier_league_data = df[df['League'] == 'Premier League']
            if not premier_league_data.empty:
                dfs.append(premier_league_data)
            
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded dataset with shape: {combined_df.shape}")
        
        # Save raw data
        os.makedirs('data/raw', exist_ok=True)
        raw_path = 'data/raw/premier_league_data.csv'
        combined_df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw data to: {raw_path}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def preprocess_match_data(df):
    """
    Preprocess the match data for prediction model
    """
    try:
        # Extract relevant features
        features = [
            'HomeTeam', 'AwayTeam',
            'FTH Goals', 'FTA Goals',  # Full Time Goals
            'HTH Goals', 'HTA Goals',  # Half Time Goals
            'H Shots', 'A Shots',  # Shots
            'H SOT', 'A SOT',  # Shots on Target
            'H Fouls', 'A Fouls',  # Fouls
            'H Corners', 'A Corners',  # Corners
            'H Yellow', 'A Yellow',  # Yellow Cards
            'H Red', 'A Red'  # Red Cards
        ]
        
        # Select only the features that exist in the dataset
        available_features = [f for f in features if f in df.columns]
        match_data = df[available_features].copy()
        
        # Handle missing values
        match_data = match_data.dropna()
        
        # Create target variables based on full-time goals
        match_data['HomePoints'] = match_data.apply(
            lambda x: 3 if x['FTH Goals'] > x['FTA Goals'] 
                     else (1 if x['FTH Goals'] == x['FTA Goals'] else 0),
            axis=1
        )
        match_data['AwayPoints'] = match_data.apply(
            lambda x: 3 if x['FTA Goals'] > x['FTH Goals']
                     else (1 if x['FTA Goals'] == x['FTH Goals'] else 0),
            axis=1
        )
        
        # Calculate team performance metrics
        team_stats = calculate_team_stats(match_data)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        match_data.to_csv('data/processed/match_data.csv', index=False)
        team_stats.to_csv('data/processed/team_stats.csv', index=False)
        
        logger.info(f"Processed match data shape: {match_data.shape}")
        logger.info(f"Team stats shape: {team_stats.shape}")
        
        return match_data, team_stats
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def calculate_team_stats(match_data):
    """
    Calculate team statistics from match data
    """
    team_stats = pd.DataFrame()
    
    # Get unique teams
    teams = pd.unique(match_data[['HomeTeam', 'AwayTeam']].values.ravel())
    
    stats = []
    for team in teams:
        # Home matches
        home_matches = match_data[match_data['HomeTeam'] == team]
        # Away matches
        away_matches = match_data[match_data['AwayTeam'] == team]
        
        # Calculate statistics
        total_matches = len(home_matches) + len(away_matches)
        total_goals_scored = home_matches['FTH Goals'].sum() + away_matches['FTA Goals'].sum()
        total_goals_conceded = home_matches['FTA Goals'].sum() + away_matches['FTH Goals'].sum()
        
        home_points = home_matches['HomePoints'].sum()
        away_points = away_matches['AwayPoints'].sum()
        total_points = home_points + away_points
        
        # Calculate win rate
        home_wins = len(home_matches[home_matches['FTH Goals'] > home_matches['FTA Goals']])
        away_wins = len(away_matches[away_matches['FTA Goals'] > away_matches['FTH Goals']])
        win_rate = (home_wins + away_wins) / total_matches if total_matches > 0 else 0
        
        stats.append({
            'Team': team,
            'TotalMatches': total_matches,
            'GoalsScored': total_goals_scored,
            'GoalsConceded': total_goals_conceded,
            'GoalDifference': total_goals_scored - total_goals_conceded,
            'TotalPoints': total_points,
            'WinRate': win_rate,
            'PointsPerGame': total_points / total_matches if total_matches > 0 else 0
        })
    
    team_stats = pd.DataFrame(stats)
    return team_stats

if __name__ == "__main__":
    # Test data ingestion and preprocessing
    df = fetch_premier_league_data()
    match_data, team_stats = preprocess_match_data(df)
    print("Data ingestion and preprocessing completed successfully!")
