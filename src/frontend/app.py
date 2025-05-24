import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
import sys # Add sys import

# Adjust import path by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_model import MatchPredictor # Adjusted import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Title and headers */
    .stTitle {
        font-size: 2.5rem !important;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 1.5rem !important;
        font-weight: 600;
    }
    
    /* Documentation section */
    .docs-section {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #4CAF50;
        color: #ffffff;
    }
    
    /* Team selection area */
    .team-selection {
        background-color: #2d2d2d;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 20px 0;
        color: #ffffff;
    }
    
    /* Prediction results */
    .prediction-card {
        background-color: #2d2d2d;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 15px 0;
        text-align: center;
        border-top: 5px solid #4CAF50;
        color: #ffffff;
    }
    
    /* Metric values */
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #4CAF50;
    }
    
    /* Team stats */
    .team-stats {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        color: #ffffff;
    }

    /* Override Streamlit's default white background */
    .stApp {
        background-color: #1a1a1a;
    }

    /* Style for paragraphs and text */
    p, li, div {
        color: #ffffff;
    }

    /* Links */
    a {
        color: #4CAF50 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: #ffffff;
        border: none;
    }

    /* Selectbox */
    .stSelectbox {
        background-color: #2d2d2d;
        color: #ffffff;
    }

    /* Custom text colors */
    .text-success {
        color: #4CAF50 !important;
    }
    
    .text-muted {
        color: #a0a0a0 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource # Cache the model and data loading
def load_predictor_and_data():
    """Load the trained model and team statistics"""
    try:
        # Define paths relative to this script (src/frontend/app.py)
        base_path = Path(__file__).resolve().parent.parent.parent # Project root
        team_stats_path = base_path / "data" / "processed" / "team_stats.csv"
        scaler_mean_path = base_path / "models" / "scaler_mean.npy"
        scaler_scale_path = base_path / "models" / "scaler_scale.npy"

        # Load team statistics
        team_stats_df = pd.read_csv(team_stats_path)
        
        # Initialize and load model
        predictor = MatchPredictor()
        
        # Load scaler parameters
        if scaler_mean_path.exists() and scaler_scale_path.exists():
            predictor.scaler.mean_ = np.load(scaler_mean_path)
            predictor.scaler.scale_ = np.load(scaler_scale_path)
        else:
            st.error("Scaler files not found. Please ensure 'scaler_mean.npy' and 'scaler_scale.npy' are in the 'models' directory.")
            return None, None

        # Load the model from the latest MLflow run
        # Ensure MLFLOW_TRACKING_URI is set if not using local ./mlruns
        # For Streamlit Cloud, you might need to package the model differently or use a remote tracking server.
        # For simplicity here, we assume ./mlruns is available.
        mlflow_tracking_uri = (base_path / "mlruns").as_uri() # Corrected indentation
        mlflow.set_tracking_uri(mlflow_tracking_uri) # Corrected indentation
        
        runs = mlflow.search_runs(experiment_ids=["0"], order_by=["start_time DESC"]) # Assuming experiment ID is "0" # Corrected indentation
        if not runs.empty: # Corrected indentation
            latest_run_id = runs.iloc[0].run_id # Corrected indentation
            model_uri = f"runs:/{latest_run_id}/model" # Corrected indentation
            try: # Corrected indentation
                predictor.model = mlflow.sklearn.load_model(model_uri) # Corrected indentation
                logger.info(f"Model loaded successfully from run ID: {latest_run_id}") # Corrected indentation
            except Exception as e: # Corrected indentation
                st.error(f"Error loading MLflow model from {model_uri}: {str(e)}") # Corrected indentation
                logger.error(f"Error loading MLflow model: {str(e)}") # Corrected indentation
                return None, None # Corrected indentation
        else: # Corrected indentation
            st.error("No MLflow runs found. Cannot load model.") # Corrected indentation
            logger.error("No MLflow runs found.") # Corrected indentation
            return None, None # Corrected indentation
            
        logger.info("Predictor and team stats loaded successfully") # Corrected indentation
        return predictor, team_stats_df # Corrected indentation
        
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        logger.error(f"Error loading model or data: {str(e)}")
        return None, None

predictor, team_stats_df = load_predictor_and_data()

# Documentation Section
st.markdown("""
    <div class="docs-section">
        <h2>📖 How to Use This Predictor</h2>
        <ol>
            <li><strong>Select Teams:</strong> Choose the home and away teams from the dropdown menus below.</li>
            <li><strong>View Team Stats:</strong> Check detailed team statistics in the sidebar by selecting a team.</li>
            <li><strong>Get Prediction:</strong> Click the "Predict Match" button to see the match prediction.</li>
            <li><strong>Analyze Results:</strong> Review the win probabilities, expected points, and prediction confidence.</li>
        </ol>
        <p><em>Note: Predictions are based on historical performance data and statistical analysis.</em></p>
    </div>
""", unsafe_allow_html=True)

# Main content - Team Selection
st.markdown("<h2 class='section-header'>🎯 Match Prediction</h2>", unsafe_allow_html=True)

if predictor and team_stats_df is not None:
    teams = sorted(team_stats_df['Team'].tolist())
    
    with st.container():
        st.markdown("<div class='team-selection'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p style='color: #1976d2; font-weight: bold;'>Home Team</p>", unsafe_allow_html=True)
            home_team = st.selectbox("Select Home Team", teams, key="home_team", label_visibility="collapsed")
            
        with col2:
            st.markdown("<p style='color: #1976d2; font-weight: bold;'>Away Team</p>", unsafe_allow_html=True)
            away_teams_list = [team for team in teams if team != home_team]
            away_team = st.selectbox("Select Away Team", away_teams_list, key="away_team", label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Prediction button
        if st.button("🎮 Generate Match Prediction", type="primary", use_container_width=True):
            if not home_team or not away_team:
                st.warning("Please select both home and away teams.")
            elif home_team == away_team:
                st.warning("Home and away teams cannot be the same.")
            else:
                try:
                    prediction = predictor.predict_match(home_team, away_team, team_stats_df)
                    
                    # Results section
                    st.markdown("<h3 class='section-header'>📊 Prediction Results</h3>", unsafe_allow_html=True)
                    
                    # Display predictions in columns
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    outcomes = [
                        (res_col1, home_team, "Home Win", prediction['home_win_prob']),
                        (res_col2, "Draw", "Draw", prediction['draw_prob']),
                        (res_col3, away_team, "Away Win", prediction['away_win_prob'])
                    ]
                    
                    for r_col, team_name_display, outcome_display, prob_display in outcomes:
                        with r_col:
                            st.markdown(
                                f"""
                                <div class='prediction-card'>
                                    <h4>{team_name_display}</h4>
                                    <div class='metric-value'>{prob_display:.1%}</div>
                                    <div style='color: #666;'>{outcome_display} Probability</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Analysis section
                    st.markdown("<h3 class='section-header'>📈 Match Analysis</h3>", unsafe_allow_html=True)
                    
                    an_col1, an_col2 = st.columns(2)
                    
                    with an_col1:
                        # Expected points
                        st.markdown( # Corrected indentation
                            f"""
                            <div class='team-stats'>
                                <h4 style='color: #1976d2;'>Expected Points</h4>
                                <p><strong>{home_team}:</strong> {prediction['home_points']:.2f}</p>
                                <p><strong>{away_team}:</strong> {prediction['away_points']:.2f}</p>
                            </div>
                            """, # Corrected indentation
                            unsafe_allow_html=True # Corrected indentation
                        ) # Corrected indentation
                        
                    with an_col2: # Corrected indentation
                        # Match Analysis
                        st.markdown( # Corrected indentation
                            f"""
                            <div class='team-stats'>
                                <h4 style='color: #1976d2;'>Match Analysis</h4>
                                <p>Based on our analysis:</p>
                            </div>
                            """, # Corrected indentation
                            unsafe_allow_html=True # Corrected indentation
                        ) # Corrected indentation
                        
                        # Beginner-friendly match insight explanation
                        home_prob = prediction['home_win_prob']
                        away_prob = prediction['away_win_prob']
                        draw_prob = prediction['draw_prob']
                        
                        if home_prob > 0.5:
                            insight = f"✓ {home_team} are more likely to win with a probability of {home_prob:.1%}."
                        elif away_prob > 0.5:
                            insight = f"✓ {away_team} are more likely to win with a probability of {away_prob:.1%}."
                        elif draw_prob > max(home_prob, away_prob):
                            insight = f"⚖️ The match is likely to end in a draw with a probability of {draw_prob:.1%}."
                        else:
                            insight = (
                                f"The match is very close with chances:\n"
                                f"• {home_team}: {home_prob:.1%}\n"
                                f"• Draw: {draw_prob:.1%}\n"
                                f"• {away_team}: {away_prob:.1%}"
                            )
                        
                        explanation = (
                            "These predictions are based on:\n"
                            "• Past team performances\n"
                            "• Goals scored and conceded\n"
                            "• Recent win rates\n"
                            "• Points per game\n\n"
                            "The higher the percentage, the more likely that outcome is to happen."
                        )
                        
                        st.markdown(
                            f"""
                            <div class='team-stats' style='margin-top: 10px;'>
                                <p style='font-size: 1.1rem; margin-bottom: 15px;'>{insight}</p>
                                <p style='color: #555; background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>{explanation}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Confidence gauge
                    max_prob = max(
                        prediction['home_win_prob'],
                        prediction['draw_prob'],
                        prediction['away_win_prob']
                    )
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=max_prob * 100,
                        title={'text': "Prediction Confidence", 'font': {'color': "#1976d2", 'size': 24}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "#1976d2"},
                            'bar': {'color': "#1976d2"},
                            'steps': [
                                {'range': [0, 33], 'color': "#ffebee"},
                                {'range': [33, 66], 'color': "#e3f2fd"},
                                {'range': [66, 100], 'color': "#e8f5e9"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    logger.error(f"Error generating prediction: {str(e)}")
else:
    st.error("Failed to load the prediction model or team data. Please check the logs for more details. The application cannot proceed.")

# Sidebar with team statistics
with st.sidebar:
    st.image("https://logos-world.net/wp-content/uploads/2020/06/Premier-League-Logo.png", width=250)
    st.markdown("<h3 class='section-header'>📊 Team Statistics</h3>", unsafe_allow_html=True)
    
    if predictor and team_stats_df is not None:
        try:
            # Use the already loaded teams list
            selected_team_sidebar = st.selectbox("Select a team to view detailed stats", teams, key="sidebar_team_select")
            
            if selected_team_sidebar:
                # Get stats directly from the DataFrame
                stats = team_stats_df[team_stats_df['Team'] == selected_team_sidebar].iloc[0].to_dict()
                
                st.markdown(
                    f"""
                    <div class='team-stats'>
                        <h4 style='color: #1976d2;'>{selected_team_sidebar}</h4>
                        <p><strong>Matches Played:</strong> {stats.get('TotalMatches', 'N/A')}</p>
                        <p><strong>Total Points:</strong> {stats.get('TotalPoints', 'N/A')}</p>
                        <p><strong>Goals Scored:</strong> {stats.get('GoalsScored', 'N/A')}</p>
                        <p><strong>Goals Conceded:</strong> {stats.get('GoalsConceded', 'N/A')}</p>
                        <p><strong>Goal Difference:</strong> {stats.get('GoalDifference', 'N/A')}</p>
                        <p><strong>Win Rate:</strong> {stats.get('WinRate', 0):.1%}</p>
                        <p><strong>Points per Game:</strong> {stats.get('PointsPerGame', 0):.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error loading team statistics for sidebar: {str(e)}")
            logger.error(f"Error loading team statistics for sidebar: {str(e)}")
    else:
        st.warning("Team data not available for sidebar.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666;'>Powered by Machine Learning & Historical Premier League Data</p>
        <p style='color: #666; font-size: 0.8rem;'>This predictor uses statistical analysis and machine learning to generate match predictions</p>
    </div>
""", unsafe_allow_html=True)
