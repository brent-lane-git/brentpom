# config.py
import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of config.py
DATA_DIR = os.path.join(BASE_DIR, "data")

JSON_FILE_NAME = "2085.json" # Replace
STATS_CSV_FILE_NAME = "2085.csv" # Replace
# RECRUITING_CSV_FILE_NAME = "your_recruiting_data.csv" # For later

JSON_FILE_PATH = os.path.join(DATA_DIR, JSON_FILE_NAME)
STATS_CSV_FILE_PATH = os.path.join(DATA_DIR, STATS_CSV_FILE_NAME)
# RECRUITING_CSV_PATH = os.path.join(DATA_DIR, RECRUITING_CSV_FILE_NAME)

# --- Calculation Constants ---
POSSESSION_FTA_COEFFICIENT = 0.44
LUCK_PYTHAGOREAN_EXPONENT = 6
RPI_WEIGHT_WIN_PERCENTAGE = 0.25
RPI_WEIGHT_OPPONENT_WIN_PERCENTAGE = 0.50
RPI_WEIGHT_OPPONENT_OPPONENT_WIN_PERCENTAGE = 0.25
RPI_LOCATION_WEIGHT_HOME_WIN = 0.6
RPI_LOCATION_WEIGHT_HOME_LOSS = 1.4 # Symmetrical: a home loss is as bad as an away win is good
RPI_LOCATION_WEIGHT_AWAY_WIN = 1.4
RPI_LOCATION_WEIGHT_AWAY_LOSS = 0.6 # Symmetrical
RPI_LOCATION_WEIGHT_NEUTRAL = 1.0
HOME_COURT_ADVANTAGE_POINTS = 1.5 # Points per 100 poss added to home team's expected OE
NUM_ADJUSTMENT_ITERATIONS = 100 # Number of iterations for the adjustment algorithm
MAX_TEAMS_FOR_RANKING = 128 # Typical D1 teams, adjust as needed for your league size
# --- WAB Configuration ---
# Defines the rank (by adjEM, higher adjEM = better rank) of a typical "bubble" team.
# Adjust this based on your league's typical tournament qualification spots.
BUBBLE_TEAM_RANK_THRESHOLD = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE_PATH = os.path.join(BASE_DIR, "data", "brentpom.db") 

# Scaling factor for converting adjEM difference to win probability.
# A difference of X in adjEM, when scaled, determines win likelihood.
# Roughly, an adjEM difference of 10 points often corresponds to ~85% win prob.
# 1 / (1 + exp(-Margin * C)) = WinProb. If Margin=10, WinProb=0.85 -> C ~ 0.173
WIN_PROB_SCALING_FACTOR = 0.173
# --- Quadrant Definitions (User Provided) ---
# Ranks are inclusive. Opponent's rank is based on our adjEM.
# Higher adjEM = better rank (Rank 1 is best).
QUADRANT_DEFINITIONS = {
    "Q1": {
        "home_rank_range": (1, 15),
        "neutral_rank_range": (1, 20),
        "away_rank_range": (1, 25)
    },
    "Q2": {
        "home_rank_range": (16, 39),
        "neutral_rank_range": (21, 46),
        "away_rank_range": (26, 53)
    },
    "Q3": {
        "home_rank_range": (40, 74),
        "neutral_rank_range": (47, 82),
        "away_rank_range": (54, 90)
    },
    "Q4": {
        "home_rank_range": (75, MAX_TEAMS_FOR_RANKING),
        "neutral_rank_range": (83, MAX_TEAMS_FOR_RANKING),
        "away_rank_range": (91, MAX_TEAMS_FOR_RANKING)
    }
}

