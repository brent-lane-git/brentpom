# config.py
import os

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of config.py
DATA_DIR = os.path.join(BASE_DIR, "data")

JSON_FILE_NAME = "export_2085.json" # Replace
STATS_CSV_FILE_NAME = "2085.csv" # Replace
# RECRUITING_CSV_FILE_NAME = "your_recruiting_data.csv" # For later

JSON_FILE_PATH = os.path.join(DATA_DIR, JSON_FILE_NAME)
STATS_CSV_FILE_PATH = os.path.join(DATA_DIR, STATS_CSV_FILE_NAME)
# RECRUITING_CSV_PATH = os.path.join(DATA_DIR, RECRUITING_CSV_FILE_NAME)
# --- Recruiting Configuration ---
RECRUITING_CSV_FILE_NAME = "recruiting_2084.csv" # REPLACE with your actual file name
# Path will be constructed in main_processor.py or data_loader.py if dedicated loader
# For simplicity, we can add it here if main_processor handles the loading directly for now.
# RECRUITING_CSV_FILE_PATH = os.path.join(DATA_DIR, RECRUITING_CSV_FILE_NAME) # This line can be in main_processor.py
COACH_CSV_FILE_NAME = "coaches_2085.csv"
# Star rating definitions (based on numeric rank for High School recruits)
# Rank: (min_rank, max_rank) -> stars
STAR_DEFINITIONS = {
    5: (1, 30),
    4: (31, 175),
    3: (176, 300),
    2: (301, 400),
    1: (401, 510) # HS Ranks 401-500 are 1-star, >500 are effectively 0-star or unranked for stars
}
# Max numeric rank to consider for star assignment (ranks above this are "unranked" for stars)
MAX_HS_RANK_FOR_STARS = 510

# NSPN Points
NSPN_POINTS = {
    "5_star": 10,
    "GT": 8,
    "4_star": 6,
    "JUCO": 4,
    "3_star": 3,
    "CPR": 2,
    "2_star": 1,
    "1_star": 0,
    "HS_unranked": 0, # For HS recruits outside star ranges
    "Unknown": 0      # Default for unknown types
}
KTV_DESIRED_MEAN = 100.0
KTV_DESIRED_STD_DEV = 15.0 
# Storms.com Formula Parameters
STORMS_OVR_BASELINE = 50
STORMS_OVR_POWER = 1.5 # (OVR - Baseline)^Power
STORMS_OVR_MAX_CONTRIBUTION = 40 # Max points from OVR portion to prevent one super high OVR dominating too much
STORMS_BONUS = {
    "5_star": 20, "4_star": 12, "3_star": 6, "2_star": 2, "1_star": 0, # HS Star bonus
    "GT": 10, "JUCO": 6, "CPR": 3, "HS_unranked": 0, "Unknown": 0   # Transfer type bonus
}

# 24/8 Sports Formula Parameters
TWENTYFOUR_EIGHT_HS_RANK_DECAY_FAST = 0.030 # For rank points: exp(-decay * (rank-1))
TWENTYFOUR_EIGHT_HS_RANK_DECAY_SLOW = 0.00005 # For rank points: exp(-decay_slow * (rank-1)^2)
TWENTYFOUR_EIGHT_TRANSFER_OVR_BASELINE = 40.0
TWENTYFOUR_EIGHT_TRANSFER_POINTS_SCALE = { # Multiplier for (TrueOVR_OnZ - Baseline)
    "GT": 1.5,
    "JUCO": 1.1,
    "CPR": 0.9
}
# For 24/8, we'll sum points. A direct quality score, not an average.
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
MAX_PREDICTED_SCORE_DIFFERENCE = 50.0
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
TARGET_SEASON_FOR_STATS = 2085
TARGET_RECRUITING_CLASS_YEAR = 2084
TARGET_COACH_ASSIGNMENT_YEAR = 2085
POSTSEASON_RESULTS_CSV_FILE_NAME = "postseason_2085.csv"
POSTSEASON_CSV_COLS = ['Event', 'Seed', 'Team', 'Result']
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

