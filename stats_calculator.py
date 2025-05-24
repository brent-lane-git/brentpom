# stats_calculator.py
import pandas as pd
import numpy as np
import config

# --- Helper function for RPI WP calculation ---
def _calculate_team_rpi_wp(team_tid_to_calc, season_of_calc, all_games_df, excluding_opponent_tid=None):
    # ... (content from previous version - confirmed working) ...
    team_games = all_games_df[
        (all_games_df['team_tid'] == team_tid_to_calc) &
        (all_games_df['season'] == season_of_calc)
    ].copy()
    if excluding_opponent_tid is not None:
        team_games = team_games[team_games['opponent_tid'] != excluding_opponent_tid]
    if team_games.empty: return 0.0
    team_games.loc[:, 'team_score_official_num'] = pd.to_numeric(team_games['team_score_official'], errors='coerce').fillna(0)
    team_games.loc[:, 'opponent_score_official_num'] = pd.to_numeric(team_games['opponent_score_official'], errors='coerce').fillna(0)
    team_games.loc[:, 'is_win_for_calc'] = (team_games['team_score_official_num'] > team_games['opponent_score_official_num']).astype(int)
    weighted_wins = 0.0
    for _, row in team_games.iterrows():
        if row['is_win_for_calc'] == 1:
            if row['location'] == 'Home': weighted_wins += config.RPI_LOCATION_WEIGHT_HOME_WIN
            elif row['location'] == 'Away': weighted_wins += config.RPI_LOCATION_WEIGHT_AWAY_WIN
            else: weighted_wins += config.RPI_LOCATION_WEIGHT_NEUTRAL
    actual_games_played = len(team_games)
    return (weighted_wins / actual_games_played) if actual_games_played > 0 else 0.0

# --- Main Calculation Functions ---
def calculate_game_possessions_oe_de(game_team_stats_df):
    # ... (content from previous version - confirmed working) ...
    if game_team_stats_df.empty: return game_team_stats_df
    df = game_team_stats_df.copy()
    poss_cols_team = ['team_fga', 'team_oreb', 'team_tov', 'team_fta']
    for col in poss_cols_team:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    poss_cols_opp = ['opponent_fga', 'opponent_oreb', 'opponent_tov', 'opponent_fta']
    for col in poss_cols_opp:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    oe_de_cols = ['team_pts', 'opponent_pts']
    for col in oe_de_cols:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['team_poss'] = (df['team_fga'] - df['team_oreb'] + df['team_tov'] +
                       (config.POSSESSION_FTA_COEFFICIENT * df['team_fta']))
    df['opp_poss'] = (df['opponent_fga'] - df['opponent_oreb'] + df['opponent_tov'] +
                      (config.POSSESSION_FTA_COEFFICIENT * df['opponent_fta']))
    df['raw_oe'] = np.divide(df['team_pts'] * 100, df['team_poss'], out=np.zeros_like(df['team_pts'], dtype=float), where=df['team_poss']!=0)
    df['raw_de'] = np.divide(df['opponent_pts'] * 100, df['opp_poss'], out=np.zeros_like(df['opponent_pts'], dtype=float), where=df['opp_poss']!=0)
    print("Calculated per-game possessions, raw OE, and raw DE.")
    return df

def calculate_game_four_factors_and_shooting(game_team_stats_df):
    # ... (content from previous version - confirmed working) ...
    if game_team_stats_df.empty: return game_team_stats_df
    if 'team_poss' not in game_team_stats_df.columns or 'opp_poss' not in game_team_stats_df.columns:
        print("ERROR: 'team_poss' or 'opp_poss' columns are missing for Four Factors.")
        return game_team_stats_df
    df = game_team_stats_df.copy()
    ff_stat_cols = ['team_fgm', 'team_fgm3', 'team_fga', 'team_tov', 'team_oreb', 'opponent_dreb', 'team_fta',
                    'opponent_fgm', 'opponent_fgm3', 'opponent_fga', 'opponent_tov', 'opponent_oreb', 'team_dreb', 'opponent_fta',
                    'team_fga3', 'opponent_fga3']
    for col in ff_stat_cols:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['off_efg_pct'] = np.divide((df['team_fgm'] + 0.5 * df['team_fgm3']), df['team_fga'], out=np.zeros_like(df['team_fgm'], dtype=float), where=df['team_fga']!=0)
    df['off_tov_pct'] = np.divide(df['team_tov'], df['team_poss'], out=np.zeros_like(df['team_tov'], dtype=float), where=df['team_poss']!=0)
    df['off_orb_pct'] = np.divide(df['team_oreb'], (df['team_oreb'] + df['opponent_dreb']), out=np.zeros_like(df['team_oreb'], dtype=float), where=(df['team_oreb'] + df['opponent_dreb'])!=0)
    df['off_ft_rate'] = np.divide(df['team_fta'], df['team_fga'], out=np.zeros_like(df['team_fta'], dtype=float), where=df['team_fga']!=0)
    df['def_efg_pct'] = np.divide((df['opponent_fgm'] + 0.5 * df['opponent_fgm3']), df['opponent_fga'], out=np.zeros_like(df['opponent_fgm'], dtype=float), where=df['opponent_fga']!=0)
    df['def_tov_pct'] = np.divide(df['opponent_tov'], df['opp_poss'], out=np.zeros_like(df['opponent_tov'], dtype=float), where=df['opp_poss']!=0)
    df['def_opp_orb_pct'] = np.divide(df['opponent_oreb'], (df['opponent_oreb'] + df['team_dreb']), out=np.zeros_like(df['opponent_oreb'], dtype=float), where=(df['opponent_oreb'] + df['team_dreb'])!=0)
    df['def_ft_rate'] = np.divide(df['opponent_fta'], df['opponent_fga'], out=np.zeros_like(df['opponent_fta'], dtype=float), where=df['opponent_fga']!=0)
    df['team_drb_pct'] = np.divide(df['team_dreb'], (df['team_dreb'] + df['opponent_oreb']), out=np.zeros_like(df['team_dreb'], dtype=float), where=(df['team_dreb'] + df['opponent_oreb'])!=0)
    team_fga2 = df['team_fga'] - df['team_fga3']; team_fgm2 = df['team_fgm'] - df['team_fgm3']
    df['team_2p_pct'] = np.divide(team_fgm2, team_fga2, out=np.zeros_like(team_fgm2, dtype=float), where=team_fga2!=0)
    opp_fga2 = df['opponent_fga'] - df['opponent_fga3']; opp_fgm2 = df['opponent_fgm'] - df['opponent_fgm3']
    df['opp_2p_pct'] = np.divide(opp_fgm2, opp_fga2, out=np.zeros_like(opp_fgm2, dtype=float), where=opp_fga2!=0)
    df['team_3p_pct'] = np.divide(df['team_fgm3'], df['team_fga3'], out=np.zeros_like(df['team_fgm3'], dtype=float), where=df['team_fga3']!=0)
    df['opp_3p_pct'] = np.divide(df['opponent_fgm3'], df['opponent_fga3'], out=np.zeros_like(df['opponent_fgm3'], dtype=float), where=df['opponent_fga3']!=0)
    df['team_3p_rate'] = np.divide(df['team_fga3'], df['team_fga'], out=np.zeros_like(df['team_fga3'], dtype=float), where=df['team_fga']!=0)
    df['opp_3p_rate'] = np.divide(df['opponent_fga3'], df['opponent_fga'], out=np.zeros_like(df['opponent_fga3'], dtype=float), where=df['opponent_fga']!=0)
    print("Calculated per-game Four Factors and shooting percentages.")
    return df

def calculate_season_team_aggregates(game_team_stats_df):
    # ... (Exact same as the version you confirmed working) ...
    if game_team_stats_df.empty: return pd.DataFrame()
    df_for_agg = game_team_stats_df.copy()
    if 'win' not in df_for_agg.columns or 'loss' not in df_for_agg.columns:
        df_for_agg.loc[:, 'team_score_official_num'] = pd.to_numeric(df_for_agg['team_score_official'], errors='coerce').fillna(0)
        df_for_agg.loc[:, 'opponent_score_official_num'] = pd.to_numeric(df_for_agg['opponent_score_official'], errors='coerce').fillna(0)
        df_for_agg.loc[:, 'win'] = (df_for_agg['team_score_official_num'] > df_for_agg['opponent_score_official_num']).astype(int)
        df_for_agg.loc[:, 'loss'] = (df_for_agg['team_score_official_num'] < df_for_agg['opponent_score_official_num']).astype(int)
    aggregations = {'gid': 'count', 'win': 'sum', 'loss': 'sum','team_pts': 'sum', 'team_poss': 'sum', 'team_fgm': 'sum', 'team_fga': 'sum', 'team_fgm3': 'sum', 'team_fga3': 'sum', 'team_ftm': 'sum', 'team_fta': 'sum', 'team_oreb': 'sum', 'team_dreb': 'sum', 'team_tov': 'sum', 'team_ast': 'sum', 'team_stl': 'sum', 'team_blk': 'sum', 'team_pf': 'sum','opponent_pts': 'sum', 'opp_poss': 'sum', 'opponent_fgm': 'sum', 'opponent_fga': 'sum', 'opponent_fgm3': 'sum', 'opponent_fga3': 'sum','opponent_ftm': 'sum', 'opponent_fta': 'sum', 'opponent_oreb': 'sum', 'opponent_dreb': 'sum','opponent_tov': 'sum', 'opponent_ast': 'sum', 'opponent_stl': 'sum', 'opponent_blk': 'sum', 'opponent_pf': 'sum'}
    valid_aggregations = {k: v for k, v in aggregations.items() if k in df_for_agg.columns}
    grouping_cols = ['season', 'team_tid', 'team_abbrev']
    for col in grouping_cols:
        if col not in df_for_agg.columns: return pd.DataFrame()
    essential_agg_cols = ['gid', 'win', 'loss']
    for col in essential_agg_cols:
        if col not in valid_aggregations: return pd.DataFrame()
    season_summary_df = df_for_agg.groupby(grouping_cols, as_index=False).agg(valid_aggregations)
    season_summary_df.rename(columns={'gid': 'games_played', 'win': 'wins', 'loss': 'losses'}, inplace=True)
    season_summary_df['raw_oe'] = np.divide(season_summary_df.get('team_pts', 0) * 100, season_summary_df.get('team_poss', 0), out=np.zeros_like(season_summary_df.get('team_pts', 0), dtype=float), where=season_summary_df.get('team_poss', 0)!=0)
    season_summary_df['raw_de'] = np.divide(season_summary_df.get('opponent_pts', 0) * 100, season_summary_df.get('opp_poss', 0), out=np.zeros_like(season_summary_df.get('opponent_pts', 0), dtype=float), where=season_summary_df.get('opp_poss', 0)!=0)
    season_summary_df['raw_em'] = season_summary_df['raw_oe'] - season_summary_df['raw_de']
    season_summary_df['off_efg_pct'] = np.divide((season_summary_df.get('team_fgm', 0) + 0.5 * season_summary_df.get('team_fgm3', 0)), season_summary_df.get('team_fga', 0), out=np.zeros_like(season_summary_df.get('team_fgm', 0), dtype=float), where=season_summary_df.get('team_fga', 0)!=0)
    season_summary_df['off_tov_pct'] = np.divide(season_summary_df.get('team_tov', 0), season_summary_df.get('team_poss', 0), out=np.zeros_like(season_summary_df.get('team_tov', 0), dtype=float), where=season_summary_df.get('team_poss', 0)!=0)
    season_summary_df['off_orb_pct'] = np.divide(season_summary_df.get('team_oreb', 0), (season_summary_df.get('team_oreb', 0) + season_summary_df.get('opponent_dreb', 0)), out=np.zeros_like(season_summary_df.get('team_oreb', 0), dtype=float), where=(season_summary_df.get('team_oreb', 0) + season_summary_df.get('opponent_dreb', 0))!=0)
    season_summary_df['off_ft_rate'] = np.divide(season_summary_df.get('team_fta', 0), season_summary_df.get('team_fga', 0), out=np.zeros_like(season_summary_df.get('team_fta', 0), dtype=float), where=season_summary_df.get('team_fga', 0)!=0)
    season_summary_df['def_efg_pct'] = np.divide((season_summary_df.get('opponent_fgm', 0) + 0.5 * season_summary_df.get('opponent_fgm3', 0)), season_summary_df.get('opponent_fga', 0), out=np.zeros_like(season_summary_df.get('opponent_fgm', 0), dtype=float), where=season_summary_df.get('opponent_fga', 0)!=0)
    season_summary_df['def_tov_pct'] = np.divide(season_summary_df.get('opponent_tov', 0), season_summary_df.get('opp_poss', 0), out=np.zeros_like(season_summary_df.get('opponent_tov', 0), dtype=float), where=season_summary_df.get('opp_poss', 0)!=0)
    season_summary_df['def_opp_orb_pct'] = np.divide(season_summary_df.get('opponent_oreb', 0), (season_summary_df.get('opponent_oreb', 0) + season_summary_df.get('team_dreb', 0)), out=np.zeros_like(season_summary_df.get('opponent_oreb', 0), dtype=float), where=(season_summary_df.get('opponent_oreb', 0) + season_summary_df.get('team_dreb', 0))!=0)
    season_summary_df['def_ft_rate'] = np.divide(season_summary_df.get('opponent_fta', 0), season_summary_df.get('opponent_fga', 0), out=np.zeros_like(season_summary_df.get('opponent_fta', 0), dtype=float), where=season_summary_df.get('opponent_fga', 0)!=0)
    season_summary_df['team_drb_pct'] = np.divide(season_summary_df.get('team_dreb', 0), (season_summary_df.get('team_dreb', 0) + season_summary_df.get('opponent_oreb', 0)), out=np.zeros_like(season_summary_df.get('team_dreb', 0), dtype=float), where=(season_summary_df.get('team_dreb', 0) + season_summary_df.get('opponent_oreb', 0))!=0)
    s_team_fga2 = season_summary_df.get('team_fga', 0) - season_summary_df.get('team_fga3', 0); s_team_fgm2 = season_summary_df.get('team_fgm', 0) - season_summary_df.get('team_fgm3', 0)
    season_summary_df['team_2p_pct'] = np.divide(s_team_fgm2, s_team_fga2, out=np.zeros_like(s_team_fgm2, dtype=float), where=s_team_fga2!=0)
    s_opp_fga2 = season_summary_df.get('opponent_fga', 0) - season_summary_df.get('opponent_fga3', 0); s_opp_fgm2 = season_summary_df.get('opponent_fgm', 0) - season_summary_df.get('opponent_fgm3', 0)
    season_summary_df['opp_2p_pct'] = np.divide(s_opp_fgm2, s_opp_fga2, out=np.zeros_like(s_opp_fgm2, dtype=float), where=s_opp_fga2!=0)
    season_summary_df['team_3p_pct'] = np.divide(season_summary_df.get('team_fgm3', 0), season_summary_df.get('team_fga3', 0), out=np.zeros_like(season_summary_df.get('team_fgm3', 0), dtype=float), where=season_summary_df.get('team_fga3', 0)!=0)
    season_summary_df['opp_3p_pct'] = np.divide(season_summary_df.get('opponent_fgm3', 0), season_summary_df.get('opponent_fga3', 0), out=np.zeros_like(season_summary_df.get('opponent_fgm3', 0), dtype=float), where=season_summary_df.get('opponent_fga3', 0)!=0)
    season_summary_df['team_3p_rate'] = np.divide(season_summary_df.get('team_fga3', 0), season_summary_df.get('team_fga', 0), out=np.zeros_like(season_summary_df.get('team_fga3', 0), dtype=float), where=season_summary_df.get('team_fga', 0)!=0)
    season_summary_df['opp_3p_rate'] = np.divide(season_summary_df.get('opponent_fga3', 0), season_summary_df.get('opponent_fga', 0), out=np.zeros_like(season_summary_df.get('opponent_fga3', 0), dtype=float), where=season_summary_df.get('opponent_fga', 0)!=0)
    if 'wins' in season_summary_df.columns and 'games_played' in season_summary_df.columns:
        season_summary_df['win_pct'] = np.divide(season_summary_df['wins'], season_summary_df['games_played'], out=np.zeros_like(season_summary_df['wins'], dtype=float), where=season_summary_df['games_played']!=0)
    else: season_summary_df['win_pct'] = 0.0
    if 'team_poss' in season_summary_df.columns and 'games_played' in season_summary_df.columns:
        season_summary_df['avg_tempo'] = np.divide(season_summary_df['team_poss'], season_summary_df['games_played'], out=np.zeros_like(season_summary_df['team_poss'], dtype=float), where=season_summary_df['games_played']!=0)
    else: season_summary_df['avg_tempo'] = 0.0
    print("Calculated season-level aggregates and raw metrics for each team.")
    return season_summary_df

def calculate_luck(season_team_summary_df):
    """
    Calculates Luck for each team based on their ADJUSTED offensive/defensive efficiencies.
    Luck = Actual Wins - Expected Wins (from Pythagorean expectation using adj_o and adj_d).

    Args:
        season_team_summary_df (pd.DataFrame): DataFrame with season aggregates per team.
                                               Must include 'wins', 'games_played',
                                               'adj_o', 'adj_d'.
    
    Returns:
        pd.DataFrame: The input season_team_summary_df with 'expected_wins_adj' and 'luck_adj' columns.
    """
    if season_team_summary_df.empty:
        print("WARNING: Input season_team_summary_df is empty in calculate_luck (adjusted).")
        return season_team_summary_df

    df = season_team_summary_df.copy()

    # Now expects adj_o and adj_d
    required_cols = ['wins', 'games_played', 'adj_o', 'adj_d']
    all_cols_present = all(col in df.columns for col in required_cols)
    
    if not all_cols_present:
        missing = [col for col in required_cols if col not in df.columns]
        print(f"ERROR: Missing required columns for Adjusted Luck calculation: {missing}. Skipping Luck calculation.")
        df['expected_wins_adj'] = np.nan # Use new column names
        df['luck_adj'] = np.nan
        return df

    # Ensure adj_o and adj_d are float for power operation
    adj_o_float = df['adj_o'].astype(float)
    adj_d_float = df['adj_d'].astype(float)
    k = config.LUCK_PYTHAGOREAN_EXPONENT # 13.5

    # Calculate components, handling potential division by zero or 0^k issues
    oe_pow_k = np.power(adj_o_float, k)
    de_pow_k = np.power(adj_d_float, k)
    denominator = oe_pow_k + de_pow_k
    
    # Default to 0.5 win_pct if denominator is 0 (e.g., adj_o and adj_d are both 0)
    pyth_exp_win_pct_adj = np.divide(oe_pow_k, denominator,
                                     out=np.full_like(oe_pow_k, 0.5, dtype=float),
                                     where=denominator != 0)
    
    # Handle specific cases for 0 efficiency values
    pyth_exp_win_pct_adj = np.where((adj_o_float > 0) & (adj_d_float == 0), 1.0, pyth_exp_win_pct_adj)
    pyth_exp_win_pct_adj = np.where((adj_o_float == 0) & (adj_d_float > 0), 0.0, pyth_exp_win_pct_adj)

    df['expected_wins_adj'] = pyth_exp_win_pct_adj * df['games_played']
    df['luck_adj'] = df['wins'] - df['expected_wins_adj'] # Luck based on adjusted metrics
    
    # Optionally remove old luck columns if they exist and you only want the adjusted one
    # if 'expected_wins' in df.columns: df.drop(columns=['expected_wins'], inplace=True)
    # if 'luck' in df.columns: df.drop(columns=['luck'], inplace=True)
    
    print("Calculated Luck based on adjusted efficiencies.")
    return df

def calculate_rpi_sos(game_team_stats_df, season_team_summary_df):
    # ... (Exact same as the version you confirmed working) ...
    if game_team_stats_df.empty or season_team_summary_df.empty:
        print("WARNING: Empty input DataFrame(s) for RPI/SOS calculation.")
        if not season_team_summary_df.empty:
            for col_name in ['rpi_wp', 'owp', 'oowp', 'rpi', 'sos_bcs']: season_team_summary_df[col_name] = 0.0
        return season_team_summary_df
    print("Calculating RPI and SOS (BCS)...")
    required_game_cols = ['season', 'team_tid', 'opponent_tid', 'location', 'team_score_official', 'opponent_score_official', 'gid']
    if not all(col in game_team_stats_df.columns for col in required_game_cols):
        missing_req_game_cols = [col for col in required_game_cols if col not in game_team_stats_df.columns]
        print(f"ERROR: Missing required columns in game_team_stats_df for RPI/SOS: {missing_req_game_cols}.")
        if not season_team_summary_df.empty:
            for col_name in ['rpi_wp', 'owp', 'oowp', 'rpi', 'sos_bcs']: season_team_summary_df[col_name] = 0.0
        return season_team_summary_df
    rpi_wp_values = {}
    temp_rpi_wp_df = game_team_stats_df.copy()
    temp_rpi_wp_df.loc[:, 'team_score_official'] = pd.to_numeric(temp_rpi_wp_df['team_score_official'], errors='coerce').fillna(0)
    temp_rpi_wp_df.loc[:, 'opponent_score_official'] = pd.to_numeric(temp_rpi_wp_df['opponent_score_official'], errors='coerce').fillna(0)
    def get_rpi_weighted_win_value(row):
        is_win = row['team_score_official'] > row['opponent_score_official']
        if not is_win: return 0.0
        if row['location'] == 'Home': return config.RPI_LOCATION_WEIGHT_HOME_WIN
        if row['location'] == 'Away': return config.RPI_LOCATION_WEIGHT_AWAY_WIN
        return config.RPI_LOCATION_WEIGHT_NEUTRAL
    temp_rpi_wp_df.loc[:, 'rpi_weighted_win_value'] = temp_rpi_wp_df.apply(get_rpi_weighted_win_value, axis=1)
    rpi_wp_agg = temp_rpi_wp_df.groupby(['season', 'team_tid']).agg(total_rpi_weighted_wins=('rpi_weighted_win_value', 'sum'), games_played_for_rpi=('gid', 'count')).reset_index()
    rpi_wp_agg.loc[:, 'rpi_wp'] = np.divide(rpi_wp_agg['total_rpi_weighted_wins'], rpi_wp_agg['games_played_for_rpi'], out=np.zeros_like(rpi_wp_agg['total_rpi_weighted_wins'], dtype=float), where=rpi_wp_agg['games_played_for_rpi'] != 0)
    for _, row in rpi_wp_agg.iterrows(): rpi_wp_values[(row['season'], row['team_tid'])] = row['rpi_wp']
    owp_values = {}
    oowp_values = {}
    unique_teams_played = game_team_stats_df[['season', 'team_tid']].drop_duplicates()
    for _, team_row in unique_teams_played.iterrows():
        current_season = team_row['season']; team_a_tid = team_row['team_tid']
        opponents_of_team_a_tids = game_team_stats_df[(game_team_stats_df['team_tid'] == team_a_tid) & (game_team_stats_df['season'] == current_season)]['opponent_tid'].unique()
        if len(opponents_of_team_a_tids) == 0: owp_values[(current_season, team_a_tid)] = 0.0; continue
        sum_opponent_rpi_wp_excluding_team_a = 0.0
        for opponent_b_tid in opponents_of_team_a_tids:
            sum_opponent_rpi_wp_excluding_team_a += _calculate_team_rpi_wp(opponent_b_tid, current_season, game_team_stats_df, excluding_opponent_tid=team_a_tid)
        owp_values[(current_season, team_a_tid)] = np.divide(sum_opponent_rpi_wp_excluding_team_a, len(opponents_of_team_a_tids)) if len(opponents_of_team_a_tids) > 0 else 0.0
    for _, team_row in unique_teams_played.iterrows():
        current_season = team_row['season']; team_a_tid = team_row['team_tid']
        opponents_of_team_a_tids = game_team_stats_df[(game_team_stats_df['team_tid'] == team_a_tid) & (game_team_stats_df['season'] == current_season)]['opponent_tid'].unique()
        if len(opponents_of_team_a_tids) == 0: oowp_values[(current_season, team_a_tid)] = 0.0; continue
        sum_oowp_components = 0.0
        for opponent_b_tid in opponents_of_team_a_tids: sum_oowp_components += owp_values.get((current_season, opponent_b_tid), 0.0)
        oowp_values[(current_season, team_a_tid)] = np.divide(sum_oowp_components, len(opponents_of_team_a_tids)) if len(opponents_of_team_a_tids) > 0 else 0.0
    output_df = season_team_summary_df.copy()
    output_df.loc[:, 'rpi_wp'] = output_df.apply(lambda row: rpi_wp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'owp'] = output_df.apply(lambda row: owp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'oowp'] = output_df.apply(lambda row: oowp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'rpi'] = (output_df['rpi_wp'] * config.RPI_WEIGHT_WIN_PERCENTAGE + output_df['owp'] * config.RPI_WEIGHT_OPPONENT_WIN_PERCENTAGE + output_df['oowp'] * config.RPI_WEIGHT_OPPONENT_OPPONENT_WIN_PERCENTAGE)
    output_df.loc[:, 'sos_bcs'] = (output_df['owp'] * (2/3)) + (output_df['oowp'] * (1/3))
    print("Calculated RPI and SOS (BCS).")
    return output_df

def calculate_adjusted_efficiencies(game_team_stats_df, season_team_summary_df, config_params):
    # ... (Exact same as the version you confirmed working) ...
    if game_team_stats_df.empty:
        print("WARNING: game_team_stats_df is empty. Cannot calculate adjusted efficiencies.")
        if not season_team_summary_df.empty: season_team_summary_df['adj_o'] = np.nan; season_team_summary_df['adj_d'] = np.nan; season_team_summary_df['adj_em'] = np.nan
        return season_team_summary_df
    print("Calculating Adjusted Offensive and Defensive Efficiencies...")
    output_df = season_team_summary_df.copy()
    adj_eff_req_game_cols = ['raw_oe', 'raw_de', 'team_poss', 'opp_poss', 'team_tid', 'opponent_tid', 'location', 'season', 'team_pts']
    for col in adj_eff_req_game_cols:
        if col not in game_team_stats_df.columns:
            print(f"ERROR: Missing column '{col}' in game_team_stats_df for adjusted efficiency calculation.")
            output_df['adj_o'] = np.nan; output_df['adj_d'] = np.nan; output_df['adj_em'] = np.nan
            return output_df
        if col not in ['location', 'season', 'team_tid', 'opponent_tid'] and not pd.api.types.is_numeric_dtype(game_team_stats_df[col]):
             game_team_stats_df.loc[:, col] = pd.to_numeric(game_team_stats_df[col], errors='coerce').fillna(0)
    total_league_pts_for_avg = game_team_stats_df['team_pts'].sum()
    total_league_poss_for_avg = game_team_stats_df['team_poss'].sum()
    if total_league_poss_for_avg == 0:
        print("ERROR: Total league possessions are zero. Cannot calculate league_avg_oe for adjustments.")
        output_df['adj_o'] = np.nan; output_df['adj_d'] = np.nan; output_df['adj_em'] = np.nan
        return output_df
    league_avg_oe = (total_league_pts_for_avg / total_league_poss_for_avg) * 100
    print(f"Iterative Adjustment Baseline League Average Raw OE: {league_avg_oe:.2f}")
    unique_tids = output_df['team_tid'].unique()
    adj_o = {tid: league_avg_oe for tid in unique_tids}; adj_d = {tid: league_avg_oe for tid in unique_tids}
    hca_eff_boost = config_params.HOME_COURT_ADVANTAGE_POINTS
    for iteration in range(config_params.NUM_ADJUSTMENT_ITERATIONS):
        off_residual_points_sum = {tid: 0.0 for tid in unique_tids}; def_residual_points_sum = {tid: 0.0 for tid in unique_tids}
        possessions_sum_for_off_adj = {tid: 0.0 for tid in unique_tids}; possessions_sum_for_def_adj = {tid: 0.0 for tid in unique_tids}
        for _, game_row in game_team_stats_df.iterrows():
            team_tid = game_row['team_tid']; opp_tid = game_row['opponent_tid']; location = game_row['location']
            actual_team_oe_in_game = game_row['raw_oe']; actual_team_de_in_game = game_row['raw_de']
            team_poss_in_game = game_row['team_poss']; opp_poss_in_game = game_row['opp_poss']
            if pd.isna(team_tid) or pd.isna(opp_tid): continue
            if team_poss_in_game <= 0 and opp_poss_in_game <= 0 : continue
            current_adj_o_team = adj_o.get(team_tid, league_avg_oe); current_adj_d_team = adj_d.get(team_tid, league_avg_oe)
            current_adj_o_opp = adj_o.get(opp_tid, league_avg_oe); current_adj_d_opp = adj_d.get(opp_tid, league_avg_oe)
            hca_oe_game_effect = 0.0; hca_de_game_effect = 0.0
            if location == 'Home': hca_oe_game_effect = hca_eff_boost / 2.0; hca_de_game_effect = -hca_eff_boost / 2.0
            elif location == 'Away': hca_oe_game_effect = -hca_eff_boost / 2.0; hca_de_game_effect = hca_eff_boost / 2.0
            predicted_oe_team = current_adj_o_team + current_adj_d_opp - league_avg_oe + hca_oe_game_effect
            predicted_de_team = current_adj_d_team + current_adj_o_opp - league_avg_oe + hca_de_game_effect
            if team_poss_in_game > 0:
                off_residual = actual_team_oe_in_game - predicted_oe_team
                off_residual_points_sum[team_tid] += off_residual * (team_poss_in_game / 100.0)
                possessions_sum_for_off_adj[team_tid] += team_poss_in_game
            if opp_poss_in_game > 0:
                def_residual = actual_team_de_in_game - predicted_de_team
                def_residual_points_sum[team_tid] += def_residual * (opp_poss_in_game / 100.0)
                possessions_sum_for_def_adj[team_tid] += opp_poss_in_game
        new_adj_o = {}; new_adj_d = {}
        for tid in unique_tids:
            if possessions_sum_for_off_adj.get(tid, 0) > 0: new_adj_o[tid] = adj_o.get(tid, league_avg_oe) + (off_residual_points_sum[tid] / possessions_sum_for_off_adj[tid]) * 100
            else: new_adj_o[tid] = adj_o.get(tid, league_avg_oe)
            if possessions_sum_for_def_adj.get(tid, 0) > 0: new_adj_d[tid] = adj_d.get(tid, league_avg_oe) + (def_residual_points_sum[tid] / possessions_sum_for_def_adj[tid]) * 100
            else: new_adj_d[tid] = adj_d.get(tid, league_avg_oe)
        adj_o = new_adj_o; adj_d = new_adj_d
        total_poss_all_teams_off = sum(v for v in possessions_sum_for_off_adj.values() if v > 0)
        total_poss_all_teams_def = sum(v for v in possessions_sum_for_def_adj.values() if v > 0)
        current_w_avg_o = league_avg_oe
        if total_poss_all_teams_off > 0:
            weighted_sum_o = sum(adj_o.get(tid, league_avg_oe) * possessions_sum_for_off_adj.get(tid,0) for tid in unique_tids)
            current_w_avg_o = weighted_sum_o / total_poss_all_teams_off
        current_w_avg_d = league_avg_oe
        if total_poss_all_teams_def > 0:
            weighted_sum_d = sum(adj_d.get(tid, league_avg_oe) * possessions_sum_for_def_adj.get(tid,0) for tid in unique_tids)
            current_w_avg_d = weighted_sum_d / total_poss_all_teams_def
        o_correction = league_avg_oe - current_w_avg_o; d_correction = league_avg_oe - current_w_avg_d
        for tid in unique_tids:
            adj_o[tid] = adj_o.get(tid, league_avg_oe) + o_correction
            adj_d[tid] = adj_d.get(tid, league_avg_oe) + d_correction
        if (iteration + 1) % 25 == 0 or iteration == config_params.NUM_ADJUSTMENT_ITERATIONS -1 :
            print(f"Iteration {iteration + 1}/{config_params.NUM_ADJUSTMENT_ITERATIONS} complete. AvgO: {current_w_avg_o:.2f}, AvgD: {current_w_avg_d:.2f}")
    output_df.loc[:, 'adj_o'] = output_df['team_tid'].map(adj_o).fillna(league_avg_oe)
    output_df.loc[:, 'adj_d'] = output_df['team_tid'].map(adj_d).fillna(league_avg_oe)
    output_df.loc[:, 'adj_em'] = output_df['adj_o'] - output_df['adj_d']
    print("Calculated Adjusted Offensive and Defensive Efficiencies.")
    return output_df

# --- WAB HELPER AND FUNCTION ---
def _calculate_win_probability(team_adjem, opp_adjem, location, hca_points):
    """
    Calculates win probability for team1 against team2 using a logistic function
    based on adjEM difference and home-court advantage.
    """
    hca_effect = 0.0
    if location == 'Home':
        hca_effect = hca_points
    elif location == 'Away':
        hca_effect = -hca_points
    
    margin = team_adjem - opp_adjem + hca_effect
    win_prob = 1 / (1 + np.exp(-margin * config.WIN_PROB_SCALING_FACTOR))
    return win_prob

def calculate_wab(game_team_stats_df, season_team_summary_df, config_params):
    """
    Calculates Wins Above Bubble (WAB) for each team.
    """
    if season_team_summary_df.empty or 'adj_em' not in season_team_summary_df.columns:
        print("WARNING: season_team_summary_df is empty or missing 'adj_em'. Cannot calculate WAB.")
        if not season_team_summary_df.empty: season_team_summary_df['wab'] = np.nan
        return season_team_summary_df
    if game_team_stats_df.empty:
        print("WARNING: game_team_stats_df is empty. Cannot calculate WAB.")
        if not season_team_summary_df.empty: season_team_summary_df['wab'] = np.nan
        return season_team_summary_df

    print("Calculating Wins Above Bubble (WAB)...")
    output_df = season_team_summary_df.copy()
    
    if 'rank_adj_em' not in output_df.columns:
        print("INFO: 'rank_adj_em' not found in season_team_summary_df for WAB. Calculating now.")
        if 'adj_em' in output_df.columns and output_df['adj_em'].notna().any():
             output_df.loc[:, 'rank_adj_em'] = output_df.groupby('season')['adj_em'].rank(method='dense', ascending=False).astype(int)
        else:
            print("ERROR: 'adj_em' is missing or all NaN, cannot rank for WAB.")
            output_df['wab'] = np.nan
            return output_df
            
    bubble_adjem_by_season = {}
    for season, group in output_df.groupby('season'):
        bubble_team_candidate = group[group['rank_adj_em'] == config_params.BUBBLE_TEAM_RANK_THRESHOLD]
        if not bubble_team_candidate.empty:
            bubble_adjem_by_season[season] = bubble_team_candidate['adj_em'].iloc[0]
        else:
            fallback_bubble_teams = group[
                (group['rank_adj_em'] >= config_params.BUBBLE_TEAM_RANK_THRESHOLD - 5) &
                (group['rank_adj_em'] <= config_params.BUBBLE_TEAM_RANK_THRESHOLD + 5)
            ] # Ensure this uses .loc for boolean indexing if group is a slice
            if not fallback_bubble_teams.empty:
                 bubble_adjem_by_season[season] = fallback_bubble_teams['adj_em'].median()
            else:
                 bubble_adjem_by_season[season] = group['adj_em'].mean() - 2.0 # Fallback
            print(f"Season {season}: Bubble rank {config_params.BUBBLE_TEAM_RANK_THRESHOLD} not exact or not found, using adjEM: {bubble_adjem_by_season[season]:.2f}")
    
    team_adjem_lookup = output_df.set_index(['season', 'team_tid'])['adj_em'].to_dict()
    wab_values = []

    required_wab_game_cols = ['season', 'team_tid', 'opponent_tid', 'location']
    if not all(col in game_team_stats_df.columns for col in required_wab_game_cols):
        print(f"ERROR: game_team_stats_df missing one of {required_wab_game_cols} for WAB.")
        output_df['wab'] = np.nan
        return output_df
    if 'wins' not in output_df.columns: # Check in output_df (season_summary)
        print(f"ERROR: 'wins' column missing in season_summary_df for WAB.")
        output_df['wab'] = np.nan
        return output_df


    for _, team_row in output_df.iterrows():
        current_team_tid = team_row['team_tid']; current_season = team_row['season']
        actual_wins = team_row.get('wins', 0)
        current_bubble_adjem = bubble_adjem_by_season.get(current_season)

        if current_bubble_adjem is None or pd.isna(current_bubble_adjem):
            wab_values.append(np.nan); continue
            
        expected_wins_for_bubble = 0.0
        team_schedule = game_team_stats_df[
            (game_team_stats_df['team_tid'] == current_team_tid) &
            (game_team_stats_df['season'] == current_season)
        ]
        if team_schedule.empty:
            wab_values.append(0.0 if actual_wins == 0 else actual_wins - 0.0); continue # If team played 0 games, WAB is actual_wins

        for _, game in team_schedule.iterrows():
            opponent_tid = game['opponent_tid']; game_location = game['location']
            # Fallback for opponent_adjem if opponent_tid is not in our ranked summary (e.g. non-league game)
            opponent_adjem = team_adjem_lookup.get((current_season, opponent_tid))
            if opponent_adjem is None or pd.isna(opponent_adjem): # If opponent is not in our system/summary
                # Use league average adjEM as a proxy for an unranked opponent
                opponent_adjem = output_df[output_df['season']==current_season]['adj_em'].mean()
                if pd.isna(opponent_adjem): opponent_adjem = 0 # Further fallback
            
            win_prob_bubble = _calculate_win_probability(
                current_bubble_adjem, opponent_adjem,
                game_location, config_params.HOME_COURT_ADVANTAGE_POINTS
            )
            expected_wins_for_bubble += win_prob_bubble
            
        wab_values.append(actual_wins - expected_wins_for_bubble)

    output_df['wab'] = wab_values
    print("Calculated Wins Above Bubble (WAB).")
    return output_df

# --- ADJUSTED SOS METRICS FUNCTION ---
def calculate_adjusted_sos_metrics(game_team_stats_df, season_team_summary_df_with_cid):
    # ... (Exact same as the version you confirmed working) ...
    if season_team_summary_df_with_cid.empty: return season_team_summary_df_with_cid
    if game_team_stats_df.empty:
        for col in ['avg_opp_adj_o', 'avg_opp_adj_d', 'avg_opp_adj_em', 'avg_nonconf_opp_adj_em']:
            if col not in season_team_summary_df_with_cid.columns: season_team_summary_df_with_cid[col] = np.nan
        return season_team_summary_df_with_cid
    print("Calculating Adjusted SOS metrics...")
    output_df = season_team_summary_df_with_cid.copy()
    required_summary_cols = ['season', 'team_tid', 'adj_o', 'adj_d', 'adj_em', 'cid']
    if not all(col in output_df.columns for col in required_summary_cols):
        missing_cols = [col for col in required_summary_cols if col not in output_df.columns]
        print(f"ERROR: Missing required columns in season_team_summary_df for Adjusted SOS: {missing_cols}.")
        for col in ['avg_opp_adj_o', 'avg_opp_adj_d', 'avg_opp_adj_em', 'avg_nonconf_opp_adj_em']:
            if col not in output_df.columns: output_df[col] = np.nan
        return output_df
    team_adj_stats_lookup = {}
    for _, row in output_df.iterrows():
        team_adj_stats_lookup[(row['season'], row['team_tid'])] = {'adj_o': row['adj_o'], 'adj_d': row['adj_d'], 'adj_em': row['adj_em'], 'cid': row.get('cid', -999)}
    avg_opp_adj_o_col_data = []; avg_opp_adj_d_col_data = []; avg_opp_adj_em_col_data = []; avg_nonconf_opp_adj_em_col_data = []
    for _, team_row in output_df.iterrows():
        current_team_tid = team_row['team_tid']; current_season = team_row['season']; current_team_cid = team_row.get('cid', -999)
        team_schedule = game_team_stats_df[(game_team_stats_df['team_tid'] == current_team_tid) & (game_team_stats_df['season'] == current_season)]
        if team_schedule.empty:
            avg_opp_adj_o_col_data.append(np.nan); avg_opp_adj_d_col_data.append(np.nan); avg_opp_adj_em_col_data.append(np.nan); avg_nonconf_opp_adj_em_col_data.append(np.nan)
            continue
        opp_adjo_for_team = []; opp_adjd_for_team = []; opp_adjem_for_team = []; nonconf_opp_adjem_for_team = []
        for _, game in team_schedule.iterrows():
            opponent_tid = game['opponent_tid']
            opponent_stats = team_adj_stats_lookup.get((current_season, opponent_tid))
            if opponent_stats:
                opp_adjo_for_team.append(opponent_stats['adj_o']); opp_adjd_for_team.append(opponent_stats['adj_d']); opp_adjem_for_team.append(opponent_stats['adj_em'])
                opponent_cid = opponent_stats.get('cid', -998)
                if current_team_cid != -999 and opponent_cid != -999 and current_team_cid != -998 and opponent_cid != -998 and current_team_cid != opponent_cid:
                    nonconf_opp_adjem_for_team.append(opponent_stats['adj_em'])
        avg_opp_adj_o_col_data.append(np.mean(opp_adjo_for_team) if opp_adjo_for_team else np.nan)
        avg_opp_adj_d_col_data.append(np.mean(opp_adjd_for_team) if opp_adjd_for_team else np.nan)
        avg_opp_adj_em_col_data.append(np.mean(opp_adjem_for_team) if opp_adjem_for_team else np.nan)
        avg_nonconf_opp_adj_em_col_data.append(np.mean(nonconf_opp_adjem_for_team) if nonconf_opp_adjem_for_team else np.nan)
    output_df['avg_opp_adj_o'] = avg_opp_adj_o_col_data; output_df['avg_opp_adj_d'] = avg_opp_adj_d_col_data
    output_df['avg_opp_adj_em'] = avg_opp_adj_em_col_data ; output_df['avg_nonconf_opp_adj_em'] = avg_nonconf_opp_adj_em_col_data
    print("Calculated Adjusted SOS metrics.")
    return output_df

# --- QUADRANT RECORD FUNCTIONS ---
def get_quadrant(location, opponent_rank, quadrant_defs, max_teams_for_ranking_config):
    if pd.isna(opponent_rank): return "Q4"
    for q_name, q_rules in quadrant_defs.items():
        rank_range_key = ""
        if location == 'Home': rank_range_key = "home_rank_range"
        elif location == 'Away': rank_range_key = "away_rank_range"
        elif location == 'Neutral': rank_range_key = "neutral_rank_range"
        else: rank_range_key = "neutral_rank_range"
        if rank_range_key:
            min_r, max_r = q_rules.get(rank_range_key, (9999, 9999))
            if min_r <= opponent_rank <= max_r: return q_name
    # Fallback if Q4 ranges in config are not perfectly catching all higher ranks
    q4_home_min = quadrant_defs.get("Q4", {}).get("home_rank_range",(config.MAX_TEAMS_FOR_RANKING + 1, config.MAX_TEAMS_FOR_RANKING + 1))[0] # Use a large default if Q4 not defined
    q4_neutral_min = quadrant_defs.get("Q4", {}).get("neutral_rank_range",(config.MAX_TEAMS_FOR_RANKING + 1, config.MAX_TEAMS_FOR_RANKING + 1))[0]
    q4_away_min = quadrant_defs.get("Q4", {}).get("away_rank_range",(config.MAX_TEAMS_FOR_RANKING + 1, config.MAX_TEAMS_FOR_RANKING + 1))[0]

    if location == 'Home' and opponent_rank >= q4_home_min: return "Q4"
    if location == 'Neutral' and opponent_rank >= q4_neutral_min: return "Q4"
    if location == 'Away' and opponent_rank >= q4_away_min: return "Q4"
    if opponent_rank >= min(q3_home_min, q3_neutral_min, q3_away_min, default=config.MAX_TEAMS_FOR_RANKING + 1): # If past Q3 lower bounds
        return "Q4"
    return "Q_Error"

def calculate_quadrant_records(game_team_stats_df, season_team_summary_df, config_params):
    """
    Calculates quadrant records for each team AND adds a 'game_quadrant' column
    to the game_team_stats_df.
    """
    if game_team_stats_df.empty or season_team_summary_df.empty:
        print("WARNING: Empty input DataFrame(s) for Quadrant Records calculation.")
        if not season_team_summary_df.empty:
            for i in range(1, 5): season_team_summary_df[f'q{i}_w'] = 0; season_team_summary_df[f'q{i}_l'] = 0
        # Return original game_df if it was passed, or an empty one with expected new col
        if 'game_quadrant' not in game_team_stats_df.columns: game_team_stats_df['game_quadrant'] = "N/A"
        return game_team_stats_df, season_team_summary_df # Return both

    print("Calculating Quadrant Records and adding 'game_quadrant' to game data...")
    output_summary_df = season_team_summary_df.copy()
    games_df_with_quads = game_team_stats_df.copy()

    if 'adj_em' not in output_summary_df.columns:
        print("ERROR: 'adj_em' column missing in season_team_summary_df. Cannot rank for Quadrants.")
        if 'game_quadrant' not in games_df_with_quads.columns: games_df_with_quads['game_quadrant'] = "N/A"
        return games_df_with_quads, output_summary_df

    if 'rank_adj_em' not in output_summary_df.columns: # Ensure rank exists or calculate it
        output_summary_df.loc[:, 'rank_adj_em'] = output_summary_df.groupby('season')['adj_em'].rank(method='dense', ascending=False).astype(int)
    
    team_ranks_lookup = output_summary_df.set_index(['season', 'team_tid'])['rank_adj_em'].to_dict()

    games_df_with_quads.loc[:, 'opponent_rank'] = games_df_with_quads.apply(
        lambda row: team_ranks_lookup.get((row['season'], row['opponent_tid']), config_params.MAX_TEAMS_FOR_RANKING + 1), axis=1
    )
    games_df_with_quads.loc[:, 'game_quadrant'] = games_df_with_quads.apply( # New column
        lambda row: get_quadrant(
            row['location'], row['opponent_rank'],
            config_params.QUADRANT_DEFINITIONS, config_params.MAX_TEAMS_FOR_RANKING
        ), axis=1
    )

    if 'win' not in games_df_with_quads.columns:
        games_df_with_quads.loc[:, 'team_score_official_num'] = pd.to_numeric(games_df_with_quads['team_score_official'], errors='coerce').fillna(0)
        games_df_with_quads.loc[:, 'opponent_score_official_num'] = pd.to_numeric(games_df_with_quads['opponent_score_official'], errors='coerce').fillna(0)
        games_df_with_quads.loc[:, 'win'] = (games_df_with_quads['team_score_official_num'] > games_df_with_quads['opponent_score_official_num']).astype(int)
    
    if 'Q_Error' in games_df_with_quads['game_quadrant'].unique():
        print("WARNING: Some games could not be classified into a quadrant ('Q_Error' found).")
        # games_df_with_quads = games_df_with_quads[games_df_with_quads['game_quadrant'] != 'Q_Error'] # Optional: filter out errors

    if games_df_with_quads.empty or 'game_quadrant' not in games_df_with_quads.columns:
        print("WARNING: No valid games with quadrant info to aggregate for team summary.")
    else:
        quad_records_agg = games_df_with_quads.groupby(['season', 'team_tid', 'game_quadrant', 'win']).size().unstack(fill_value=0)
        if 0 in quad_records_agg.columns: quad_records_agg.rename(columns={0: 'L'}, inplace=True)
        else: quad_records_agg['L'] = 0
        if 1 in quad_records_agg.columns: quad_records_agg.rename(columns={1: 'W'}, inplace=True)
        else: quad_records_agg['W'] = 0
        if 'W' not in quad_records_agg.columns: quad_records_agg['W'] = 0
        if 'L' not in quad_records_agg.columns: quad_records_agg['L'] = 0
        
        quad_records_final = quad_records_agg[['W', 'L']].unstack(level='game_quadrant', fill_value=0)
        if not quad_records_final.empty:
            new_cols = [f"{col_q_name.lower()}_{col_stat_type.lower()}" for col_stat_type, col_q_name in quad_records_final.columns]
            quad_records_final.columns = new_cols
            quad_records_final.reset_index(inplace=True)
            output_summary_df = pd.merge(output_summary_df, quad_records_final, on=['season', 'team_tid'], how='left')
        else:
            print("WARNING: quad_records_final is empty after unstacking. No quadrant records to merge.")

    for i in range(1, 5):
        for wl_char in ['w', 'l']:
            col_name = f'q{i}_{wl_char}'
            if col_name not in output_summary_df.columns: output_summary_df[col_name] = 0
            else: output_summary_df[col_name] = output_summary_df[col_name].fillna(0).astype(int)
    
    # Ensure the game_quadrant column exists in the returned games_df
    if 'game_quadrant' not in games_df_with_quads.columns:
        games_df_with_quads['game_quadrant'] = "N/A"

    print("Calculated Quadrant Records (and added 'game_quadrant' to game data).")
    return games_df_with_quads, output_summary_df
