# stats_calculator.py
import pandas as pd
import numpy as np
import config

# --- Helper function for RPI WP calculation ---
def _calculate_team_rpi_wp(team_tid_to_calc, season_of_calc, all_games_df, excluding_opponent_tid=None):
    # ... (Keep this exactly as in your file from response #112) ...
    team_games = all_games_df[ (all_games_df['team_tid'] == team_tid_to_calc) & (all_games_df['season'] == season_of_calc) ].copy()
    if excluding_opponent_tid is not None: team_games = team_games[team_games['opponent_tid'] != excluding_opponent_tid]
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
    # ... (Keep this exactly as in your file from response #112) ...
    if game_team_stats_df.empty: return game_team_stats_df
    df = game_team_stats_df.copy()
    poss_cols_team = ['team_fga', 'team_oreb', 'team_tov', 'team_fta']; oe_de_cols = ['team_pts', 'opponent_pts']
    for col_list in [poss_cols_team, ['opponent_fga', 'opponent_oreb', 'opponent_tov', 'opponent_fta'], oe_de_cols]:
        for col in col_list:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['team_poss'] = (df['team_fga'] - df['team_oreb'] + df['team_tov'] + (config.POSSESSION_FTA_COEFFICIENT * df['team_fta']))
    df['opp_poss'] = (df['opponent_fga'] - df['opponent_oreb'] + df['opponent_tov'] + (config.POSSESSION_FTA_COEFFICIENT * df['opponent_fta']))
    df['raw_oe'] = np.divide(df['team_pts'] * 100, df['team_poss'], out=np.zeros_like(df['team_pts'], dtype=float), where=df['team_poss']!=0)
    df['raw_de'] = np.divide(df['opponent_pts'] * 100, df['opp_poss'], out=np.zeros_like(df['opponent_pts'], dtype=float), where=df['opp_poss']!=0)
    return df

def calculate_game_four_factors_and_shooting(game_team_stats_df):
    # ... (Keep this exactly as in your file from response #112) ...
    if game_team_stats_df.empty: return game_team_stats_df
    if 'team_poss' not in game_team_stats_df.columns or 'opp_poss' not in game_team_stats_df.columns: return game_team_stats_df
    df = game_team_stats_df.copy()
    ff_stat_cols = ['team_fgm', 'team_fgm3', 'team_fga', 'team_tov', 'team_oreb', 'opponent_dreb', 'team_fta', 'opponent_fgm', 'opponent_fgm3', 'opponent_fga', 'opponent_tov', 'opponent_oreb', 'team_dreb', 'opponent_fta', 'team_fga3', 'opponent_fga3']
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
    return df

def calculate_season_team_aggregates(game_team_stats_df):
    if game_team_stats_df.empty:
        print("WARNING: calculate_season_team_aggregates received empty DataFrame.")
        return pd.DataFrame()
        
    print("INFO: Inside calculate_season_team_aggregates...")
    df_for_agg = game_team_stats_df.copy()
    
    # --- Ensure win/loss are present and numeric (0 or 1 integers) ---
    # This block should ideally NOT be hit if main_processor.py prepares them correctly.
    # But as a safeguard or if this function is called from elsewhere:
    if 'win' not in df_for_agg.columns or 'loss' not in df_for_agg.columns or \
       not pd.api.types.is_integer_dtype(df_for_agg.get('win')) or \
       not pd.api.types.is_integer_dtype(df_for_agg.get('loss')) or \
       df_for_agg['win'].isna().any() or df_for_agg['loss'].isna().any(): # Check for NaNs too

        print("INFO (calculate_season_team_aggregates): 'win' or 'loss' columns missing, not integer, or contain NaNs. Recalculating from scores.")
        
        # Ensure scores are numeric for calculation
        df_for_agg.loc[:, 'team_score_official_num'] = pd.to_numeric(df_for_agg.get('team_score_official', 0), errors='coerce').fillna(0)
        df_for_agg.loc[:, 'opponent_score_official_num'] = pd.to_numeric(df_for_agg.get('opponent_score_official', 0), errors='coerce').fillna(0)
        
        df_for_agg.loc[:, 'win'] = (df_for_agg['team_score_official_num'] > df_for_agg['opponent_score_official_num']).astype(int)
        df_for_agg.loc[:, 'loss'] = (df_for_agg['team_score_official_num'] < df_for_agg['opponent_score_official_num']).astype(int)
    
    # --- DEBUG PRINT AFTER ENSURING WIN/LOSS ---
    print("\nDEBUG STATS_CALC: df_for_agg just BEFORE groupby in calculate_season_team_aggregates:")
    cols_to_check_agg_internal = ['gid', 'season', 'team_tid', 'team_abbrev', 'win', 'loss']
    existing_cols_check_agg_internal = [c for c in cols_to_check_agg_internal if c in df_for_agg.columns]
    if existing_cols_check_agg_internal:
        print(df_for_agg[existing_cols_check_agg_internal].head(10))
        if 'win' in df_for_agg.columns:
            print("Value counts for 'win' column (df_for_agg):")
            print(df_for_agg['win'].value_counts(dropna=False))
            print("Dtype for 'win' column (df_for_agg):", df_for_agg['win'].dtype)
    # --- END DEBUG ---

    aggregations = {'gid': 'count', 'win': 'sum', 'loss': 'sum','team_pts': 'sum', 'team_poss': 'sum', 'team_fgm': 'sum', 'team_fga': 'sum', 'team_fgm3': 'sum', 'team_fga3': 'sum', 'team_ftm': 'sum', 'team_fta': 'sum', 'team_oreb': 'sum', 'team_dreb': 'sum', 'team_tov': 'sum', 'team_ast': 'sum', 'team_stl': 'sum', 'team_blk': 'sum', 'team_pf': 'sum','opponent_pts': 'sum', 'opp_poss': 'sum', 'opponent_fgm': 'sum', 'opponent_fga': 'sum', 'opponent_fgm3': 'sum', 'opponent_fga3': 'sum','opponent_ftm': 'sum', 'opponent_fta': 'sum', 'opponent_oreb': 'sum', 'opponent_dreb': 'sum','opponent_tov': 'sum', 'opponent_ast': 'sum', 'opponent_stl': 'sum', 'opponent_blk': 'sum', 'opponent_pf': 'sum'}
    valid_aggregations = {k: v for k, v in aggregations.items() if k in df_for_agg.columns}
    grouping_cols = ['season', 'team_tid', 'team_abbrev']
    for col in grouping_cols:
        if col not in df_for_agg.columns: print(f"ERROR: Grouping column '{col}' missing for season_team_aggregates."); return pd.DataFrame()
    essential_agg_cols = ['gid', 'win', 'loss']
    for col in essential_agg_cols:
        if col not in valid_aggregations: print(f"ERROR: Essential agg column '{col}' missing for season_team_aggregates (not in df_for_agg or not in aggregations dict)."); return pd.DataFrame()

    season_summary_df = df_for_agg.groupby(grouping_cols, as_index=False).agg(valid_aggregations)
    
    if season_summary_df.empty and not df_for_agg.empty:
        print("WARNING: Season summary is empty after groupby in calculate_season_team_aggregates, but df_for_agg was not. Check grouping keys or aggregation logic.")
        return pd.DataFrame()
    elif season_summary_df.empty:
        print("WARNING: Season summary is empty after groupby (likely because df_for_agg was empty).")
        return pd.DataFrame()

    season_summary_df.rename(columns={'gid': 'games_played', 'win': 'wins', 'loss': 'losses'}, inplace=True)
    
    # ... (rest of the function: raw_oe/de, four factors, percentages, win_pct, avg_tempo - keep as is from response #112) ...
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
        season_summary_df['win_pct'] = np.divide(season_summary_df['wins'], season_summary_df['games_played'], out=np.zeros_like(season_summary_df.get('wins',0), dtype=float), where=season_summary_df['games_played']!=0)
    else: season_summary_df['win_pct'] = 0.0
    if 'team_poss' in season_summary_df.columns and 'games_played' in season_summary_df.columns:
        season_summary_df['avg_tempo'] = np.divide(season_summary_df['team_poss'], season_summary_df['games_played'], out=np.zeros_like(season_summary_df.get('team_poss',0), dtype=float), where=season_summary_df['games_played']!=0)
    else: season_summary_df['avg_tempo'] = 0.0

    print("Calculated season-level aggregates and raw metrics for each team.")
    return season_summary_df

def calculate_luck(season_team_summary_df):
    # ... (Keep this exactly as in your file from response #112) ...
    if season_team_summary_df.empty: return season_team_summary_df
    df = season_team_summary_df.copy()
    required_cols = ['wins', 'games_played', 'adj_o', 'adj_d']
    all_cols_present = all(col in df.columns for col in required_cols)
    if not all_cols_present:
        missing = [col for col in required_cols if col not in df.columns]; print(f"ERROR: Missing for Luck: {missing}."); df['expected_wins_adj']=np.nan; df['luck_adj']=np.nan; return df
    adj_o_float = df['adj_o'].astype(float); adj_d_float = df['adj_d'].astype(float)
    k = config.LUCK_PYTHAGOREAN_EXPONENT
    oe_pow_k = np.power(adj_o_float, k); de_pow_k = np.power(adj_d_float, k)
    denominator = oe_pow_k + de_pow_k
    pyth_exp_win_pct_adj = np.divide(oe_pow_k, denominator, out=np.full(df.shape[0], 0.5, dtype=float), where=denominator != 0)
    pyth_exp_win_pct_adj = np.where((adj_o_float > 0) & (adj_d_float == 0), 1.0, pyth_exp_win_pct_adj)
    pyth_exp_win_pct_adj = np.where((adj_o_float == 0) & (adj_d_float > 0), 0.0, pyth_exp_win_pct_adj)
    df['expected_wins_adj'] = pyth_exp_win_pct_adj * df['games_played']
    df['luck_adj'] = df['wins'] - df['expected_wins_adj']
    print("Calculated Luck based on adjusted efficiencies.")
    return df

def calculate_rpi_sos(game_team_stats_df, season_team_summary_df): # game_team_stats_df here should be played games
    # ... (Keep this exactly as in your file from response #112) ...
    if game_team_stats_df.empty or season_team_summary_df.empty:
        if not season_team_summary_df.empty:
            for col_name in ['rpi_wp', 'owp', 'oowp', 'rpi', 'sos_bcs']: season_team_summary_df[col_name] = 0.0
        return season_team_summary_df
    print("Calculating RPI and SOS (BCS)...")
    required_game_cols = ['season', 'team_tid', 'opponent_tid', 'location', 'team_score_official', 'opponent_score_official', 'gid']
    if not all(col in game_team_stats_df.columns for col in required_game_cols):
        missing_req_game_cols = [col for col in required_game_cols if col not in game_team_stats_df.columns]; print(f"ERROR: Missing game_team_stats_df cols for RPI/SOS: {missing_req_game_cols}.")
        if not season_team_summary_df.empty:
            for col_name in ['rpi_wp', 'owp', 'oowp', 'rpi', 'sos_bcs']: season_team_summary_df[col_name] = 0.0
        return season_team_summary_df
    rpi_wp_values = {} ; temp_rpi_wp_df = game_team_stats_df.copy()
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
    owp_values = {}; oowp_values = {}
    unique_teams_played = game_team_stats_df[['season', 'team_tid']].drop_duplicates()
    for _, team_row in unique_teams_played.iterrows():
        current_season = team_row['season']; team_a_tid = team_row['team_tid']
        opponents_of_team_a_tids = game_team_stats_df[(game_team_stats_df['team_tid'] == team_a_tid) & (game_team_stats_df['season'] == current_season)]['opponent_tid'].unique()
        if len(opponents_of_team_a_tids) == 0: owp_values[(current_season, team_a_tid)] = 0.0; continue
        sum_opponent_rpi_wp_excluding_team_a = sum(_calculate_team_rpi_wp(opp_b_tid, current_season, game_team_stats_df, excluding_opponent_tid=team_a_tid) for opp_b_tid in opponents_of_team_a_tids)
        owp_values[(current_season, team_a_tid)] = np.divide(sum_opponent_rpi_wp_excluding_team_a, len(opponents_of_team_a_tids)) if len(opponents_of_team_a_tids) > 0 else 0.0
    for _, team_row in unique_teams_played.iterrows():
        current_season = team_row['season']; team_a_tid = team_row['team_tid']
        opponents_of_team_a_tids = game_team_stats_df[(game_team_stats_df['team_tid'] == team_a_tid) & (game_team_stats_df['season'] == current_season)]['opponent_tid'].unique()
        if len(opponents_of_team_a_tids) == 0: oowp_values[(current_season, team_a_tid)] = 0.0; continue
        sum_oowp_components = sum(owp_values.get((current_season, opp_b_tid), 0.0) for opp_b_tid in opponents_of_team_a_tids)
        oowp_values[(current_season, team_a_tid)] = np.divide(sum_oowp_components, len(opponents_of_team_a_tids)) if len(opponents_of_team_a_tids) > 0 else 0.0
    output_df = season_team_summary_df.copy()
    output_df.loc[:, 'rpi_wp'] = output_df.apply(lambda row: rpi_wp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'owp'] = output_df.apply(lambda row: owp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'oowp'] = output_df.apply(lambda row: oowp_values.get((row['season'], row['team_tid']), 0.0), axis=1)
    output_df.loc[:, 'rpi'] = (output_df['rpi_wp'] * config.RPI_WEIGHT_WIN_PERCENTAGE + output_df['owp'] * config.RPI_WEIGHT_OPPONENT_WIN_PERCENTAGE + output_df['oowp'] * config.RPI_WEIGHT_OPPONENT_OPPONENT_WIN_PERCENTAGE)
    output_df.loc[:, 'sos_bcs'] = (output_df['owp'] * (2/3)) + (output_df['oowp'] * (1/3))
    print("Calculated RPI and SOS (BCS).")
    return output_df

def calculate_adjusted_efficiencies(game_team_stats_df, season_team_summary_df, config_params): # game_team_stats_df here should be played games
    # ... (Keep this exactly as in your file from response #112) ...
    if game_team_stats_df.empty:
        if not season_team_summary_df.empty: season_team_summary_df['adj_o']=np.nan; season_team_summary_df['adj_d']=np.nan; season_team_summary_df['adj_em']=np.nan
        return season_team_summary_df
    print("Calculating Adjusted Offensive and Defensive Efficiencies...")
    output_df = season_team_summary_df.copy()
    adj_eff_req_game_cols = ['raw_oe', 'raw_de', 'team_poss', 'opp_poss', 'team_tid', 'opponent_tid', 'location', 'season', 'team_pts']
    for col in adj_eff_req_game_cols:
        if col not in game_team_stats_df.columns: print(f"ERROR: Missing '{col}' for adj eff calc."); output_df['adj_o']=np.nan; output_df['adj_d']=np.nan; output_df['adj_em']=np.nan; return output_df
        if col not in ['location','season','team_tid','opponent_tid'] and not pd.api.types.is_numeric_dtype(game_team_stats_df[col]): game_team_stats_df.loc[:, col]=pd.to_numeric(game_team_stats_df[col],errors='coerce').fillna(0)
    total_league_pts_for_avg = game_team_stats_df['team_pts'].sum(); total_league_poss_for_avg = game_team_stats_df['team_poss'].sum()
    if total_league_poss_for_avg==0: print("ERROR: Total league poss zero for adj eff."); output_df['adj_o']=np.nan; output_df['adj_d']=np.nan; output_df['adj_em']=np.nan; return output_df
    league_avg_oe = (total_league_pts_for_avg / total_league_poss_for_avg) * 100
    print(f"Iterative Adjustment Baseline League Average Raw OE: {league_avg_oe:.2f}")
    unique_tids = output_df['team_tid'].unique(); adj_o={tid: league_avg_oe for tid in unique_tids}; adj_d={tid: league_avg_oe for tid in unique_tids}
    hca_eff_boost = config_params.HOME_COURT_ADVANTAGE_POINTS
    for iteration in range(config_params.NUM_ADJUSTMENT_ITERATIONS):
        off_res_pts_sum={tid:0.0 for tid in unique_tids}; def_res_pts_sum={tid:0.0 for tid in unique_tids}
        poss_sum_off={tid:0.0 for tid in unique_tids}; poss_sum_def={tid:0.0 for tid in unique_tids}
        for _, game_row in game_team_stats_df.iterrows(): # Iterates over played games
            team_tid=game_row['team_tid']; opp_tid=game_row['opponent_tid']; loc=game_row['location']
            act_oe=game_row['raw_oe']; act_de=game_row['raw_de']; t_poss=game_row['team_poss']; o_poss=game_row['opp_poss']
            if pd.isna(team_tid) or pd.isna(opp_tid) or (t_poss<=0 and o_poss<=0): continue
            cur_adj_o_t=adj_o.get(team_tid,league_avg_oe); cur_adj_d_t=adj_d.get(team_tid,league_avg_oe)
            cur_adj_o_o=adj_o.get(opp_tid,league_avg_oe); cur_adj_d_o=adj_d.get(opp_tid,league_avg_oe)
            hca_off_eff=0.0; hca_def_eff=0.0
            if loc=='Home': hca_off_eff=hca_eff_boost/2.0; hca_def_eff=-hca_eff_boost/2.0
            elif loc=='Away': hca_off_eff=-hca_eff_boost/2.0; hca_def_eff=hca_eff_boost/2.0
            pred_oe_t=cur_adj_o_t+cur_adj_d_o-league_avg_oe+hca_off_eff
            pred_de_t=cur_adj_d_t+cur_adj_o_o-league_avg_oe+hca_def_eff
            if t_poss>0: off_res=act_oe-pred_oe_t; off_res_pts_sum[team_tid]+=off_res*(t_poss/100.0); poss_sum_off[team_tid]+=t_poss
            if o_poss>0: def_res=act_de-pred_de_t; def_res_pts_sum[team_tid]+=def_res*(o_poss/100.0); poss_sum_def[team_tid]+=o_poss
        new_adj_o={}; new_adj_d={}
        for tid in unique_tids:
            new_adj_o[tid] = adj_o.get(tid,league_avg_oe) + (off_res_pts_sum[tid]/poss_sum_off[tid]*100 if poss_sum_off.get(tid,0)>0 else 0)
            new_adj_d[tid] = adj_d.get(tid,league_avg_oe) + (def_res_pts_sum[tid]/poss_sum_def[tid]*100 if poss_sum_def.get(tid,0)>0 else 0)
        adj_o=new_adj_o; adj_d=new_adj_d
        tot_poss_off=sum(v for v in poss_sum_off.values() if v>0); tot_poss_def=sum(v for v in poss_sum_def.values() if v>0)
        cur_w_avg_o=league_avg_oe; cur_w_avg_d=league_avg_oe
        if tot_poss_off>0: cur_w_avg_o=sum(adj_o.get(tid,league_avg_oe)*poss_sum_off.get(tid,0) for tid in unique_tids)/tot_poss_off
        if tot_poss_def>0: cur_w_avg_d=sum(adj_d.get(tid,league_avg_oe)*poss_sum_def.get(tid,0) for tid in unique_tids)/tot_poss_def
        o_corr=league_avg_oe-cur_w_avg_o; d_corr=league_avg_oe-cur_w_avg_d
        for tid in unique_tids: adj_o[tid]=adj_o.get(tid,league_avg_oe)+o_corr; adj_d[tid]=adj_d.get(tid,league_avg_oe)+d_corr
        if (iteration+1)%25==0 or iteration==config_params.NUM_ADJUSTMENT_ITERATIONS-1: print(f"Iter {iteration+1}/{config_params.NUM_ADJUSTMENT_ITERATIONS}. AvgO:{cur_w_avg_o:.2f} AvgD:{cur_w_avg_d:.2f}")
    output_df.loc[:,'adj_o']=output_df['team_tid'].map(adj_o).fillna(league_avg_oe)
    output_df.loc[:,'adj_d']=output_df['team_tid'].map(adj_d).fillna(league_avg_oe)
    output_df.loc[:,'adj_em']=output_df['adj_o']-output_df['adj_d']
    print("Calculated Adjusted Offensive and Defensive Efficiencies.")
    return output_df

def _calculate_single_game_win_prob_wab(team_adjem, opp_adjem, location, hca_points, win_prob_scale_factor):
    hca_effect = 0.0
    if location == 'Home': hca_effect = hca_points
    elif location == 'Away': hca_effect = -hca_points
    margin = team_adjem - opp_adjem + hca_effect
    scale_factor = getattr(config, 'WIN_PROB_SCALING_FACTOR', 0.1)
    win_prob = 1 / (1 + np.exp(-margin * scale_factor))
    return win_prob

def calculate_wab(game_team_stats_df, season_team_summary_df, config_params): # WAB based on played games
    if season_team_summary_df.empty or 'adj_em' not in season_team_summary_df.columns:
        if not season_team_summary_df.empty: season_team_summary_df['wab'] = np.nan
        return season_team_summary_df
    
    played_games_df = game_team_stats_df[game_team_stats_df['is_played'] == True].copy()
    if played_games_df.empty:
        print("WARNING: No played games to calculate WAB.")
        if not season_team_summary_df.empty: season_team_summary_df['wab'] = np.nan
        return season_team_summary_df

    print("Calculating Wins Above Bubble (WAB) based on played games...")
    output_df = season_team_summary_df.copy()
    if 'rank_adj_em' not in output_df.columns:
        if 'adj_em' in output_df.columns and output_df['adj_em'].notna().any():
             output_df.loc[:, 'rank_adj_em'] = output_df.groupby('season')['adj_em'].rank(method='dense', ascending=False).astype(int)
        else: output_df['wab'] = np.nan; return output_df
    bubble_adjem_by_season = {}
    for season, group in output_df.groupby('season'):
        bubble_team_candidate = group[group['rank_adj_em'] == config_params.BUBBLE_TEAM_RANK_THRESHOLD]
        if not bubble_team_candidate.empty: bubble_adjem_by_season[season] = bubble_team_candidate['adj_em'].iloc[0]
        else:
            fallback_bubble_teams = group[(group['rank_adj_em'] >= config_params.BUBBLE_TEAM_RANK_THRESHOLD - 5) & (group['rank_adj_em'] <= config_params.BUBBLE_TEAM_RANK_THRESHOLD + 5)]
            if not fallback_bubble_teams.empty: bubble_adjem_by_season[season] = fallback_bubble_teams['adj_em'].median()
            else: bubble_adjem_by_season[season] = group['adj_em'].mean() - 2.0
    team_adjem_lookup = output_df.set_index(['season', 'team_tid'])['adj_em'].to_dict()
    wab_values = []
    if 'wins' not in output_df.columns: output_df['wab'] = np.nan; return output_df
    for _, team_row in output_df.iterrows():
        current_team_tid = team_row['team_tid']; current_season = team_row['season']
        actual_wins = team_row.get('wins', 0)
        current_bubble_adjem = bubble_adjem_by_season.get(current_season)
        if current_bubble_adjem is None or pd.isna(current_bubble_adjem): wab_values.append(np.nan); continue
        expected_wins_for_bubble = 0.0
        team_schedule = played_games_df[(played_games_df['team_tid'] == current_team_tid) & (played_games_df['season'] == current_season)]
        if team_schedule.empty: wab_values.append(0.0 if actual_wins == 0 else actual_wins); continue
        for _, game in team_schedule.iterrows():
            opponent_tid = game['opponent_tid']; game_location = game['location']
            opponent_adjem = team_adjem_lookup.get((current_season, opponent_tid), output_df[output_df['season']==current_season]['adj_em'].mean())
            if pd.isna(opponent_adjem): opponent_adjem = 0.0
            win_prob_bubble = _calculate_single_game_win_prob_wab(current_bubble_adjem, opponent_adjem, game_location, config_params.HOME_COURT_ADVANTAGE_POINTS, config.WIN_PROB_SCALING_FACTOR)
            expected_wins_for_bubble += win_prob_bubble
        wab_values.append(actual_wins - expected_wins_for_bubble)
    output_df['wab'] = wab_values
    print("Calculated Wins Above Bubble (WAB).")
    return output_df

def calculate_adjusted_sos_metrics(game_team_stats_df, season_team_summary_df_with_cid): # game_team_stats_df should be played games
    if season_team_summary_df_with_cid.empty: return season_team_summary_df_with_cid
    played_games_df = game_team_stats_df[game_team_stats_df['is_played'] == True].copy()
    if played_games_df.empty:
        print("WARNING: No played games to calculate Adj SOS.")
        for col in ['avg_opp_adj_o','avg_opp_adj_d','avg_opp_adj_em','avg_nonconf_opp_adj_em']:
            if col not in season_team_summary_df_with_cid.columns: season_team_summary_df_with_cid[col]=np.nan
        return season_team_summary_df_with_cid
    print("Calculating Adjusted SOS metrics (based on played games)...")
    output_df = season_team_summary_df_with_cid.copy()
    required_summary_cols = ['season','team_tid','adj_o','adj_d','adj_em','cid']
    if not all(col in output_df.columns for col in required_summary_cols):
        missing_cols=[col for col in required_summary_cols if col not in output_df.columns]; print(f"ERROR: Missing for Adj SOS: {missing_cols}.")
        for col in ['avg_opp_adj_o','avg_opp_adj_d','avg_opp_adj_em','avg_nonconf_opp_adj_em']:
            if col not in output_df.columns: output_df[col]=np.nan
        return output_df
    team_adj_stats_lookup={}
    for _, row in output_df.iterrows(): team_adj_stats_lookup[(row['season'],row['team_tid'])]={'adj_o':row['adj_o'],'adj_d':row['adj_d'],'adj_em':row['adj_em'],'cid':row.get('cid',-999)}
    avg_opp_adj_o_l,avg_opp_adj_d_l,avg_opp_adj_em_l,avg_nonconf_opp_adj_em_l = [],[],[],[]
    for _, team_row in output_df.iterrows():
        cur_tid=team_row['team_tid']; cur_s=team_row['season']; cur_cid=team_row.get('cid',-999)
        sched=played_games_df[(played_games_df['team_tid']==cur_tid)&(played_games_df['season']==cur_s)]
        if sched.empty: avg_opp_adj_o_l.append(np.nan); avg_opp_adj_d_l.append(np.nan); avg_opp_adj_em_l.append(np.nan); avg_nonconf_opp_adj_em_l.append(np.nan); continue
        o_o,o_d,o_em,nc_o_em = [],[],[],[]
        for _, game in sched.iterrows():
            opp_tid=game['opponent_tid']; opp_stats=team_adj_stats_lookup.get((cur_s,opp_tid))
            if opp_stats:
                o_o.append(opp_stats['adj_o']); o_d.append(opp_stats['adj_d']); o_em.append(opp_stats['adj_em'])
                opp_cid=opp_stats.get('cid',-998)
                if cur_cid!=-999 and opp_cid!=-999 and cur_cid!=-998 and opp_cid!=-998 and cur_cid!=opp_cid: nc_o_em.append(opp_stats['adj_em'])
        avg_opp_adj_o_l.append(np.mean(o_o) if o_o else np.nan); avg_opp_adj_d_l.append(np.mean(o_d) if o_d else np.nan)
        avg_opp_adj_em_l.append(np.mean(o_em) if o_em else np.nan); avg_nonconf_opp_adj_em_l.append(np.mean(nc_o_em) if nc_o_em else np.nan)
    output_df['avg_opp_adj_o']=avg_opp_adj_o_l; output_df['avg_opp_adj_d']=avg_opp_adj_d_l
    output_df['avg_opp_adj_em']=avg_opp_adj_em_l; output_df['avg_nonconf_opp_adj_em']=avg_nonconf_opp_adj_em_l
    print("Calculated Adjusted SOS metrics.")
    return output_df

# --- QUADRANT RECORD FUNCTIONS ---
def get_quadrant(location, opponent_rank, quadrant_defs, max_teams_for_ranking_config):
    # ... (Keep this exactly as in your file from response #112) ...
    if pd.isna(opponent_rank): return "Q4"
    for q_name, q_rules in quadrant_defs.items():
        key = "neutral_rank_range";
        if location == 'Home': key = "home_rank_range"
        elif location == 'Away': key = "away_rank_range"
        min_r, max_r = q_rules.get(key, (max_teams_for_ranking_config + 1, max_teams_for_ranking_config + 2)) # Default to outside range
        if min_r <= opponent_rank <= max_r: return q_name
    # Fallback if Q4 ranges in config are not perfectly catching all higher ranks
    # This logic ensures that if a rank is valid but not caught by Q1-Q3, it becomes Q4.
    # If QUADRANT_DEFINITIONS properly define Q4 to catch all remaining ranks, this explicit check might be redundant.
    # For example, if Q4 home is 76-MAX, Q4 neutral 101-MAX, Q4 away 136-MAX
    if opponent_rank > 0: return "Q4" # If it's a valid rank and didn't fit Q1-Q3, it's Q4
    return "Q_Error"

def calculate_quadrant_records(game_team_stats_df, season_team_summary_df, config_params):
    if game_team_stats_df.empty:
        print("WARNING: game_team_stats_df empty in calculate_quadrant_records.")
        if season_team_summary_df is not None and not season_team_summary_df.empty:
            for i in range(1,5): season_team_summary_df[f'q{i}_w']=0; season_team_summary_df[f'q{i}_l']=0
        return game_team_stats_df, season_team_summary_df if season_team_summary_df is not None else pd.DataFrame()

    if season_team_summary_df is None or season_team_summary_df.empty:
        print("WARNING: season_team_summary_df is empty or None in calculate_quadrant_records. Initializing.")
        if not game_team_stats_df.empty:
            season_team_summary_df = game_team_stats_df[['season', 'team_tid']].drop_duplicates().copy()
        else:
            return game_team_stats_df, pd.DataFrame()

    print("Calculating Quadrant Records (for team summaries) and adding 'game_quadrant' to all game data...")
    output_summary_df = season_team_summary_df.copy()
    games_df_with_quads = game_team_stats_df.copy()

    if 'adj_em' not in output_summary_df.columns or output_summary_df['adj_em'].isna().all():
        print("ERROR: 'adj_em' missing or all NaN in season_team_summary_df. Cannot rank for Quadrants.")
        if 'game_quadrant' not in games_df_with_quads.columns and not games_df_with_quads.empty: games_df_with_quads['game_quadrant'] = "N/A"
        return games_df_with_quads, output_summary_df

    if 'rank_adj_em' not in output_summary_df.columns or output_summary_df['rank_adj_em'].isna().all():
        output_summary_df.loc[:, 'rank_adj_em'] = output_summary_df.groupby('season')['adj_em'].rank(method='min', ascending=False).fillna(getattr(config_params, 'MAX_TEAMS_FOR_RANKING', 360)+1).astype(int)
    
    team_ranks_lookup = output_summary_df.set_index(['season', 'team_tid'])['rank_adj_em'].to_dict()

    games_df_with_quads.loc[:, 'opponent_rank'] = games_df_with_quads.apply(
        lambda row: team_ranks_lookup.get((row['season'], row['opponent_tid']), getattr(config_params, 'MAX_TEAMS_FOR_RANKING', 360) + 1), axis=1
    )
    games_df_with_quads.loc[:, 'game_quadrant'] = games_df_with_quads.apply(
        lambda row: get_quadrant(
            row['location'], row['opponent_rank'],
            config_params.QUADRANT_DEFINITIONS, getattr(config_params, 'MAX_TEAMS_FOR_RANKING', 360)
        ), axis=1
    )
    
    played_games_for_q_summary = games_df_with_quads[games_df_with_quads['is_played'] == True].copy()
    
    if 'win' not in played_games_for_q_summary.columns and 'team_score_official' in played_games_for_q_summary.columns:
        played_games_for_q_summary.loc[:,'win'] = (pd.to_numeric(played_games_for_q_summary['team_score_official'],errors='coerce').fillna(0) >
                                                   pd.to_numeric(played_games_for_q_summary['opponent_score_official'],errors='coerce').fillna(0)).astype(int)
    elif 'win' in played_games_for_q_summary.columns:
        played_games_for_q_summary.loc[:, 'win'] = pd.to_numeric(played_games_for_q_summary['win'], errors='coerce').fillna(0).astype(int)
    else:
        print("WARNING: 'win' column cannot be determined for played_games_for_q_summary. Quadrant W-L will be 0.")
        for i in range(1, 5):
            output_summary_df[f'q{i}_w'] = 0; output_summary_df[f'q{i}_l'] = 0
        return games_df_with_quads, output_summary_df

    if 'Q_Error' in played_games_for_q_summary['game_quadrant'].unique():
        print(f"INFO: Found games with 'Q_Error' in game_quadrant. Filtering them out for quadrant summary.")
        played_games_for_q_summary = played_games_for_q_summary[played_games_for_q_summary['game_quadrant'] != 'Q_Error']
    
    valid_played_quad_games = played_games_for_q_summary[played_games_for_q_summary['game_quadrant'].isin(['Q1','Q2','Q3','Q4'])]

    if valid_played_quad_games.empty:
        print("WARNING: valid_played_quad_games DataFrame is EMPTY. No Q1-Q4 games found in played games to aggregate for team summary.")
    else:
        # --- ADDED MORE DETAILED DEBUG FOR valid_played_quad_games ---
        print("DEBUG STATS_CALC: valid_played_quad_games (before groupby for quad_summary) head(10):")
        print(valid_played_quad_games[['season', 'team_tid', 'game_quadrant', 'win']].head(10))
        print("DEBUG STATS_CALC: Dtypes of valid_played_quad_games for groupby cols:", valid_played_quad_games[['season', 'team_tid', 'game_quadrant', 'win']].dtypes)
        print("DEBUG STATS_CALC: Value counts for game_quadrant in valid_played_quad_games:")
        print(valid_played_quad_games['game_quadrant'].value_counts())
        # --- END ADDED DEBUG ---

        quad_summary = valid_played_quad_games.groupby(['season','team_tid','game_quadrant'])['win'].agg(
            W='sum',
            G='count'
        ).reset_index()
        quad_summary['L'] = quad_summary['G'] - quad_summary['W']
        
        # This was the line from your debug output:
        print("DEBUG STATS_CALC: quad_summary (after W, G, L calc, before pivot):\n", quad_summary.head(10 if not quad_summary.empty else 0))
        
        if not quad_summary.empty:
            quad_records_final = quad_summary.pivot_table(
                index=['season', 'team_tid'],
                columns='game_quadrant',
                values=['W', 'L'],
                fill_value=0
            )
            # print("DEBUG STATS_CALC: quad_records_final (after pivot table) head:\n", quad_records_final.head())
            
            if not quad_records_final.empty:
                new_cols = [f"{quad_col.lower()}_{val_col.lower()}" for val_col, quad_col in quad_records_final.columns]
                quad_records_final.columns = new_cols
                quad_records_final.reset_index(inplace=True)
                # print("DEBUG STATS_CALC: quad_records_final (after flatten) head:\n", quad_records_final.head())
                
                output_summary_df['season'] = output_summary_df['season'].astype(int)
                output_summary_df['team_tid'] = output_summary_df['team_tid'].astype(int)
                quad_records_final['season'] = quad_records_final['season'].astype(int)
                quad_records_final['team_tid'] = quad_records_final['team_tid'].astype(int)
                
                output_summary_df = pd.merge(output_summary_df, quad_records_final, on=['season','team_tid'], how='left')
            else:
                print("WARNING: quad_records_final is empty after pivot_table. No team summary quadrant records to merge.")
        else:
            print("WARNING: quad_summary (aggregated W,G,L per quadrant) is empty. No data to pivot.")


    for i in range(1,5):
        for wl_char in ['w','l']:
            col_name = f'q{i}_{wl_char}'
            if col_name not in output_summary_df.columns: output_summary_df[col_name]=0
            else: output_summary_df[col_name]=output_summary_df[col_name].fillna(0).astype(int)
            
    if 'game_quadrant' not in games_df_with_quads.columns and not games_df_with_quads.empty:
        games_df_with_quads['game_quadrant']="N/A"
    
    print("Calculated Quadrant Records (team summary based on played games) and added 'game_quadrant' to all game data.")
    return games_df_with_quads, output_summary_df


# --- Prediction Functions from prediction_calculator.py (response #106) ---
def _calculate_single_game_prediction_metrics_from_calculator(team_a_adjem, team_b_adjem,
                                          adj_o_a, adj_d_a, adj_o_b, adj_d_b,
                                          location_for_team_a, league_avg_oe,
                                          hca_points, win_prob_scale_factor):
    hca_effect_margin = 0.0
    hca_effect_score_a = 0.0
    if location_for_team_a == 'Home':
        hca_effect_margin = hca_points
        hca_effect_score_a = hca_points / 2.0
    elif location_for_team_a == 'Away':
        hca_effect_margin = -hca_points
        hca_effect_score_a = -hca_points / 2.0
    team_a_adjem = team_a_adjem if pd.notna(team_a_adjem) else 0.0
    team_b_adjem = team_b_adjem if pd.notna(team_b_adjem) else 0.0
    predicted_margin = team_a_adjem - team_b_adjem + hca_effect_margin
    win_prob = 1 / (1 + np.exp(-predicted_margin * win_prob_scale_factor))
    adj_o_a = adj_o_a if pd.notna(adj_o_a) else league_avg_oe
    adj_d_a = adj_d_a if pd.notna(adj_d_a) else league_avg_oe
    adj_o_b = adj_o_b if pd.notna(adj_o_b) else league_avg_oe
    adj_d_b = adj_d_b if pd.notna(adj_d_b) else league_avg_oe
    pred_score_a = league_avg_oe + (adj_o_a - league_avg_oe) + (adj_d_b - league_avg_oe) + hca_effect_score_a
    pred_score_b = league_avg_oe + (adj_o_b - league_avg_oe) + (adj_d_a - league_avg_oe) - hca_effect_score_a
    return win_prob, predicted_margin, int(round(pred_score_a)), int(round(pred_score_b))

def calculate_game_predictions(games_to_predict_df, team_summary_df): # Removed config_params, use direct import
    if games_to_predict_df.empty:
        print("INFO: No future games provided to calculate_game_predictions.")
        return games_to_predict_df
    required_summary_cols = ['season', 'team_tid', 'adj_em', 'adj_o', 'adj_d', 'avg_tempo']
    if team_summary_df.empty or not all(col in team_summary_df.columns for col in required_summary_cols):
        missing_cols_message = [col for col in required_summary_cols if col not in team_summary_df.columns] if not team_summary_df.empty else "all"
        print(f"ERROR: Team summary data is empty or missing required columns ({missing_cols_message}) for predictions.")
        for col in ['pred_win_prob_team', 'pred_margin_team', 'pred_score_team', 'pred_score_opponent']:
            if col not in games_to_predict_df.columns: games_to_predict_df[col] = np.nan
        return games_to_predict_df
    print(f"Calculating predictions with margin capping for {len(games_to_predict_df)} future game entries...")
    output_df = games_to_predict_df.copy()
    team_stats_for_merge = team_summary_df[required_summary_cols].copy()
    team_stats_for_merge['season'] = team_stats_for_merge['season'].astype(int)
    team_stats_for_merge['team_tid'] = team_stats_for_merge['team_tid'].astype(int)
    output_df['season'] = output_df['season'].astype(int)
    output_df['team_tid'] = output_df['team_tid'].astype(int)
    if 'opponent_tid' in output_df.columns: output_df['opponent_tid'] = output_df['opponent_tid'].astype(int)
    output_df = pd.merge(output_df, team_stats_for_merge.rename(columns={'adj_em': 'adj_em_a', 'adj_o': 'adj_o_a', 'adj_d': 'adj_d_a', 'avg_tempo': 'avg_tempo_a'}), on=['season', 'team_tid'], how='left')
    output_df = pd.merge(output_df, team_stats_for_merge.rename(columns={'team_tid': 'opponent_tid', 'adj_em': 'adj_em_b', 'adj_o': 'adj_o_b', 'adj_d': 'adj_d_b', 'avg_tempo': 'avg_tempo_b'}), on=['season', 'opponent_tid'], how='left')
    league_avg_oe_overall = np.nan
    try:
        if 'adj_o' in team_summary_df.columns and team_summary_df['adj_o'].notna().any():
             league_avg_oe_overall = team_summary_df['adj_o'].mean()
    except Exception as e: print(f"WARNING: Error calculating league_avg_overall_oe: {e}. Will use default.")
    if pd.isna(league_avg_oe_overall):
        print("WARNING: league_avg_oe_overall is NaN after mean. Defaulting to 100.0 for score predictions.")
        league_avg_oe_overall = 100.0
    pred_win_probs, final_pred_margins, final_pred_scores_team, final_pred_scores_opponent = [], [], [], []
    hca_points = getattr(config, 'HOME_COURT_ADVANTAGE_POINTS', 0.0)
    win_prob_scale = getattr(config, 'WIN_PROB_SCALING_FACTOR', 0.1)
    max_score_diff = getattr(config, 'MAX_PREDICTED_SCORE_DIFFERENCE', 50.0)
    for _, row in output_df.iterrows():
        adj_em_a = row.get('adj_em_a'); adj_em_b = row.get('adj_em_b')
        adj_o_a = row.get('adj_o_a'); adj_d_a = row.get('adj_d_a')
        adj_o_b = row.get('adj_o_b'); adj_d_b = row.get('adj_d_b')
        avg_tempo_a = row.get('avg_tempo_a'); avg_tempo_b = row.get('avg_tempo_b')
        location = row['location']
        if pd.isna(adj_em_a) or pd.isna(adj_em_b):
            pred_win_probs.append(0.5);
            if pd.isna(adj_o_a) or pd.isna(adj_d_b) or pd.isna(adj_o_b) or pd.isna(adj_d_a) or pd.isna(avg_tempo_a) or pd.isna(avg_tempo_b):
                final_pred_margins.append(0.0); final_pred_scores_team.append(np.nan); final_pred_scores_opponent.append(np.nan)
                continue
        win_prob, raw_pred_margin_for_win_prob = _calculate_single_game_win_prob(adj_em_a, adj_em_b, location, hca_points, win_prob_scale)
        pred_win_probs.append(win_prob)
        avg_tempo_a = avg_tempo_a if pd.notna(avg_tempo_a) else 70.0
        avg_tempo_b = avg_tempo_b if pd.notna(avg_tempo_b) else 70.0
        predicted_game_poss = (avg_tempo_a + avg_tempo_b) / 2.0
        hca_off_A = 0.0; hca_off_B = 0.0
        if location == 'Home': hca_off_A = hca_points / 2.0; hca_off_B = -hca_points / 2.0
        elif location == 'Away': hca_off_A = -hca_points / 2.0; hca_off_B = hca_points / 2.0
        adj_o_a_calc = adj_o_a if pd.notna(adj_o_a) else league_avg_overall_oe
        adj_d_a_calc = adj_d_a if pd.notna(adj_d_a) else league_avg_overall_oe
        adj_o_b_calc = adj_o_b if pd.notna(adj_o_b) else league_avg_overall_oe
        adj_d_b_calc = adj_d_b if pd.notna(adj_d_b) else league_avg_overall_oe
        expected_oe_a = league_avg_overall_oe + (adj_o_a_calc - league_avg_overall_oe) + (adj_d_b_calc - league_avg_overall_oe) + hca_off_A
        expected_oe_b = league_avg_overall_oe + (adj_o_b_calc - league_avg_overall_oe) + (adj_d_a_calc - league_avg_overall_oe) + hca_off_B
        raw_pred_score_a = expected_oe_a * (predicted_game_poss / 100.0)
        raw_pred_score_b = expected_oe_b * (predicted_game_poss / 100.0)
        raw_score_difference = raw_pred_score_a - raw_pred_score_b
        capped_score_difference = np.clip(raw_score_difference, -max_score_diff, max_score_diff)
        final_pred_score_a = raw_pred_score_a; final_pred_score_b = raw_pred_score_b
        if raw_score_difference != capped_score_difference:
            adjustment_needed = raw_score_difference - capped_score_difference
            final_pred_score_a = raw_pred_score_a - (adjustment_needed / 2.0)
            final_pred_score_b = raw_pred_score_b + (adjustment_needed / 2.0)
        final_pred_margins.append(final_pred_score_a - final_pred_score_b)
        final_pred_scores_team.append(int(round(final_pred_score_a))); final_pred_scores_opponent.append(int(round(final_pred_score_b)))
    output_df['pred_win_prob_team'] = pred_win_probs
    output_df['pred_margin_team'] = final_pred_margins
    output_df['pred_score_team'] = pd.Series(final_pred_scores_team, index=output_df.index).astype('Int64')
    output_df['pred_score_opponent'] = pd.Series(final_pred_scores_opponent, index=output_df.index).astype('Int64')
    cols_to_drop_after_pred = ['adj_em_a', 'adj_o_a', 'adj_d_a', 'avg_tempo_a', 'adj_em_b', 'adj_o_b', 'adj_d_b', 'avg_tempo_b']
    output_df.drop(columns=[col for col in cols_to_drop_after_pred if col in output_df.columns], inplace=True, errors='ignore')
    print(f"Finished calculating predictions with margin capping for {len(output_df)} future game entries.")
    return output_df
