# prediction_calculator.py
import pandas as pd
import numpy as np
import config

def _calculate_single_game_win_prob(team_a_adjem, team_b_adjem, location_for_team_a, hca_points, win_prob_scale_factor):
    hca_effect = 0.0
    if location_for_team_a == 'Home': hca_effect = hca_points
    elif location_for_team_a == 'Away': hca_effect = -hca_points
    
    team_a_adjem = team_a_adjem if pd.notna(team_a_adjem) else 0.0
    team_b_adjem = team_b_adjem if pd.notna(team_b_adjem) else 0.0
    
    margin = team_a_adjem - team_b_adjem + hca_effect
    scale_factor = getattr(config, 'WIN_PROB_SCALING_FACTOR', 0.1)
    if not isinstance(scale_factor, (int, float)): scale_factor = 0.1
    win_prob = 1 / (1 + np.exp(-margin * scale_factor))
    return win_prob, margin

def calculate_game_predictions(games_to_predict_df, team_summary_df):
    if games_to_predict_df.empty:
        print("INFO: No future games provided to calculate_game_predictions.")
        return games_to_predict_df
    
    required_summary_cols = ['season', 'team_tid', 'adj_em', 'adj_o', 'adj_d', 'avg_tempo'] # Added avg_tempo
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
    if 'opponent_tid' in output_df.columns:
        output_df['opponent_tid'] = output_df['opponent_tid'].astype(int)

    output_df = pd.merge(output_df,
                         team_stats_for_merge.rename(columns={'adj_em': 'adj_em_a', 'adj_o': 'adj_o_a', 'adj_d': 'adj_d_a', 'avg_tempo': 'avg_tempo_a'}),
                         on=['season', 'team_tid'], how='left')
    output_df = pd.merge(output_df,
                         team_stats_for_merge.rename(columns={'team_tid': 'opponent_tid', 'adj_em': 'adj_em_b',
                                                              'adj_o': 'adj_o_b', 'adj_d': 'adj_d_b', 'avg_tempo': 'avg_tempo_b'}),
                         on=['season', 'opponent_tid'], how='left')

    league_avg_overall_oe = np.nan
    if 'adj_o' in team_summary_df.columns and team_summary_df['adj_o'].notna().any():
         league_avg_overall_oe = team_summary_df['adj_o'].mean()
    if pd.isna(league_avg_overall_oe):
        league_avg_overall_oe = 100.0
        print(f"WARNING: Using default league_avg_oe_overall: {league_avg_overall_oe} for predictions.")

    pred_win_probs = []
    final_pred_margins = []
    final_pred_scores_team = []
    final_pred_scores_opponent = []

    hca_points = getattr(config, 'HOME_COURT_ADVANTAGE_POINTS', 0.0)
    win_prob_scale = getattr(config, 'WIN_PROB_SCALING_FACTOR', 0.1)
    max_score_diff = getattr(config, 'MAX_PREDICTED_SCORE_DIFFERENCE', 50.0)

    for _, row in output_df.iterrows():
        adj_em_a = row.get('adj_em_a'); adj_em_b = row.get('adj_em_b')
        adj_o_a = row.get('adj_o_a'); adj_d_a = row.get('adj_d_a')
        adj_o_b = row.get('adj_o_b'); adj_d_b = row.get('adj_d_b')
        avg_tempo_a = row.get('avg_tempo_a'); avg_tempo_b = row.get('avg_tempo_b')
        location = row['location']

        if pd.isna(adj_em_a) or pd.isna(adj_em_b): # Cannot calculate win_prob or reliable margin
            pred_win_probs.append(0.5)
            # For scores and margin, if any core stat is missing, append NaN
            if pd.isna(adj_o_a) or pd.isna(adj_d_b) or pd.isna(adj_o_b) or pd.isna(adj_d_a) or pd.isna(avg_tempo_a) or pd.isna(avg_tempo_b):
                final_pred_margins.append(0.0) # Default margin if adjEMs are also NaN
                final_pred_scores_team.append(np.nan)
                final_pred_scores_opponent.append(np.nan)
                continue # Skip to next game
            # If adjEMs were NaN but others not (unlikely), proceed with score calc that might also be NaN
        
        # 1. Calculate Win Probability from raw margin
        # (Use original _calculate_single_game_win_prob as it also returns the raw margin for HCA)
        win_prob, raw_pred_margin_for_win_prob = _calculate_single_game_win_prob(
            adj_em_a, adj_em_b, location, hca_points, win_prob_scale
        )
        pred_win_probs.append(win_prob)

        # 2. Predict Game Possessions
        # Ensure tempos are numeric, default if not
        avg_tempo_a = avg_tempo_a if pd.notna(avg_tempo_a) else 70.0 # Default tempo
        avg_tempo_b = avg_tempo_b if pd.notna(avg_tempo_b) else 70.0 # Default tempo
        predicted_game_poss = (avg_tempo_a + avg_tempo_b) / 2.0

        # 3. Predict Offensive Efficiency for Each Team in Matchup
        hca_off_A = 0.0; hca_off_B = 0.0
        if location == 'Home': hca_off_A = hca_points / 2.0; hca_off_B = -hca_points / 2.0
        elif location == 'Away': hca_off_A = -hca_points / 2.0; hca_off_B = hca_points / 2.0
        
        # Ensure adj_o/d are numeric, default to league_avg_oe if NaN
        adj_o_a_calc = adj_o_a if pd.notna(adj_o_a) else league_avg_overall_oe
        adj_d_a_calc = adj_d_a if pd.notna(adj_d_a) else league_avg_overall_oe
        adj_o_b_calc = adj_o_b if pd.notna(adj_o_b) else league_avg_overall_oe
        adj_d_b_calc = adj_d_b if pd.notna(adj_d_b) else league_avg_overall_oe

        expected_oe_a = league_avg_overall_oe + (adj_o_a_calc - league_avg_overall_oe) + (adj_d_b_calc - league_avg_overall_oe) + hca_off_A
        expected_oe_b = league_avg_overall_oe + (adj_o_b_calc - league_avg_overall_oe) + (adj_d_a_calc - league_avg_overall_oe) + hca_off_B
        
        # 4. Calculate Initial Predicted Scores
        raw_pred_score_a = expected_oe_a * (predicted_game_poss / 100.0)
        raw_pred_score_b = expected_oe_b * (predicted_game_poss / 100.0)

        # 5. Cap the Score Difference
        raw_score_difference = raw_pred_score_a - raw_pred_score_b
        capped_score_difference = np.clip(raw_score_difference, -max_score_diff, max_score_diff)

        # 6. Adjust Scores to Meet Capped Difference
        final_pred_score_a = raw_pred_score_a
        final_pred_score_b = raw_pred_score_b
        if raw_score_difference != capped_score_difference:
            adjustment_needed = raw_score_difference - capped_score_difference # How much total to reduce the gap by
            final_pred_score_a = raw_pred_score_a - (adjustment_needed / 2.0)
            final_pred_score_b = raw_pred_score_b + (adjustment_needed / 2.0)
        
        final_pred_margins.append(final_pred_score_a - final_pred_score_b)
        final_pred_scores_team.append(int(round(final_pred_score_a)))
        final_pred_scores_opponent.append(int(round(final_pred_score_b)))

    output_df['pred_win_prob_team'] = pred_win_probs
    output_df['pred_margin_team'] = final_pred_margins # Store the capped margin
    output_df['pred_score_team'] = pd.Series(final_pred_scores_team, index=output_df.index).astype('Int64')
    output_df['pred_score_opponent'] = pd.Series(final_pred_scores_opponent, index=output_df.index).astype('Int64')
    
    cols_to_drop_after_pred = ['adj_em_a', 'adj_o_a', 'adj_d_a', 'avg_tempo_a', 'adj_em_b', 'adj_o_b', 'adj_d_b', 'avg_tempo_b']
    output_df.drop(columns=[col for col in cols_to_drop_after_pred if col in output_df.columns], inplace=True, errors='ignore')

    print(f"Finished calculating predictions with margin capping for {len(output_df)} future game entries.")
    return output_df
