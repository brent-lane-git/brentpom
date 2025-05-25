# main_processor.py
import config # Ensures config is imported to access new variables
from data_loader import load_json_file, load_recruiting_csv, load_coach_csv,load_postseason_csv
from data_transformer import create_team_lookup, extract_game_data_and_stats_from_json
from recruiting_analyzer import process_individual_recruits, calculate_team_recruiting_summary
from coach_analyzer import process_coach_csv_data, calculate_coach_season_stats, calculate_coach_career_stats, calculate_coach_head_to_head
# Assuming calculate_game_predictions is in prediction_calculator.py as per your setup
from prediction_calculator import calculate_game_predictions
from postseason_analyzer import process_postseason_data
from stats_calculator import (
    calculate_game_possessions_oe_de,
    calculate_game_four_factors_and_shooting,
    calculate_season_team_aggregates,
    calculate_luck,
    calculate_rpi_sos,
    calculate_adjusted_efficiencies,
    calculate_adjusted_sos_metrics,
    calculate_quadrant_records,
    calculate_wab
    # calculate_game_predictions was moved to prediction_calculator.py
)
import database_ops
import pandas as pd
import numpy as np
import os
import re

# get_year_from_filename might still be useful for other purposes or if you revert,
# but won't be used for the primary season/class year determination if using config.
def get_year_from_filename(filename, default_year=None):
    if filename is None: return default_year
    match = re.search(r'(\b\d{4}\b)', filename)
    if match: return int(match.group(1))
    return default_year

def main():
    print("--- BrentPom Processor Initializing ---")
    print("\n--- Stage 0: Initializing Database ---")
    database_ops.create_tables_if_not_exist()

    print("\n--- Stage 1: Loading Static & Game Data ---")
    zengm_data = load_json_file(config.JSON_FILE_PATH);
    if zengm_data is None: print("Halting: ZenGM JSON data failed to load."); return
    print("Raw JSON data loaded successfully.")
    teams_df, tid_to_abbrev, abbrev_to_tid = create_team_lookup(zengm_data)
    if not teams_df.empty: database_ops.save_teams_df_to_db(teams_df)
    else: print("Critical error: Team lookup creation failed."); return

    # --- Use Configured Years ---
    try:
        current_processing_season = int(config.TARGET_SEASON_FOR_STATS)
        recruiting_data_class_year_to_process = int(config.TARGET_RECRUITING_CLASS_YEAR)
        print(f"INFO: Using Configured TARGET_SEASON_FOR_STATS: {current_processing_season}")
        print(f"INFO: Using Configured TARGET_RECRUITING_CLASS_YEAR: {recruiting_data_class_year_to_process}")
    except (AttributeError, ValueError) as e:
        print(f"ERROR: Problem with year definitions in config.py. Ensure TARGET_SEASON_FOR_STATS and TARGET_RECRUITING_CLASS_YEAR are defined as integers. {e}")
        # Fallback to dynamic determination or exit if these are critical
        # For now, let's try dynamic if config fails, as per your original file structure.
        print("INFO: Falling back to dynamic season determination for main_processor.")
        if zengm_data.get('gameAttributes'): # ... your original dynamic logic ...
        # ... ensure current_processing_season is set one way or another ...
        # ... and then set recruiting_data_class_year_to_process based on it for the dynamic case ...
            if current_processing_season is not None: # If dynamic logic set it
                recruiting_data_class_year_to_process = get_year_from_filename(config.RECRUITING_CSV_FILE_NAME, current_processing_season -1)


    print("\n--- Recruiting Data Ingestion (if CSV specified) ---")
    recruiting_csv_filename = getattr(config, 'RECRUITING_CSV_FILE_NAME', None)
    if recruiting_csv_filename:
        recruiting_csv_path = os.path.join(config.DATA_DIR, recruiting_csv_filename)
        # Assuming RECRUIT_COLS_TO_LOAD is defined in your config.py
        recruit_cols_to_load = getattr(config, 'RECRUIT_COLS_TO_LOAD', ['#', 'Name', 'COMMITTED TO', 'OVR', 'Position'])
        
        raw_recruits_df = load_recruiting_csv(recruiting_csv_path, selected_columns=recruit_cols_to_load)
        
        if raw_recruits_df is not None and not raw_recruits_df.empty:
            print(f"DEBUG MAIN: Raw recruits loaded. Shape: {raw_recruits_df.shape}")
            processed_recruits_df = process_individual_recruits(
                raw_recruits_df,
                recruiting_data_class_year_to_process,
                teams_df,
                abbrev_to_tid
            )
            
            # --- ADD DEBUG FOR processed_recruits_df ---
            if processed_recruits_df is not None and not processed_recruits_df.empty:
                print(f"DEBUG MAIN: Processed recruits DF shape: {processed_recruits_df.shape}")
                print("DEBUG MAIN: Processed recruits DF columns:", processed_recruits_df.columns.tolist())
                print("DEBUG MAIN: Processed recruits DF head:\n", processed_recruits_df.head())
                print("DEBUG MAIN: Processed recruits DF dtypes:\n", processed_recruits_df.dtypes)
                # Check for NaNs in key NOT NULL columns
                for key_col in ['recruiting_class_year', 'effective_season', 'recruit_name']:
                    if key_col in processed_recruits_df.columns:
                        print(f"DEBUG MAIN: NaNs in '{key_col}': {processed_recruits_df[key_col].isna().sum()}")
                    else:
                        print(f"DEBUG MAIN: Column '{key_col}' MISSING from processed_recruits_df.")
            else:
                print("DEBUG MAIN: processed_recruits_df is empty or None after process_individual_recruits.")
            # --- END DEBUG ---

            if processed_recruits_df is not None and not processed_recruits_df.empty:
                database_ops.save_processed_recruits(
                    processed_recruits_df,
                    recruiting_data_class_year_to_process
                )
            else:
                print("INFO MAIN: No processed recruits to save.")
        else:
            print("No raw recruiting data loaded from CSV or file was empty.")
    else:
        print("No RECRUITING_CSV_FILE_NAME defined in config. Skipping recruiting CSV processing.")
    

    print("\n--- Coach Assignment Processing ---")
    coach_csv_filename = getattr(config, 'COACH_CSV_FILE_NAME', None)
    if coach_csv_filename:
        coach_csv_path = os.path.join(config.DATA_DIR, coach_csv_filename)
        coach_cols_to_load = ['Year', 'Team', 'Coach Name'] # As per your CSV example
        raw_coach_df = load_coach_csv(coach_csv_path, selected_columns=coach_cols_to_load)
        if raw_coach_df is not None and not raw_coach_df.empty:
            # If your coach CSV contains multiple years, you might want to filter it here
            # based on config.TARGET_COACH_ASSIGNMENT_YEAR or current_processing_season
            # For example:
            # if coach_assignment_year_to_process and 'Year' in raw_coach_df.columns:
            #     raw_coach_df = raw_coach_df[pd.to_numeric(raw_coach_df['Year'], errors='coerce') == coach_assignment_year_to_process]
            
            processed_coach_assignments_df = process_coach_csv_data(raw_coach_df, teams_df)
            if not processed_coach_assignments_df.empty:
                database_ops.save_coach_data(processed_coach_assignments_df)
        else: print("No raw coach assignment data loaded from CSV or file was empty.")
    else: print("No COACH_CSV_FILE_NAME defined in config. Skipping coach assignment processing.")
    
    print("\n--- Stage 2: Extracting & Initially Processing ALL Game Data (Played & Future) from JSON ---")
    # extract_game_data_and_stats_from_json now uses the configured current_processing_season
    all_games_from_json_df = extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev, current_processing_season)
    
    master_game_team_stats_df = pd.DataFrame()
    if not all_games_from_json_df.empty:
        played_games_initial_df = all_games_from_json_df[all_games_from_json_df['is_played'] == True].copy()
        future_games_initial_df = all_games_from_json_df[all_games_from_json_df['is_played'] == False].copy()

        if not played_games_initial_df.empty:
            print("Calculating per-game stats for played JSON games...")
            played_games_stats_calculated_df = calculate_game_possessions_oe_de(played_games_initial_df)
            played_games_stats_calculated_df = calculate_game_four_factors_and_shooting(played_games_stats_calculated_df)
            
            if not future_games_initial_df.empty:
                all_cols = list(set(played_games_stats_calculated_df.columns) | set(future_games_initial_df.columns))
                played_games_stats_calculated_df = played_games_stats_calculated_df.reindex(columns=all_cols)
                future_games_initial_df = future_games_initial_df.reindex(columns=all_cols)
                master_game_team_stats_df = pd.concat([played_games_stats_calculated_df, future_games_initial_df], ignore_index=True)
            else:
                master_game_team_stats_df = played_games_stats_calculated_df
        elif not future_games_initial_df.empty:
            master_game_team_stats_df = future_games_initial_df
            print("Only future games found in JSON for this season.")
        # else: # This case was: print("No games (played or future) extracted from JSON for this season.") -> covered by all_games_from_json_df.empty check

        if not master_game_team_stats_df.empty:
            print(f"Saving all {len(master_game_team_stats_df)} game entries for season {current_processing_season} to DB (initial save)...")
            database_ops.save_game_team_stats_to_db(master_game_team_stats_df, current_processing_season)
    else:
        print("No game data extracted from JSON for the target season.")

    print(f"\n--- Stage 3: Loading Games from DB & Calculating Team/Game Summaries for Season {current_processing_season} ---")
    master_game_team_stats_df = database_ops.load_all_game_stats_for_season(current_processing_season)

    season_team_summary_df = pd.DataFrame()
    if not master_game_team_stats_df.empty:
        played_games_for_summary_df = master_game_team_stats_df[master_game_team_stats_df['is_played'] == True].copy()
        
        if not played_games_for_summary_df.empty:
            print("INFO: Ensuring 'win' and 'loss' columns are correctly populated for played_games_for_summary_df.")
            s_team_score = pd.to_numeric(played_games_for_summary_df['team_score_official'], errors='coerce').fillna(0)
            s_opp_score = pd.to_numeric(played_games_for_summary_df['opponent_score_official'], errors='coerce').fillna(0)
            win_condition = s_team_score > s_opp_score
            loss_condition = s_team_score < s_opp_score
            played_games_for_summary_df.loc[:, 'win'] = win_condition.astype(int)
            played_games_for_summary_df.loc[:, 'loss'] = loss_condition.astype(int)
            if played_games_for_summary_df['win'].isna().any():
                played_games_for_summary_df.loc[:, 'win'] = played_games_for_summary_df['win'].fillna(0)
            if played_games_for_summary_df['loss'].isna().any():
                played_games_for_summary_df.loc[:, 'loss'] = played_games_for_summary_df['loss'].fillna(0)
            played_games_for_summary_df['win'] = played_games_for_summary_df['win'].astype(int)
            played_games_for_summary_df['loss'] = played_games_for_summary_df['loss'].astype(int)

            print("Calculating Season-Level Aggregates (on played games)...")
            season_team_summary_df = calculate_season_team_aggregates(played_games_for_summary_df)
        else:
            print(f"No PLAYED game data in database for season {current_processing_season} to calculate summaries.")
    else:
         print(f"No game data at all in database for season {current_processing_season} to calculate summaries.")

    if not season_team_summary_df.empty:
        season_team_summary_df = calculate_rpi_sos(played_games_for_summary_df, season_team_summary_df)
        season_team_summary_df = calculate_adjusted_efficiencies(played_games_for_summary_df, season_team_summary_df, config)
        season_team_summary_df = calculate_luck(season_team_summary_df)
        
        if not teams_df.empty and 'cid' in teams_df.columns:
            teams_df_for_merge = teams_df[['tid', 'cid']].copy(); teams_df_for_merge.rename(columns={'tid': 'team_tid'}, inplace=True)
            season_team_summary_df = pd.merge(season_team_summary_df, teams_df_for_merge, on='team_tid', how='left')
            season_team_summary_df['cid'] = season_team_summary_df['cid'].fillna(-999).astype(int)
        else:
            if 'cid' not in season_team_summary_df.columns: season_team_summary_df['cid'] = -999
        
        season_team_summary_df = calculate_adjusted_sos_metrics(played_games_for_summary_df, season_team_summary_df)
        
        print("\nCalculating Predictions for Future Games...")
        future_games_to_predict_df = master_game_team_stats_df[master_game_team_stats_df['is_played'] == False].copy()
        if not future_games_to_predict_df.empty and not season_team_summary_df.empty:
            predicted_games_df = calculate_game_predictions(future_games_to_predict_df, season_team_summary_df)
            
            if not predicted_games_df.empty:
                prediction_cols_to_update = ['pred_win_prob_team', 'pred_margin_team', 'pred_score_team', 'pred_score_opponent']
                cols_to_keep_from_master = [col for col in master_game_team_stats_df.columns if col not in prediction_cols_to_update or col in ['gid', 'season', 'team_tid']]
                
                master_game_team_stats_df = pd.merge(
                    master_game_team_stats_df[cols_to_keep_from_master],
                    predicted_games_df[['gid', 'season', 'team_tid'] + prediction_cols_to_update],
                    on=['gid', 'season', 'team_tid'],
                    how='left'
                )
                # Fill predictions for played games with NaN if they were not in predicted_games_df
                for pred_col in prediction_cols_to_update:
                    if pred_col in master_game_team_stats_df:
                         master_game_team_stats_df[pred_col] = np.where(master_game_team_stats_df['is_played']==True, np.nan, master_game_team_stats_df[pred_col])

                print(f"Predictions merged/updated for future game entries in master_game_team_stats_df.")
        else:
            print("No future games to predict or team summaries missing for predictions.")

        print("\nCalculating Quadrant Records (for all games, including future)...")
        master_game_team_stats_df, season_team_summary_df = calculate_quadrant_records(
            master_game_team_stats_df, season_team_summary_df, config
        )
        
        if not master_game_team_stats_df.empty:
             print("Re-saving game data with quadrants and predictions to DB...")
             database_ops.save_game_team_stats_to_db(master_game_team_stats_df, current_processing_season)
        
        season_team_summary_df = calculate_wab(played_games_for_summary_df, season_team_summary_df, config)

        print(f"\nCalculating Team Recruiting Summary for playing season {current_processing_season}...")
        # Use the configured recruiting_data_class_year_to_process to ensure alignment
        # The effective_season for recruits is recruiting_class_year + 1.
        # This should match current_processing_season if configured correctly.
        recruits_for_summary_df = database_ops.load_all_recruits_for_effective_season(recruiting_data_class_year_to_process + 1)
        if (recruiting_data_class_year_to_process + 1) != current_processing_season:
            print(f"WARNING: Effective season for recruits ({recruiting_data_class_year_to_process + 1}) does not match current processing season ({current_processing_season}). Recruiting summary might be misaligned.")

        if not recruits_for_summary_df.empty:
            team_recruiting_summary_df = calculate_team_recruiting_summary(recruits_for_summary_df, current_processing_season) # Summarize for current playing season
            if not team_recruiting_summary_df.empty:
                season_team_summary_df = pd.merge(season_team_summary_df, team_recruiting_summary_df, on=['season', 'team_tid'], how='left')
                rec_cols_all = ['num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked','num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv']
                for r_col in rec_cols_all:
                    if r_col in season_team_summary_df.columns:
                        is_count_col = any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked'])
                        season_team_summary_df[r_col] = season_team_summary_df[r_col].fillna(0)
                        if is_count_col: season_team_summary_df[r_col] = season_team_summary_df[r_col].astype(int)
                    else: season_team_summary_df[r_col] = 0 if any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked']) else 0.0
        else:
            print(f"INFO: No recruits found for effective season {recruiting_data_class_year_to_process + 1} to include in team summaries for season {current_processing_season}.")
            rec_cols_to_ensure = ['num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked', 'num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv']
            for r_col in rec_cols_to_ensure:
                if r_col not in season_team_summary_df.columns: season_team_summary_df[r_col] = 0 if any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked']) else 0.0
        print("\n--- Postseason Results Processing ---")
    postseason_csv_filename = getattr(config, 'POSTSEASON_RESULTS_CSV_FILE_NAME', None)
    postseason_data_for_merge = pd.DataFrame() # Initialize empty

    if postseason_csv_filename:
        postseason_csv_path = os.path.join(config.DATA_DIR, postseason_csv_filename)
        raw_postseason_df = load_postseason_csv(postseason_csv_path, selected_columns=config.POSTSEASON_CSV_COLS)
        
        if raw_postseason_df is not None and not raw_postseason_df.empty:
            # Process using current_processing_season to tag the data
            processed_postseason_df = process_postseason_data(raw_postseason_df, teams_df, current_processing_season)
            if not processed_postseason_df.empty:
                database_ops.save_postseason_results(processed_postseason_df, current_processing_season)
                # Load it back for merging into season_team_summary_df
                postseason_data_for_merge = database_ops.load_postseason_results_for_season(current_processing_season)
        else:
            print("No raw postseason data loaded from CSV or file was empty.")
    else:
        print("No POSTSEASON_RESULTS_CSV_FILE_NAME defined in config. Skipping postseason results processing.")
    # --- END Postseason Results Processing ---


    # Continue with finalizing season_team_summary_df
    if not season_team_summary_df.empty:
        # ... (Recruiting summary merge happened just before the new postseason section based on #113 flow) ...

        # --- MERGE POSTSEASON DATA into season_team_summary_df ---
        if not postseason_data_for_merge.empty:
            print("Merging postseason results into team summaries...")
            # Pivot postseason_data_for_merge to get nt_seed, nt_result, nit_seed, nit_result per team
            # Input: season, team_tid, event_type, seed, result
            # Output needed: season, team_tid, nt_seed, nt_result, nit_seed, nit_result
            
            postseason_pivoted_list = []
            for (season, team_tid), group in postseason_data_for_merge.groupby(['season', 'team_tid']):
                team_entry = {'season': season, 'team_tid': team_tid}
                nt_info = group[group['event_type'] == 'NT']
                if not nt_info.empty:
                    team_entry['nt_seed'] = nt_info['seed'].iloc[0] if pd.notna(nt_info['seed'].iloc[0]) else None
                    team_entry['nt_result'] = nt_info['result'].iloc[0]
                
                nit_info = group[group['event_type'] == 'NIT']
                if not nit_info.empty:
                    team_entry['nit_seed'] = nit_info['seed'].iloc[0] if pd.notna(nit_info['seed'].iloc[0]) else None
                    team_entry['nit_result'] = nit_info['result'].iloc[0]
                postseason_pivoted_list.append(team_entry)
            
            if postseason_pivoted_list:
                postseason_pivoted_df = pd.DataFrame(postseason_pivoted_list)
                season_team_summary_df = pd.merge(season_team_summary_df, postseason_pivoted_df,
                                                  on=['season', 'team_tid'], how='left')
                print("Postseason results merged.")
            else:
                print("No pivoted postseason data to merge.")
        else:
            print("No postseason data loaded to merge into team summaries.")
        
        # Ensure new postseason columns exist in season_team_summary_df and fill NaNs
        # (This is also handled by save_season_summary_to_db's logic now)
        postseason_cols_in_summary = ['nt_seed', 'nt_result', 'nit_seed', 'nit_result']
        for ps_col in postseason_cols_in_summary:
            if ps_col not in season_team_summary_df.columns:
                if 'seed' in ps_col:
                    season_team_summary_df[ps_col] = 0 # Default seed to 0 or np.nan if preferred
                else: # result
                    season_team_summary_df[ps_col] = "N/A" # Default result
            else: # Fill NaNs if column came from merge
                if 'seed' in ps_col:
                    season_team_summary_df[ps_col] = season_team_summary_df[ps_col].fillna(0) # Or np.nan
                else:
                     season_team_summary_df[ps_col] = season_team_summary_df[ps_col].fillna("N/A")

        print("\n--- Coach Statistics Calculation ---")
        coach_assignments_df = database_ops.load_coach_assignments_from_db()
        coaches_info_df = database_ops.load_coaches_from_db()
        if not coach_assignments_df.empty and not master_game_team_stats_df.empty and not season_team_summary_df.empty:
            coach_season_stats_df = calculate_coach_season_stats(coach_assignments_df, master_game_team_stats_df, season_team_summary_df, coaches_info_df)
            if not coach_season_stats_df.empty:
                database_ops.save_coach_season_stats(coach_season_stats_df)
                coach_career_stats_df = calculate_coach_career_stats(coach_season_stats_df)
                if not coach_career_stats_df.empty:
                    database_ops.save_coach_career_stats(coach_career_stats_df)
                    coach_h2h_df = calculate_coach_head_to_head(master_game_team_stats_df, coach_assignments_df)
                    if not coach_h2h_df.empty: database_ops.save_coach_head_to_head_stats(coach_h2h_df)
        else: print("Skipping coach statistics calculation due to missing data.")

        print("\nSeason Team Summary (Final - Top 10 by AdjEM):")
        # ... (Final print logic - same as before) ...
        cols_to_show_final = ['season', 'team_abbrev', 'adj_em', 'wins', 'wab', 'Q1', 'Q2', 'Q3', 'Q4', 'avg_opp_adj_em', 'luck_adj', 'expected_wins_adj', 'score_onz', 'score_nspn', 'score_ktv', 'num_recruits']
        for i in range(1, 5):
            q_w_col=f'q{i}_w'; q_l_col=f'q{i}_l'; q_rec_col=f'Q{i}'
            if q_w_col in season_team_summary_df.columns and q_l_col in season_team_summary_df.columns:
                season_team_summary_df[q_rec_col] = season_team_summary_df[q_w_col].astype(str) + "-" + season_team_summary_df[q_l_col].astype(str)
            elif q_rec_col not in season_team_summary_df.columns: season_team_summary_df[q_rec_col] = "0-0"
        existing_cols_final = [col for col in cols_to_show_final if col in season_team_summary_df.columns]
        if existing_cols_final and 'adj_em' in season_team_summary_df.columns:
            print(season_team_summary_df.sort_values(by='adj_em', ascending=False)[existing_cols_final].head(10))
        
        print("\nSaving final season summary (with recruiting) to DB...")
        database_ops.save_season_summary_to_db(season_team_summary_df, current_processing_season)
    else:
        print(f"Warning: Season team summary is empty for season {current_processing_season} after all calculations.")
    
    print("\n--- BrentPom Processor: All stats including predictions for future games, calculated and saved! ---")
    
    
    print("\n--- Stage 4: Saving CSV Outputs (Optional) ---")
    output_dir = os.path.join("output", str(current_processing_season));
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    base_fn = os.path.splitext(os.path.basename(config.JSON_FILE_PATH))[0] # Use basename for JSON_FILE_PATH
    if not master_game_team_stats_df.empty and 'gid' in master_game_team_stats_df.columns:
        game_csv_path = os.path.join(output_dir, f"brentpom_ALL_games_s{current_processing_season}_{base_fn}.csv")
        try: master_game_team_stats_df.to_csv(game_csv_path, index=False)
        except Exception as e: print(f"Error saving master_game_team_stats_df to CSV: {e}")
    if not season_team_summary_df.empty:
        if 'adj_em' in season_team_summary_df.columns: season_team_summary_df.sort_values(by='adj_em', ascending=False, inplace=True)
        summary_csv_path = os.path.join(output_dir, f"brentpom_SUMMARY_s{current_processing_season}_{base_fn}.csv")
        season_team_summary_df.to_csv(summary_csv_path, index=False)
        print(f"Full season summary saved to CSV.")

    print("\n--- BrentPom Processor Fully Finished ---")

if __name__ == "__main__":
    main()
