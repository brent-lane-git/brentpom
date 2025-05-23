# main_processor.py
import config
from data_loader import load_json_file, load_recruiting_csv
from data_transformer import create_team_lookup, extract_game_data_and_stats_from_json
from recruiting_analyzer import process_individual_recruits, calculate_team_recruiting_summary
from stats_calculator import (
    calculate_game_possessions_oe_de,
    calculate_game_four_factors_and_shooting,
    calculate_season_team_aggregates,
    calculate_luck, # Will be called in the correct order now
    calculate_rpi_sos,
    calculate_adjusted_efficiencies,
    calculate_adjusted_sos_metrics,
    calculate_quadrant_records,
    calculate_wab
)
import database_ops
import pandas as pd
import numpy as np
import os
import re

def get_year_from_filename(filename, default_year=None):
    if filename is None: return default_year
    match = re.search(r'(\b\d{4}\b)', filename)
    if match:
        return int(match.group(1))
    print(f"WARNING: Could not extract year from filename '{filename}'. Using default: {default_year}")
    return default_year

def main():
    print("--- BrentPom Processor Initializing ---")

    print("\n--- Stage 0: Initializing Database ---")
    database_ops.create_tables_if_not_exist()

    print("\n--- Stage 1: Loading Game Data from JSON ---")
    zengm_data = load_json_file(config.JSON_FILE_PATH)
    if zengm_data is None:
        print("Halting: ZenGM JSON data failed to load.")
        return
    print("\nRaw JSON data loaded successfully.")

    print("\n--- Stage 2: Transforming Core Game Data ---")
    teams_df, tid_to_abbrev, abbrev_to_tid = create_team_lookup(zengm_data)
    if not teams_df.empty:
        database_ops.save_teams_df_to_db(teams_df)
    else:
        print("Critical error: Team lookup creation failed. Exiting.")
        return
    
    raw_game_data_from_json = extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev)
    
    games_to_save_df = pd.DataFrame()
    if not raw_game_data_from_json.empty:
        print("\nCalculating per-game stats for JSON games...")
        games_to_save_df = calculate_game_possessions_oe_de(raw_game_data_from_json)
        games_to_save_df = calculate_game_four_factors_and_shooting(games_to_save_df)
    
    print("\n--- Saving New Game Data to DB ---")
    if not games_to_save_df.empty:
        database_ops.save_game_team_stats_to_db(games_to_save_df)
    else:
        print("No new game data from JSON to save to DB this run.")

    current_processing_season = None
    if not games_to_save_df.empty and 'season' in games_to_save_df.columns and games_to_save_df['season'].nunique() > 0 :
        current_processing_season = int(games_to_save_df['season'].mode()[0])
        print(f"DEBUG MAIN: Identified current processing season from processed game data: {current_processing_season} (type: {type(current_processing_season)})")
    
    if current_processing_season is None and zengm_data.get('gameAttributes'):
        ga_list = zengm_data.get('gameAttributes', [])
        ga_source = ga_list[-1] if isinstance(ga_list, list) and ga_list else (ga_list if isinstance(ga_list, dict) else {})
        if isinstance(ga_source, dict):
             current_processing_season_attr = ga_source.get('season')
             if current_processing_season_attr is not None:
                current_processing_season = int(current_processing_season_attr)
                print(f"DEBUG MAIN: Identified current processing season from gameAttributes: {current_processing_season} (type: {type(current_processing_season)})")
    
    if current_processing_season is None:
        current_processing_season = get_year_from_filename(config.JSON_FILE_NAME, 2024)
        print(f"WARNING: Using season {current_processing_season} derived from JSON filename or default as last resort.")
    
    if current_processing_season is None:
        print("ERROR: Could not determine the current processing season. Exiting.")
        return
    print(f"Determined current processing season for summaries: {current_processing_season}")
    
    print("\n--- Recruiting Data Processing ---")
    recruiting_csv_filename = getattr(config, 'RECRUITING_CSV_FILE_NAME', None)
    recruiting_data_class_year = None
    if recruiting_csv_filename:
        recruiting_data_class_year = get_year_from_filename(recruiting_csv_filename, current_processing_season -1)
        print(f"Processing recruiting data for CLASS OF {recruiting_data_class_year} (effective for playing season {recruiting_data_class_year + 1})")
        recruiting_csv_path = os.path.join(config.DATA_DIR, recruiting_csv_filename)
        recruit_cols_to_load = ['#', 'Name', 'COMMITTED TO', 'OVR', 'Position']
        raw_recruits_df = load_recruiting_csv(recruiting_csv_path, selected_columns=recruit_cols_to_load)

        if raw_recruits_df is not None and not raw_recruits_df.empty:
            processed_recruits_df = process_individual_recruits(raw_recruits_df, recruiting_data_class_year, teams_df, abbrev_to_tid)
            if not processed_recruits_df.empty:
                database_ops.save_processed_recruits(processed_recruits_df, recruiting_data_class_year)
        else:
            print("No raw recruiting data loaded or processed from CSV.")
    else:
        print("No RECRUITING_CSV_FILE_NAME defined in config. Skipping recruiting processing.")
    
    print(f"\n--- Stage 3: Loading ALL Games for Season {current_processing_season} from DB & Calculating Summaries ---")
    master_game_team_stats_df = database_ops.load_all_game_stats_for_season(current_processing_season)

    if master_game_team_stats_df.empty:
        print(f"Warning: No game data in database for season {current_processing_season}. Cannot calculate full summaries.")
        season_team_summary_df = pd.DataFrame()
    else:
        print("\nCalculating Season-Level Aggregates (on full season data from DB)...")
        season_team_summary_df = calculate_season_team_aggregates(master_game_team_stats_df)

    if not season_team_summary_df.empty:
        print("\nCalculating RPI and SOS (BCS)...")
        season_team_summary_df = calculate_rpi_sos(master_game_team_stats_df, season_team_summary_df)
        
        print("\nCalculating Adjusted Efficiencies...")
        season_team_summary_df = calculate_adjusted_efficiencies(master_game_team_stats_df, season_team_summary_df, config)
        
        # --- CORRECTED ORDER: Calculate Luck AFTER Adjusted Efficiencies ---
        print("\nCalculating Luck (based on Adjusted Efficiencies)...")
        season_team_summary_df = calculate_luck(season_team_summary_df)

        # Merge Conference ID (cid)
        if not teams_df.empty and 'cid' in teams_df.columns:
            teams_df_for_merge = teams_df[['tid', 'cid']].copy(); teams_df_for_merge.rename(columns={'tid': 'team_tid'}, inplace=True)
            season_team_summary_df = pd.merge(season_team_summary_df, teams_df_for_merge, on='team_tid', how='left')
            season_team_summary_df['cid'] = season_team_summary_df['cid'].fillna(-999).astype(int)
        else:
            if 'cid' not in season_team_summary_df.columns: season_team_summary_df['cid'] = -999
        
        print("\nCalculating Adjusted SOS metrics...")
        season_team_summary_df = calculate_adjusted_sos_metrics(master_game_team_stats_df, season_team_summary_df)
        print("\nCalculating Quadrant Records...")
        season_team_summary_df = calculate_quadrant_records(master_game_team_stats_df, season_team_summary_df, config)
        print("\nCalculating Wins Above Bubble (WAB)...")
        season_team_summary_df = calculate_wab(master_game_team_stats_df, season_team_summary_df, config)

        print(f"\nCalculating Team Recruiting Summary for playing season {current_processing_season}...")
        recruits_for_summary_df = database_ops.load_all_recruits_for_effective_season(current_processing_season)

        if not recruits_for_summary_df.empty:
            team_recruiting_summary_df = calculate_team_recruiting_summary(recruits_for_summary_df, current_processing_season)
            if not team_recruiting_summary_df.empty:
                season_team_summary_df = pd.merge(season_team_summary_df, team_recruiting_summary_df,
                                                  on=['season', 'team_tid'], how='left')
                rec_cols_all = [
                    'num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star',
                    'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked',
                    'num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn',
                    'score_storms', 'score_248sports', 'score_ktv']
                for r_col in rec_cols_all:
                    if r_col in season_team_summary_df.columns:
                        is_count_col = any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked'])
                        season_team_summary_df[r_col] = season_team_summary_df[r_col].fillna(0)
                        if is_count_col: season_team_summary_df[r_col] = season_team_summary_df[r_col].astype(int)
                    else: season_team_summary_df[r_col] = 0 if any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked']) else 0.0
        else:
            print(f"No processed recruit data in DB for playing season {current_processing_season} to add to team summaries.")
            rec_cols_to_ensure = ['num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked', 'num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv']
            for r_col in rec_cols_to_ensure:
                if r_col not in season_team_summary_df.columns:
                    season_team_summary_df[r_col] = 0 if any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked']) else 0.0

        print("\nSeason Team Summary (Top 10 by AdjEM, including Recruiting):")
        cols_to_show_final = ['season', 'team_abbrev', 'adj_em', 'wins', 'wab',
                              'Q1', 'Q2', 'Q3', 'Q4', 'avg_opp_adj_em',
                              'luck_adj', 'expected_wins_adj', # Corrected luck columns
                              'score_onz', 'score_nspn', 'score_storms',
                              'score_248sports', 'score_ktv', 'avg_recruit_ovr', 'num_recruits',
                              'num_5_star', 'num_4_star', 'num_gt', 'num_juco', 'num_cpr']
        for i in range(1, 5):
            q_w_col = f'q{i}_w'; q_l_col = f'q{i}_l'; q_rec_col = f'Q{i}'
            if q_w_col in season_team_summary_df.columns and q_l_col in season_team_summary_df.columns:
                season_team_summary_df[q_rec_col] = season_team_summary_df[q_w_col].astype(str) + "-" + season_team_summary_df[q_l_col].astype(str)
            elif q_rec_col not in season_team_summary_df.columns: season_team_summary_df[q_rec_col] = "0-0"
        
        existing_cols_final = [col for col in cols_to_show_final if col in season_team_summary_df.columns]
        if existing_cols_final and 'adj_em' in season_team_summary_df.columns:
            print(season_team_summary_df.sort_values(by='adj_em', ascending=False)[existing_cols_final].head(10))
        
        print("\nSaving final season summary (with recruiting) to DB...")
        database_ops.save_season_summary_to_db(season_team_summary_df, current_processing_season)
    else:
        print(f"Warning: Season team summary is empty for season {current_processing_season}.")
    
    print("\n--- BrentPom Processor: All stats including recruiting calculated and saved! ---")
    
    print("\n--- Stage 4: Saving CSV Outputs (Optional) ---")
    output_dir = os.path.join("output", str(current_processing_season))
    if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created directory: {output_dir}")
    base_input_filename = os.path.splitext(config.JSON_FILE_NAME)[0]
    if not master_game_team_stats_df.empty:
        game_csv_path = os.path.join(output_dir, f"brentpom_ALL_game_details_{base_input_filename}_s{current_processing_season}.csv")
        try: master_game_team_stats_df.to_csv(game_csv_path, index=False)
        except Exception as e: print(f"Error saving master_game_team_stats_df to CSV: {e}")
    if not season_team_summary_df.empty:
        summary_csv_path = os.path.join(output_dir, f"brentpom_season_summary_{base_input_filename}_s{current_processing_season}.csv")
        try:
            if 'adj_em' in season_team_summary_df.columns: season_team_summary_df.sort_values(by='adj_em', ascending=False, inplace=True)
            season_team_summary_df.to_csv(summary_csv_path, index=False)
            print(f"Full season summary saved to: {summary_csv_path}")
        except Exception as e: print(f"Error saving season_team_summary_df to CSV: {e}")

    print("\n--- BrentPom Processor Fully Finished ---")

if __name__ == "__main__":
    main()
