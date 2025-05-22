# main_processor.py
import config
from data_loader import load_json_file
from data_transformer import create_team_lookup, extract_game_data_and_stats_from_json
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

def main():
    print("--- BrentPom Processor Initializing ---")

    # --- Stage 0: Initialize Database ---
    print("\n--- Stage 0: Initializing Database ---")
    database_ops.create_tables_if_not_exist()

    # --- Stage 1: Loading Data from JSON file ---
    print("\n--- Stage 1: Loading Data from JSON ---")
    zengm_data = load_json_file(config.JSON_FILE_PATH)
    if zengm_data is None:
        print("\nCritical error: Could not load JSON data file. Exiting.")
        return
    print("\nRaw JSON data loaded successfully.")

    # --- Stage 2: Transforming Core Data ---
    print("\n--- Stage 2: Transforming Core Data ---")
    teams_df, tid_to_abbrev, abbrev_to_tid = create_team_lookup(zengm_data)
    if not teams_df.empty:
        database_ops.save_teams_df_to_db(teams_df)
    else:
        print("Critical error: Team lookup creation failed. Cannot proceed without team data.")
        return
    
    raw_game_data_from_json = extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev)
    
    games_to_potentially_save_df = pd.DataFrame()
    if not raw_game_data_from_json.empty:
        print("\nCalculating per-game stats for newly extracted JSON games...")
        games_to_potentially_save_df = calculate_game_possessions_oe_de(raw_game_data_from_json)
        games_to_potentially_save_df = calculate_game_four_factors_and_shooting(games_to_potentially_save_df)
    else:
        print("No game data extracted from JSON to calculate per-game stats on.")

    print("\n--- Stage 2.5: Saving New Game Data to DB ---")
    if not games_to_potentially_save_df.empty:
        # print(f"DEBUG MAIN: Columns in games_to_potentially_save_df BEFORE save attempt: {games_to_potentially_save_df.columns.tolist()}")
        database_ops.save_game_team_stats_to_db(games_to_potentially_save_df)
    else:
        print("No new game data from JSON to save to DB.")

    current_processing_season = None
    if not raw_game_data_from_json.empty and 'season' in raw_game_data_from_json.columns:
        unique_seasons_in_json = raw_game_data_from_json['season'].unique()
        if len(unique_seasons_in_json) > 0:
            current_processing_season = unique_seasons_in_json[0]
            if isinstance(current_processing_season, np.integer):
                current_processing_season = int(current_processing_season)
    
    if current_processing_season is None and zengm_data.get('gameAttributes'):
        ga_list = zengm_data.get('gameAttributes', [])
        ga_source = None
        if isinstance(ga_list, list) and ga_list: ga_source = ga_list[-1]
        elif isinstance(ga_list, dict): ga_source = ga_list
        if isinstance(ga_source, dict):
             current_processing_season_attr = ga_source.get('season')
             if current_processing_season_attr is not None:
                current_processing_season = int(current_processing_season_attr)

    if current_processing_season is None:
        print("ERROR: Could not determine the current processing season. Exiting.")
        return

    print(f"\n--- Stage 3: Loading ALL Games for Season {current_processing_season} from DB & Calculating Summaries ---")
    master_game_team_stats_df = database_ops.load_all_game_stats_for_season(current_processing_season)

    if master_game_team_stats_df.empty:
        print(f"Warning: No game data found in database for season {current_processing_season}. Cannot calculate season summaries.")
        print("\n--- BrentPom Processor Finished (No game data in DB for season) ---")
        return

    print("\nCalculating Season-Level Aggregates (on full season data from DB)...")
    season_team_summary_df = calculate_season_team_aggregates(master_game_team_stats_df)

    if not season_team_summary_df.empty:
        print("\nCalculating RPI and SOS (BCS)...") # RPI/SOS use raw W-L, can be before AdjEff
        season_team_summary_df = calculate_rpi_sos(master_game_team_stats_df, season_team_summary_df)
        
        print("\nCalculating Adjusted Efficiencies...")
        season_team_summary_df = calculate_adjusted_efficiencies(master_game_team_stats_df, season_team_summary_df, config)
        
        # --- CORRECTED ORDER: Calculate Luck AFTER Adjusted Efficiencies ---
        print("\nCalculating Luck (based on Adjusted Efficiencies)...")
        season_team_summary_df = calculate_luck(season_team_summary_df)

        # Merge Conference ID (cid)
        if not teams_df.empty and 'cid' in teams_df.columns and 'team_tid' in season_team_summary_df.columns:
            print("\nMerging Conference IDs into season summary...")
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

        print("\nSeason Team Summary (Top 10 by AdjEM):")
        cols_to_show_final = ['season', 'team_abbrev', 'adj_em', 'wins', 'wab',
                              'Q1', 'Q2', 'Q3', 'Q4',
                              'avg_opp_adj_em',
                              'luck_adj', 'expected_wins_adj',
                              'rpi', 'sos_bcs']
        
        for i in range(1, 5):
            q_w_col = f'q{i}_w'; q_l_col = f'q{i}_l'; q_rec_col = f'Q{i}'
            if q_w_col in season_team_summary_df.columns and q_l_col in season_team_summary_df.columns:
                season_team_summary_df[q_rec_col] = season_team_summary_df[q_w_col].astype(str) + "-" + season_team_summary_df[q_l_col].astype(str)
            elif q_rec_col not in season_team_summary_df.columns:
                 season_team_summary_df[q_rec_col] = "0-0"
        
        existing_cols_final = [col for col in cols_to_show_final if col in season_team_summary_df.columns]
        if existing_cols_final and 'adj_em' in season_team_summary_df.columns:
            print(season_team_summary_df.sort_values(by='adj_em', ascending=False)[existing_cols_final].head(10))
        elif existing_cols_final: print(season_team_summary_df[existing_cols_final].head(10))
        else: print("Season summary DataFrame is missing expected final columns for preview.")
        
        print("\nSaving final season summary to DB...")
        database_ops.save_season_summary_to_db(season_team_summary_df, current_processing_season)
    else:
        print(f"Warning: Season team summary is empty for season {current_processing_season}. Skipping advanced stat calculations for DB save.")
    
    print("\n--- BrentPom Processor: All team stats calculated and saved to DB! ---")
    
    print("\n--- Stage 4: Saving CSV Outputs (Optional) ---")
    output_dir = os.path.join("output", str(current_processing_season))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    base_input_filename = os.path.splitext(config.JSON_FILE_NAME)[0]

    if not master_game_team_stats_df.empty:
        game_output_filename = os.path.join(output_dir, f"brentpom_ALL_game_details_{base_input_filename}_s{current_processing_season}.csv")
        try:
            master_game_team_stats_df.to_csv(game_output_filename, index=False)
            print(f"Full game-by-game team stats for season {current_processing_season} saved to: {game_output_filename}")
        except Exception as e: print(f"Error saving master_game_team_stats_df to CSV: {e}")
    
    if not season_team_summary_df.empty:
        summary_output_filename = os.path.join(output_dir, f"brentpom_season_summary_{base_input_filename}_s{current_processing_season}.csv")
        try:
            if 'adj_em' in season_team_summary_df.columns:
                season_team_summary_df.sort_values(by='adj_em', ascending=False, inplace=True)
            season_team_summary_df.to_csv(summary_output_filename, index=False)
            print(f"Full season summary stats for season {current_processing_season} saved to: {summary_output_filename}")
        except Exception as e: print(f"Error saving season_team_summary_df to CSV: {e}")
    else:
        print("Season team summary is empty, nothing to save to CSV for summary.")

    print("\n--- BrentPom Processor Fully Finished ---")

if __name__ == "__main__":
    main()
