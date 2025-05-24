# main_processor.py
import config
from data_loader import load_json_file, load_recruiting_csv, load_coach_csv
from data_transformer import create_team_lookup, extract_game_data_and_stats_from_json
from recruiting_analyzer import process_individual_recruits, calculate_team_recruiting_summary
from coach_analyzer import process_coach_csv_data, calculate_coach_season_stats, calculate_coach_career_stats, calculate_coach_head_to_head
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
)
import database_ops
import pandas as pd
import numpy as np
import os
import re

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
    if zengm_data is None: return
    print("Raw JSON data loaded successfully.")
    teams_df, tid_to_abbrev, abbrev_to_tid = create_team_lookup(zengm_data)
    if not teams_df.empty: database_ops.save_teams_df_to_db(teams_df)
    else: print("Critical error: Team lookup creation failed."); return

    print("\n--- Coach Assignment Processing ---")
    coach_csv_filename = getattr(config, 'COACH_CSV_FILE_NAME', None)
    if coach_csv_filename:
        coach_csv_path = os.path.join(config.DATA_DIR, coach_csv_filename)
        coach_cols_to_load = ['Year', 'Team', 'Coach Name']
        raw_coach_df = load_coach_csv(coach_csv_path, selected_columns=coach_cols_to_load)
        if raw_coach_df is not None and not raw_coach_df.empty:
            processed_coach_assignments_df = process_coach_csv_data(raw_coach_df, teams_df)
            if not processed_coach_assignments_df.empty:
                database_ops.save_coach_data(processed_coach_assignments_df)
        else: print("No raw coach assignment data loaded.")
    else: print("No COACH_CSV_FILE_NAME in config. Skipping coach assignments.")
    
    print("\n--- Stage 2: Transforming & Saving Game Data ---")
    raw_game_data_from_json = extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev)
    games_to_save_df = pd.DataFrame()
    if not raw_game_data_from_json.empty:
        print("Calculating per-game stats for JSON games...")
        games_to_save_df = calculate_game_possessions_oe_de(raw_game_data_from_json)
        games_to_save_df = calculate_game_four_factors_and_shooting(games_to_save_df)
    
    if not games_to_save_df.empty:
        # This initial save will NOT have game_quadrant yet.
        # game_quadrant is calculated later on master_game_team_stats_df.
        # The re-save of master_game_team_stats_df (after quadrants) will add it.
        database_ops.save_game_team_stats_to_db(games_to_save_df)
    else:
        print("No new game data from JSON to save to DB this run.")

    current_processing_season = None
    if not games_to_save_df.empty and 'season' in games_to_save_df.columns and games_to_save_df['season'].nunique() > 0 :
        current_processing_season = int(games_to_save_df['season'].mode()[0])
    elif zengm_data.get('gameAttributes'):
        ga_list = zengm_data.get('gameAttributes', []); ga_source = ga_list[-1] if isinstance(ga_list, list) and ga_list else (ga_list if isinstance(ga_list, dict) else {})
        if isinstance(ga_source, dict):
             current_season_attr = ga_source.get('season');
             if current_season_attr is not None: current_processing_season = int(current_season_attr)
    if current_processing_season is None: current_processing_season = get_year_from_filename(config.JSON_FILE_NAME, 2024)
    print(f"Determined current processing season for summaries: {current_processing_season}")
    
    print("\n--- Recruiting Data Ingestion (if CSV specified) ---")
    recruiting_csv_filename = getattr(config, 'RECRUITING_CSV_FILE_NAME', None)
    if recruiting_csv_filename:
        recruiting_data_class_year = get_year_from_filename(recruiting_csv_filename, current_processing_season -1)
        print(f"Processing recruiting data for CLASS OF {recruiting_data_class_year} (effective for playing season {recruiting_data_class_year + 1})")
        recruiting_csv_path = os.path.join(config.DATA_DIR, recruiting_csv_filename)
        recruit_cols_to_load = ['#', 'Name', 'COMMITTED TO', 'OVR', 'Position']
        raw_recruits_df = load_recruiting_csv(recruiting_csv_path, selected_columns=recruit_cols_to_load)
        if raw_recruits_df is not None and not raw_recruits_df.empty:
            processed_recruits_df = process_individual_recruits(raw_recruits_df, recruiting_data_class_year, teams_df, abbrev_to_tid)
            if not processed_recruits_df.empty: database_ops.save_processed_recruits(processed_recruits_df, recruiting_data_class_year)
        else: print("No raw recruiting data loaded from CSV.")
    else: print("No RECRUITING_CSV_FILE_NAME in config. Skipping recruiting CSV processing.")
    
    print(f"\n--- Stage 3: Loading ALL Games & Calculating Summaries for Season {current_processing_season} ---")
    master_game_team_stats_df = database_ops.load_all_game_stats_for_season(current_processing_season)

    if master_game_team_stats_df.empty:
        print(f"Warning: No game data in database for season {current_processing_season}. Cannot calculate full summaries.")
        season_team_summary_df = pd.DataFrame()
    else:
        print("Calculating Season-Level Aggregates...")
        season_team_summary_df = calculate_season_team_aggregates(master_game_team_stats_df)

    if not season_team_summary_df.empty:
        season_team_summary_df = calculate_rpi_sos(master_game_team_stats_df, season_team_summary_df)
        season_team_summary_df = calculate_adjusted_efficiencies(master_game_team_stats_df, season_team_summary_df, config)
        season_team_summary_df = calculate_luck(season_team_summary_df)
        
        if not teams_df.empty and 'cid' in teams_df.columns:
            teams_df_for_merge = teams_df[['tid', 'cid']].copy(); teams_df_for_merge.rename(columns={'tid': 'team_tid'}, inplace=True)
            season_team_summary_df = pd.merge(season_team_summary_df, teams_df_for_merge, on='team_tid', how='left')
            season_team_summary_df['cid'] = season_team_summary_df['cid'].fillna(-999).astype(int)
        else:
            if 'cid' not in season_team_summary_df.columns: season_team_summary_df['cid'] = -999
        
        season_team_summary_df = calculate_adjusted_sos_metrics(master_game_team_stats_df, season_team_summary_df)
        
        print("\nCalculating Quadrant Records (and updating game data)...")
        master_game_team_stats_df, season_team_summary_df = calculate_quadrant_records(
            master_game_team_stats_df, season_team_summary_df, config
        )
        
        if 'game_quadrant' in master_game_team_stats_df.columns:
             print("Re-saving game data with quadrant information to DB...")
             database_ops.save_game_team_stats_to_db(master_game_team_stats_df)
        
        season_team_summary_df = calculate_wab(master_game_team_stats_df, season_team_summary_df, config)

        print(f"\nCalculating Team Recruiting Summary for playing season {current_processing_season}...")
        recruits_for_summary_df = database_ops.load_all_recruits_for_effective_season(current_processing_season)
        if not recruits_for_summary_df.empty:
            team_recruiting_summary_df = calculate_team_recruiting_summary(recruits_for_summary_df, current_processing_season)
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
            rec_cols_to_ensure = ['num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked', 'num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv']
            for r_col in rec_cols_to_ensure:
                if r_col not in season_team_summary_df.columns: season_team_summary_df[r_col] = 0 if any(kw in r_col for kw in ['num_', '_star', '_gt', '_juco', '_cpr', 'unranked']) else 0.0
        
        print("\n--- Coach Statistics Calculation ---")
        coach_assignments_df = database_ops.load_coach_assignments_from_db()
        coaches_info_df = database_ops.load_coaches_from_db()

        if not coach_assignments_df.empty and not master_game_team_stats_df.empty and not season_team_summary_df.empty:
            coach_season_stats_df = calculate_coach_season_stats(
                coach_assignments_df, master_game_team_stats_df,
                season_team_summary_df, coaches_info_df )
            if not coach_season_stats_df.empty:
                database_ops.save_coach_season_stats(coach_season_stats_df)
                
                coach_career_stats_df = calculate_coach_career_stats(coach_season_stats_df)
                if not coach_career_stats_df.empty:
                    database_ops.save_coach_career_stats(coach_career_stats_df)
                    
                    # --- Calculate and Save Coach Head-to-Head Stats ---
                    print("\nCalculating Coach Head-to-Head stats...")
                    coach_h2h_df = calculate_coach_head_to_head(master_game_team_stats_df, coach_assignments_df)
                    if not coach_h2h_df.empty:
                        database_ops.save_coach_head_to_head_stats(coach_h2h_df)
                        print("\nSample Coach Head-to-Head Stats (Top 5 by games played):")
                        h2h_cols_to_show = ['season', 'coach1_name', 'coach2_name', 'coach1_wins', 'coach2_wins', 'games_played']
                        existing_h2h_cols = [c for c in h2h_cols_to_show if c in coach_h2h_df.columns]
                        if 'games_played' in coach_h2h_df.columns and existing_h2h_cols:
                            print(coach_h2h_df.sort_values(by='games_played', ascending=False)[existing_h2h_cols].head())
                    else:
                        print("No H2H coach records to display.")


                    print("\nSample Coach Career Stats (Top 5 by Wins if available):")
                    career_cols_to_show = ['coach_name', 'seasons_coached', 'total_wins', 'total_losses', 'career_win_pct', 'career_q1_w', 'career_q1_l', 'career_avg_team_adj_em', 'career_avg_score_onz']
                    existing_career_cols = [col for col in career_cols_to_show if col in coach_career_stats_df.columns]
                    sort_col = 'total_wins' if 'total_wins' in coach_career_stats_df.columns else ('coach_name' if 'coach_name' in coach_career_stats_df.columns else None)
                    if sort_col and existing_career_cols:
                        print(coach_career_stats_df.sort_values(by=sort_col, ascending=False)[existing_career_cols].head())
        else:
            print("Skipping coach statistics calculation due to missing data.")

        print("\nSeason Team Summary (Final - Top 10 by AdjEM):")
        cols_to_show_final = ['season', 'team_abbrev', 'adj_em', 'wins', 'wab', 'Q1', 'Q2', 'Q3', 'Q4', 'avg_opp_adj_em', 'luck_adj', 'score_onz', 'score_nspn', 'score_ktv', 'num_recruits']
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
    
    print("\n--- BrentPom Processor: All stats including H2H calculated and saved! ---")
    
    print("\n--- Stage 4: Saving CSV Outputs (Optional) ---")
    output_dir = os.path.join("output", str(current_processing_season));
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    base_fn = os.path.splitext(config.JSON_FILE_NAME)[0]
    if not master_game_team_stats_df.empty:
        master_game_team_stats_df.to_csv(os.path.join(output_dir, f"brentpom_ALL_games_s{current_processing_season}.csv"), index=False)
    if not season_team_summary_df.empty:
        if 'adj_em' in season_team_summary_df.columns: season_team_summary_df.sort_values(by='adj_em', ascending=False, inplace=True)
        season_team_summary_df.to_csv(os.path.join(output_dir, f"brentpom_SUMMARY_s{current_processing_season}_{base_fn}.csv"), index=False)
        print(f"Full season summary saved to CSV.")

    print("\n--- BrentPom Processor Fully Finished ---")

if __name__ == "__main__":
    main()
