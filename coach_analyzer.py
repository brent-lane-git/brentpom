# coach_analyzer.py
import pandas as pd
import numpy as np
import config # Assuming config might be used for some constants if not passed

def process_coach_csv_data(raw_coach_df, teams_df):
    if raw_coach_df.empty:
        print("WARNING: Raw coach DataFrame is empty in process_coach_csv_data.")
        return pd.DataFrame()

    df = raw_coach_df.copy()
    
    df.rename(columns={
        'Year': 'season',
        'Team': 'team_identifier_str',
        'Coach Name': 'coach_name'
    }, inplace=True)

    required_cols = ['season', 'team_identifier_str', 'coach_name']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"ERROR: Coach CSV data missing essential columns after rename: {missing}")
        return pd.DataFrame()

    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df.dropna(subset=['season'], inplace=True)
    df['season'] = df['season'].astype(int)

    df['coach_name'] = df['coach_name'].astype(str).str.strip()
    df['team_identifier_str'] = df['team_identifier_str'].astype(str).str.strip()

    original_len = len(df)
    df = df[df['coach_name'] != '']
    if len(df) < original_len:
        print(f"INFO: Filtered out {original_len - len(df)} rows due to missing coach names.")
    
    if df.empty:
        print("WARNING: No valid coach assignments after filtering for missing coach names.")
        return pd.DataFrame()

    team_name_to_tid_map = {}
    if not teams_df.empty:
        for _, row in teams_df.iterrows():
            tid = row['tid']
            if pd.notna(row['region']):
                team_name_to_tid_map[str(row['region']).strip().lower()] = tid
            if pd.notna(row['abbrev']) and str(row['abbrev']).strip().lower() not in team_name_to_tid_map:
                team_name_to_tid_map[str(row['abbrev']).strip().lower()] = tid
            if pd.notna(row['full_name']) and str(row['full_name']).strip().lower() not in team_name_to_tid_map:
                team_name_to_tid_map[str(row['full_name']).strip().lower()] = tid
            if pd.notna(row['name']) and str(row['name']).strip().lower() not in team_name_to_tid_map:
                 team_name_to_tid_map[str(row['name']).strip().lower()] = tid
    
    def map_team_id(team_id_str):
        if pd.isna(team_id_str) or str(team_id_str).strip() == "": return np.nan
        normalized_name = str(team_id_str).strip().lower()
        return team_name_to_tid_map.get(normalized_name, np.nan)

    df['team_tid'] = df['team_identifier_str'].apply(map_team_id)
    
    unmapped_teams = df[df['team_tid'].isna()]['team_identifier_str'].dropna().unique()
    if len(unmapped_teams) > 0:
        print(f"WARNING: Could not map the following team identifiers from Coach CSV to tids: {list(unmapped_teams)}")

    original_len_before_team_filter = len(df)
    df.dropna(subset=['team_tid'], inplace=True)
    if not df.empty:
        df['team_tid'] = df['team_tid'].astype(int)
    if len(df) < original_len_before_team_filter:
        print(f"INFO: Dropped {original_len_before_team_filter - len(df)} coach assignments due to unmatchable team identifiers.")

    if df.empty:
        print("WARNING: No valid coach assignments left after team mapping.")
        return pd.DataFrame()

    final_cols = ['season', 'team_tid', 'coach_name']
    df = df[final_cols]
    
    print(f"Processed {len(df)} valid coach assignments from CSV.")
    return df


def calculate_coach_season_stats(coach_assignments_df, all_games_with_quads_df, full_season_team_summaries_df, coaches_df):
    if coach_assignments_df.empty:
        print("WARNING: Coach assignments data is empty. Cannot calculate coach season stats.")
        return pd.DataFrame()
    
    print("Calculating coach season-by-season stats...")
    
    if 'coach_name' not in coach_assignments_df.columns:
        if 'coach_id' in coach_assignments_df.columns and not coaches_df.empty and \
           'coach_id' in coaches_df.columns and 'coach_name' in coaches_df.columns:
            coach_assignments_df = pd.merge(coach_assignments_df, coaches_df[['coach_id', 'coach_name']], on='coach_id', how='left')
        elif 'coach_id' in coach_assignments_df.columns:
             coach_assignments_df['coach_name'] = "CoachID_" + coach_assignments_df['coach_id'].astype(str)
        else:
            print("ERROR: coach_name and coach_id missing from coach_assignments_df")
            return pd.DataFrame()

    coach_assignments_df['season'] = coach_assignments_df['season'].astype(int)
    coach_assignments_df['team_tid'] = coach_assignments_df['team_tid'].astype(int)
    if not full_season_team_summaries_df.empty:
        full_season_team_summaries_df['season'] = full_season_team_summaries_df['season'].astype(int)
        full_season_team_summaries_df['team_tid'] = full_season_team_summaries_df['team_tid'].astype(int)
    else:
        expected_summary_cols_for_merge = ['season', 'team_tid', 'team_abbrev', 'games_played', 'wins', 'losses', 'win_pct', 'adj_em', 'rank_adj_em', 'raw_oe', 'raw_de', 'avg_tempo', 'num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked', 'num_gt', 'num_juco', 'num_cpr', 'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv']
        full_season_team_summaries_df = pd.DataFrame(columns=expected_summary_cols_for_merge)

    coach_season_stats = pd.merge(
        coach_assignments_df,
        full_season_team_summaries_df,
        on=['season', 'team_tid'],
        how='left'
    )

    if coach_season_stats.empty and not coach_assignments_df.empty:
        print("WARNING: coach_season_stats is empty after merging assignments with team summaries, but assignments existed.")
        return pd.DataFrame()
    elif coach_season_stats.empty:
        return pd.DataFrame()

    team_season_q_records = pd.DataFrame()
    if not all_games_with_quads_df.empty and \
       'game_quadrant' in all_games_with_quads_df.columns and \
       all_games_with_quads_df['game_quadrant'].notna().any() and \
       'win' in all_games_with_quads_df.columns:
        
        valid_quad_games = all_games_with_quads_df[
            all_games_with_quads_df['game_quadrant'].isin(['Q1', 'Q2', 'Q3', 'Q4'])
        ].copy()
        if not valid_quad_games.empty:
            valid_quad_games.loc[:, 'season'] = valid_quad_games['season'].astype(int)
            valid_quad_games.loc[:, 'team_tid'] = valid_quad_games['team_tid'].astype(int)
            game_summary_for_coaches = valid_quad_games.groupby(
                ['season', 'team_tid', 'game_quadrant']
            )['win'].agg(W='sum', G='count').reset_index()
            game_summary_for_coaches['L'] = game_summary_for_coaches['G'] - game_summary_for_coaches['W']
            
            if not game_summary_for_coaches.empty:
                team_season_q_records = game_summary_for_coaches.pivot_table(
                    index=['season', 'team_tid'],
                    columns='game_quadrant',
                    values=['W', 'L'],
                    fill_value=0
                )
                if not team_season_q_records.empty:
                    team_season_q_records.columns = [f"{q_name.lower()}_{wl_char.lower()}" for wl_char, q_name in team_season_q_records.columns]
                    team_season_q_records.reset_index(inplace=True)
                    team_season_q_records.loc[:, 'season'] = team_season_q_records['season'].astype(int)
                    team_season_q_records.loc[:, 'team_tid'] = team_season_q_records['team_tid'].astype(int)
    
    # --- USER'S SPECIFIED MERGE LOGIC FOR QUADRANT RECORDS ---
    if not team_season_q_records.empty:
        coach_season_stats['season'] = coach_season_stats['season'].astype(int)
        coach_season_stats['team_tid'] = coach_season_stats['team_tid'].astype(int)
        team_season_q_records['season'] = team_season_q_records['season'].astype(int)
        team_season_q_records['team_tid'] = team_season_q_records['team_tid'].astype(int)

        # Identify the actual quadrant columns to merge
        quad_cols_in_source = [col for col in team_season_q_records.columns if col.startswith('q') and ('_w' in col or '_l' in col)]
        cols_to_merge_from_q_records = ['season', 'team_tid'] + quad_cols_in_source
        
        # Ensure we only try to select existing columns from team_season_q_records
        # This also handles the case where team_season_q_records might be missing some qX_w/l columns
        # if no games fell into a specific quadrant for any team.
        actual_cols_to_merge = [col for col in cols_to_merge_from_q_records if col in team_season_q_records.columns]
        
        if not all(key_col in team_season_q_records.columns for key_col in ['season', 'team_tid']):
            print("ERROR: team_season_q_records is missing season or team_tid for merge with coach_season_stats.")
        elif not all(key_col in coach_season_stats.columns for key_col in ['season', 'team_tid']):
            print("ERROR: coach_season_stats is missing season or team_tid for merge with quadrant data.")
        elif not actual_cols_to_merge or len(actual_cols_to_merge) <=2 : # only season, team_tid left
            print("WARNING: No actual quadrant columns (qX_w, qX_l) found in team_season_q_records to merge.")
        else:
            coach_season_stats = pd.merge(
                coach_season_stats,
                team_season_q_records[actual_cols_to_merge],
                on=['season', 'team_tid'],
                how='left',
                suffixes=('', '_q_dup')
            )
            # Check for _q_dup columns to see if there was a clash (should not happen if actual_cols_to_merge is built correctly)
            q_dup_cols = [col for col in coach_season_stats.columns if '_q_dup' in col]
            if q_dup_cols:
                print(f"DEBUG: Quadrant merge created duplicate-suffixed columns: {q_dup_cols}. This indicates an unexpected column name overlap.")
    # --- END USER'S SPECIFIED MERGE LOGIC ---

    rename_for_db_schema = {
        'adj_em': 'team_adj_em', 'rank_adj_em': 'team_rank_adj_em',
        'raw_oe': 'team_raw_oe', 'raw_de': 'team_raw_de',
        'avg_tempo': 'team_avg_tempo', 'games_played': 'games_coached'
    }
    existing_cols_to_rename = {k: v for k,v in rename_for_db_schema.items() if k in coach_season_stats.columns}
    coach_season_stats.rename(columns=existing_cols_to_rename, inplace=True)

    db_schema_coach_season_cols = [
        'coach_id', 'coach_name', 'season', 'team_tid', 'team_abbrev',
        'games_coached', 'wins', 'losses', 'win_pct',
        'q1_w', 'q1_l', 'q2_w', 'q2_l', 'q3_w', 'q3_l', 'q4_w', 'q4_l',
        'team_adj_em', 'team_rank_adj_em', 'team_raw_oe', 'team_raw_de', 'team_avg_tempo',
        'num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank',
        'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked',
        'num_gt', 'num_juco', 'num_cpr',
        'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv'
    ]
    
    if 'team_abbrev' not in coach_season_stats.columns and 'team_tid' in coach_season_stats.columns:
        # This relies on teams_df being available if full_season_team_summaries_df didn't provide it
        # For now, let's assume full_season_team_summaries_df contains team_abbrev
        if 'teams_df' in locals() and not teams_df.empty : # Check if teams_df was passed or is global (not ideal for global)
             abbrev_map = teams_df.set_index('tid')['abbrev'].to_dict()
             coach_season_stats['team_abbrev'] = coach_season_stats['team_tid'].map(abbrev_map)
        else: # It should be there from the initial merge with full_season_team_summaries_df
            if 'team_abbrev' not in coach_season_stats.columns: # Final fallback
                 coach_season_stats['team_abbrev'] = "N/A"


    for col in db_schema_coach_season_cols:
        if col not in coach_season_stats.columns:
            is_int_or_count = any(kw in col for kw in ['_w', '_l', 'games_coached', 'wins', 'losses', 'num_']) or \
                              col in ['coach_id', 'season', 'team_tid', 'team_rank_adj_em', 'cid']
            coach_season_stats[col] = 0 if is_int_or_count else np.nan
        else:
            is_int_or_count_fill = any(kw in col for kw in ['_w', '_l', 'games_coached', 'wins', 'losses', 'num_']) or \
                               col in ['coach_id', 'season', 'team_tid', 'team_rank_adj_em', 'cid']
            if is_int_or_count_fill:
                coach_season_stats[col] = coach_season_stats[col].fillna(0)
                if coach_season_stats[col].notna().all() and pd.api.types.is_numeric_dtype(coach_season_stats[col]):
                     try: coach_season_stats[col] = coach_season_stats[col].astype(int)
                     except ValueError: coach_season_stats[col] = pd.to_numeric(coach_season_stats[col], errors='coerce').fillna(0).astype(int)
            else:
                if coach_season_stats[col].dtype == 'object' and col in ['coach_name', 'team_abbrev']:
                    coach_season_stats[col] = coach_season_stats[col].fillna("N/A")
                else:
                    default_fill = 0.0 if 'score' in col or col in ['team_adj_em', 'team_raw_oe', 'team_raw_de', 'team_avg_tempo', 'avg_recruit_ovr', 'win_pct'] else np.nan
                    coach_season_stats[col] = coach_season_stats[col].fillna(default_fill)
    
    print(f"Processed coach season-by-season stats for {len(coach_season_stats)} entries.")
    
    final_cols_ordered = [col for col in db_schema_coach_season_cols if col in coach_season_stats.columns]
    return coach_season_stats[final_cols_ordered]


def calculate_coach_career_stats(coach_season_stats_df):
    # ... (Keep this function exactly as in response #75) ...
    if coach_season_stats_df.empty or 'coach_id' not in coach_season_stats_df.columns: return pd.DataFrame()
    print("Calculating coach career stats...")
    if 'coach_name' not in coach_season_stats_df.columns: coach_season_stats_df['coach_name'] = "CoachID_" + coach_season_stats_df['coach_id'].astype(str)
    career_aggs = {'season':'nunique','team_tid':'nunique','games_coached':'sum','wins':'sum','losses':'sum','q1_w':'sum','q1_l':'sum','q2_w':'sum','q2_l':'sum','q3_w':'sum','q3_l':'sum','q4_w':'sum','q4_l':'sum','team_adj_em':'mean','team_rank_adj_em':'mean','num_recruits':'sum','avg_recruit_ovr':'mean','num_5_star':'sum','num_4_star':'sum','num_3_star':'sum','num_2_star':'sum','num_1_star':'sum','num_hs_unranked':'sum','num_gt':'sum','num_juco':'sum','num_cpr':'sum','score_onz':'mean','score_nspn':'mean','score_storms':'mean','score_248sports':'mean','score_ktv':'mean'}
    valid_career_aggs = {k:v for k,v in career_aggs.items() if k in coach_season_stats_df.columns}
    if not valid_career_aggs or not all(key in valid_career_aggs for key in ['season','wins','losses','games_coached']):
        print(f"ERROR: Not enough valid columns for career aggregation. Valid: {list(valid_career_aggs.keys())}")
        if 'coach_id' in coach_season_stats_df.columns and 'coach_name' in coach_season_stats_df.columns: return coach_season_stats_df[['coach_id','coach_name']].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame()
    coach_career_stats = coach_season_stats_df.groupby(['coach_id','coach_name'], as_index=False).agg(valid_career_aggs)
    rename_map_career = {'season':'seasons_coached','team_tid':'teams_coached_count','games_coached':'total_games_coached','wins':'total_wins','losses':'total_losses','team_adj_em':'career_avg_team_adj_em','team_rank_adj_em':'career_avg_team_rank','num_recruits':'career_total_recruits','avg_recruit_ovr':'career_avg_recruit_ovr_of_classes','num_5_star':'career_total_5_stars','num_4_star':'career_total_4_stars','num_3_star':'career_total_3_stars','num_2_star':'career_total_2_stars','num_1_star':'career_total_1_stars','num_hs_unranked':'career_total_hs_unranked','num_gt':'career_total_gt','num_juco':'career_total_juco','num_cpr':'career_total_cpr','score_onz':'career_avg_score_onz','score_nspn':'career_avg_score_nspn','score_storms':'career_avg_score_storms','score_248sports':'career_avg_score_248sports','score_ktv':'career_avg_score_ktv'}
    for i in range(1,5):
        for wl in ['w','l']: rename_map_career[f'q{i}_{wl}'] = f'career_q{i}_{wl}'
    actual_rename_map = {k:v for k,v in rename_map_career.items() if k in coach_career_stats.columns}
    coach_career_stats.rename(columns=actual_rename_map, inplace=True)
    db_schema_career_cols = ['coach_id','coach_name','seasons_coached','teams_coached_count','total_games_coached','total_wins','total_losses','career_win_pct','career_q1_w','career_q1_l','career_q2_w','career_q2_l','career_q3_w','career_q3_l','career_q4_w','career_q4_l','career_avg_team_adj_em','career_avg_team_rank','career_total_recruits','career_avg_recruit_ovr_of_classes','career_total_5_stars','career_total_4_stars','career_total_3_stars','career_total_2_stars','career_total_1_stars','career_total_hs_unranked','career_total_gt','career_total_juco','career_total_cpr','career_avg_score_onz','career_avg_score_nspn','career_avg_score_storms','career_avg_score_248sports','career_avg_score_ktv']
    for col in db_schema_career_cols:
        if col not in coach_career_stats.columns:
             is_count_col = any(kw in col for kw in ['total_','career_q','seasons_coached','teams_coached_count']) or col == 'coach_id'
             coach_career_stats[col] = 0 if is_count_col else np.nan
    if 'total_wins' in coach_career_stats.columns and 'total_games_coached' in coach_career_stats.columns:
        coach_career_stats['career_win_pct'] = np.divide(coach_career_stats.get('total_wins',0), coach_career_stats.get('total_games_coached',0), out=np.zeros_like(coach_career_stats.get('total_wins', pd.Series(0, index=coach_career_stats.index) if not coach_career_stats.empty else 0), dtype=float), where=coach_career_stats.get('total_games_coached',0) != 0 )
    elif 'career_win_pct' not in coach_career_stats.columns: coach_career_stats['career_win_pct'] = 0.0
    print(f"Calculated career stats for {len(coach_career_stats)} coaches.")
    return coach_career_stats


def calculate_coach_head_to_head(all_games_df, coach_assignments_df):
    # ... (Keep this function exactly as in response #77 - confirmed working) ...
    if all_games_df.empty or coach_assignments_df.empty: return pd.DataFrame()
    print("Calculating coach head-to-head records...")
    coach_lookup = {}
    if 'coach_id' in coach_assignments_df.columns and 'coach_name' in coach_assignments_df.columns and 'season' in coach_assignments_df.columns and 'team_tid' in coach_assignments_df.columns:
        for _, row in coach_assignments_df.iterrows(): coach_lookup[(int(row['season']), int(row['team_tid']))] = (int(row['coach_id']), row['coach_name']) # Ensure keys are int
    else: print("ERROR: coach_assignments_df missing required columns for H2H."); return pd.DataFrame()
    
    # Ensure 'win' column and key columns are correct type in all_games_df
    if 'win' not in all_games_df.columns:
        all_games_df.loc[:,'win'] = (pd.to_numeric(all_games_df['team_score_official'],errors='coerce').fillna(0) >
                                     pd.to_numeric(all_games_df['opponent_score_official'],errors='coerce').fillna(0)).astype(int)
    all_games_df['season'] = all_games_df['season'].astype(int)
    all_games_df['team_tid'] = all_games_df['team_tid'].astype(int)
    all_games_df['opponent_tid'] = all_games_df['opponent_tid'].astype(int)


    h2h_records = {}; processed_gids_season = set()
    for _, game_row in all_games_df.iterrows():
        gid = game_row['gid']; season = game_row['season']
        if (gid, season) in processed_gids_season: continue
        processed_gids_season.add((gid, season))
        team1_tid = game_row['team_tid']; team2_tid = game_row['opponent_tid']
        team1_coach_info = coach_lookup.get((season, team1_tid)); team2_coach_info = coach_lookup.get((season, team2_tid))
        if team1_coach_info and team2_coach_info:
            coach1_id, coach1_name = team1_coach_info; coach2_id, coach2_name = team2_coach_info
            if coach1_id == coach2_id: continue
            c1_id, c1_n, c2_id, c2_n = (coach1_id, coach1_name, coach2_id, coach2_name) if coach1_id < coach2_id else (coach2_id, coach2_name, coach1_id, coach1_name)
            matchup_key = (season, c1_id, c2_id)
            if matchup_key not in h2h_records: h2h_records[matchup_key] = {'c1_name': c1_n, 'c2_name': c2_n, 'c1_wins': 0, 'c2_wins': 0, 'games': 0}
            h2h_records[matchup_key]['games'] += 1
            if game_row['win'] == 1:
                if coach1_id == c1_id: h2h_records[matchup_key]['c1_wins'] += 1
                else: h2h_records[matchup_key]['c2_wins'] += 1
            else:
                if coach2_id == c2_id: h2h_records[matchup_key]['c2_wins'] += 1
                else: h2h_records[matchup_key]['c1_wins'] += 1
    h2h_list = []
    for (season, c1_id, c2_id), data in h2h_records.items():
        h2h_list.append({'season': season, 'coach1_id': c1_id, 'coach1_name': data['c1_name'], 'coach2_id': c2_id, 'coach2_name': data['c2_name'], 'coach1_wins': data['c1_wins'], 'coach2_wins': data['c2_wins'], 'games_played': data['games']})
    coach_h2h_df = pd.DataFrame(h2h_list)
    if not coach_h2h_df.empty: print(f"Calculated {len(coach_h2h_df)} coach head-to-head season records.")
    else: print("No coach head-to-head records generated.")
    return coach_h2h_df
