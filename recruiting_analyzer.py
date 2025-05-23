# recruiting_analyzer.py
import pandas as pd
import numpy as np
import config

def get_recruit_type_and_numeric_rank(rank_str):
    """
    Parses the rank string to determine recruit type and numeric rank.
    Types: HS, GT, JUCO, CPR, Unknown, HS_Unranked
    """
    rank_str = str(rank_str).strip().upper()
    if not rank_str: # Handle empty string case
        return "Unknown", np.nan
    if rank_str == "GT": return "GT", np.nan
    if rank_str == "JUCO": return "JUCO", np.nan
    if rank_str == "CPR": return "CPR", np.nan
    try:
        numeric_val = float(rank_str) # Handle if rank is like "25.0"
        if pd.notna(numeric_val): # Check if it's a valid number after conversion
            return "HS", int(numeric_val)
        # If float conversion results in NaN (e.g., from empty string that wasn't caught, though unlikely now)
        return "Unknown", np.nan
    except ValueError:
        # If it's not GT/JUCO/CPR and not a number, it might be unranked HS or other
        return "HS_Unranked", np.nan

def assign_star_rating(numeric_rank, recruit_type, star_definitions, max_hs_rank_for_stars):
    """Assigns star rating based on numeric_rank for HS recruits."""
    if recruit_type != "HS" or pd.isna(numeric_rank): # Only HS recruits with numeric_rank get stars
        return 0
    
    # Check if rank is beyond the scope of star ratings (e.g. > 500 for 1-star)
    # This ensures that ranks like 600 (if MAX_HS_RANK_FOR_STARS is 500) get 0 stars.
    if numeric_rank > max_hs_rank_for_stars:
        return 0

    for stars, (min_r, max_r) in star_definitions.items():
        if min_r <= numeric_rank <= max_r:
            return stars
            
    # If an HS recruit has a numeric_rank within max_hs_rank_for_stars but doesn't fall into 5-2 star bins,
    # they might be a 1-star by default if not explicitly covered, or 0 if ranges are tight.
    # The current STAR_DEFINITIONS has 1-star as 401-500.
    # If a rank is, say, 350 and no 2-star definition covers it, it would fall through.
    # The loop above should handle it if definitions are comprehensive.
    # If it falls through all defined star ranges but is <= max_hs_rank_for_stars, it's likely unrated within those tiers.
    return 0 # Default to 0 stars if not fitting any defined tier but is HS and ranked within limit

def process_individual_recruits(raw_recruits_df, recruiting_class_year, teams_df, abbrev_to_tid):
    """
    Processes raw recruit data:
    - Cleans data, assigns types, ranks, stars.
    - Calculates individual scores for different "recruiting services".
    - Maps committed school to team_tid using teams_df (region, name, full_name, abbrev).
    """
    if raw_recruits_df.empty:
        print("WARNING: Raw recruits DataFrame is empty in process_individual_recruits.")
        return pd.DataFrame()

    df = raw_recruits_df.copy()
    
    # Standardize column names
    df.rename(columns={
        '#': 'original_rank_str',
        'Name': 'recruit_name',
        'COMMITTED TO': 'committed_to_name_str',
        'OVR': 'recruit_ovr',
        'Position': 'position'
    }, inplace=True)

    required_cols = ['original_rank_str', 'recruit_name', 'committed_to_name_str', 'recruit_ovr']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"ERROR: Recruiting data missing essential columns: {missing}")
        return pd.DataFrame()

    # Filter out rows with missing or empty recruit names FIRST
    original_len_total = len(df)
    df.dropna(subset=['recruit_name'], inplace=True) # Drop rows where recruit_name is NaN
    # Ensure recruit_name is string before stripping, then check for empty
    df = df[df['recruit_name'].astype(str).str.strip() != '']
    if len(df) < original_len_total:
        print(f"WARNING: Dropped {original_len_total - len(df)} recruits due to missing or empty names.")
    if df.empty:
        print("WARNING: No recruits left after filtering for missing names.")
        return pd.DataFrame()

    df['recruit_ovr'] = pd.to_numeric(df['recruit_ovr'], errors='coerce').fillna(50) # Default OVR if missing
    df['recruiting_class_year'] = int(recruiting_class_year)
    df['effective_season'] = int(recruiting_class_year) + 1

    type_rank = df['original_rank_str'].apply(get_recruit_type_and_numeric_rank)
    df['recruit_type'] = type_rank.apply(lambda x: x[0])
    df['numeric_rank'] = type_rank.apply(lambda x: x[1])

    df['star_rating'] = df.apply(
        lambda row: assign_star_rating(
            row['numeric_rank'], row['recruit_type'],
            config.STAR_DEFINITIONS, config.MAX_HS_RANK_FOR_STARS
        ), axis=1
    )

    # Map 'committed_to_name_str' to 'team_tid'
    name_to_tid_map = {}
    if not teams_df.empty:
        for _, row in teams_df.iterrows():
            tid = row['tid']
            # Add multiple potential keys, all pointing to the same tid
            if pd.notna(row['abbrev']): name_to_tid_map[str(row['abbrev']).strip().lower()] = tid
            if pd.notna(row['name']): name_to_tid_map[str(row['name']).strip().lower()] = tid # Mascot e.g. "Dukes"
            if pd.notna(row['full_name']): name_to_tid_map[str(row['full_name']).strip().lower()] = tid # Region + Name e.g. "Duquesne Dukes"
            if pd.notna(row['region']): name_to_tid_map[str(row['region']).strip().lower()] = tid   # University Name/City e.g. "Duquesne"
    
    def map_school_to_tid(school_name_str):
        if pd.isna(school_name_str) or str(school_name_str).strip() == "": return np.nan
        normalized_school_name = str(school_name_str).strip().lower()
        tid = name_to_tid_map.get(normalized_school_name)
        if pd.notna(tid): return tid
        if '(' in normalized_school_name:
            base_name = normalized_school_name.split('(')[0].strip()
            tid = name_to_tid_map.get(base_name)
            return tid if pd.notna(tid) else np.nan
        return np.nan

    df['team_tid'] = df['committed_to_name_str'].apply(map_school_to_tid)
    
    # Filter out uncommitted/unmatchable recruits
    original_len_before_commit_filter = len(df)
    df.dropna(subset=['team_tid'], inplace=True)
    if not df.empty:
        df['team_tid'] = df['team_tid'].astype(int)

    if len(df) < original_len_before_commit_filter:
        num_dropped_uncommitted = original_len_before_commit_filter - len(df)
        unmapped_names_sample = raw_recruits_df[~raw_recruits_df.index.isin(df.index)]['COMMITTED TO'].dropna().unique()
        print(f"WARNING: Dropped {num_dropped_uncommitted} recruits (uncommitted or unmatchable schools like: {list(unmapped_names_sample[:5])}).") # Show a few examples
    
    if df.empty:
        print("WARNING: No recruits left after filtering for uncommitted/unmatchable schools.")
        return pd.DataFrame()

    # Calculate OnZ "True Overall"
    def get_onz_true_ovr(row):
        ovr=row['recruit_ovr']; rtype=row['recruit_type']
        if rtype == "HS": return ovr
        if rtype == "JUCO": return ovr - 10
        if rtype == "CPR": return ovr - 12
        if rtype == "GT": return ovr - 14
        return ovr - 15 # Default for "Unknown" or "HS_Unranked" if not handled as HS
    df['true_overall_onz'] = df.apply(get_onz_true_ovr, axis=1)

    # Calculate NSPN Points
    def get_nspn_pts(row):
        rtype=row['recruit_type']; stars=row['star_rating']
        if rtype == "GT": return config.NSPN_POINTS.get("GT", 0)
        if rtype == "JUCO": return config.NSPN_POINTS.get("JUCO", 0)
        if rtype == "CPR": return config.NSPN_POINTS.get("CPR", 0)
        if rtype == "HS" or rtype == "HS_Unranked":
            key = f"{stars}_star" if stars > 0 else "HS_unranked" # Use "HS_unranked" for 0-star HS
            return config.NSPN_POINTS.get(key, 0)
        return config.NSPN_POINTS.get("Unknown", 0) # Default for other unknown types
    df['points_nspn'] = df.apply(get_nspn_pts, axis=1)

    # Calculate Storms.com Points
    def get_storms_pts(row):
        score = max(0, row['recruit_ovr'] - config.STORMS_OVR_BASELINE)
        rtype = row['recruit_type']; stars = row['star_rating']
        if rtype == "HS" or rtype == "HS_Unranked":
            bonus_key = f"{stars}_star" if stars > 0 else "HS_unranked"
            score += config.STORMS_BONUS.get(bonus_key, 0)
        else:
            score += config.STORMS_BONUS.get(rtype, 0)
        return score
    df['points_storms'] = df.apply(get_storms_pts, axis=1)
    
    # Calculate 24/8 Sports Points
    def get_248_pts(row):
        rtype = row['recruit_type']; num_rank = row['numeric_rank']
        true_ovr_onz = row['true_overall_onz']; stars = row['star_rating']
        score = 0.0
        if rtype == "HS" and pd.notna(num_rank) and 0 < num_rank <= 400 :
             rank_score = max(0, 100.5 - (0.5 * num_rank) - (0.0007 * num_rank**2))
             score = min(100, rank_score)
        elif rtype in ["GT", "JUCO", "CPR"]:
            base_val = max(0, true_ovr_onz - config.TWENTYFOUR_EIGHT_TRANSFER_OVR_BASELINE)
            score = base_val * config.TWENTYFOUR_EIGHT_TRANSFER_POINTS_SCALE.get(rtype, 0.0)
        
        # Star bonus (can stack with HS rank points or Transfer points)
        if stars == 5: score += 15
        elif stars == 4: score += 7
        elif stars == 3: score += 3
        return score
    df['points_248sports'] = df.apply(get_248_pts, axis=1)

    # Select and order columns for the database
    final_recruit_cols = [
        'effective_season', 'recruiting_class_year', 'team_tid', 'recruit_name', 'position',
        'original_rank_str', 'numeric_rank', 'recruit_type', 'star_rating',
        'recruit_ovr', 'true_overall_onz', 'points_nspn', 'points_storms', 'points_248sports'
    ]
    # Ensure all defined columns exist, adding them with NaN if they were not created
    for col in final_recruit_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[final_recruit_cols] # Select and order

    print(f"Processed {len(df)} individual committed recruits.")
    return df

def calculate_team_recruiting_summary(processed_recruits_df, effective_season_to_calc):
    """
    Aggregates individual recruit data to create team-level recruiting summaries,
    including counts of each recruit type and KTV score (Z-score based).
    """
    expected_summary_cols = [
        'season', 'team_tid', 'num_recruits', 'avg_recruit_ovr', 'avg_numeric_rank', 
        'num_5_star', 'num_4_star', 'num_3_star', 'num_2_star', 'num_1_star', 'num_hs_unranked',
        'num_gt', 'num_juco', 'num_cpr', 
        'score_onz', 'score_nspn', 'score_storms', 'score_248sports', 'score_ktv'
    ]

    if processed_recruits_df.empty:
        print(f"WARNING: Processed recruits DataFrame is empty for team summary (season {effective_season_to_calc}).")
        return pd.DataFrame(columns=expected_summary_cols)

    recruits_for_season = processed_recruits_df[
        processed_recruits_df['effective_season'] == effective_season_to_calc
    ].copy()

    if recruits_for_season.empty:
        print(f"No recruits found for effective_season {effective_season_to_calc} to summarize.")
        return pd.DataFrame(columns=expected_summary_cols)

    # --- Calculate initial team scores and counts ---
    summary_list = []
    for team_tid, group in recruits_for_season.groupby('team_tid'):
        if pd.isna(team_tid): continue 

        num_5_star = group[group['star_rating'] == 5].shape[0]
        num_4_star = group[group['star_rating'] == 4].shape[0]
        num_3_star = group[group['star_rating'] == 3].shape[0]
        num_2_star = group[group['star_rating'] == 2].shape[0]
        num_1_star = group[group['star_rating'] == 1].shape[0]
        num_hs_unranked = group[(group['recruit_type'] == 'HS_Unranked') | ((group['recruit_type'] == 'HS') & (group['star_rating'] == 0))].shape[0]
        num_gt = group[group['recruit_type'] == 'GT'].shape[0]
        num_juco = group[group['recruit_type'] == 'JUCO'].shape[0]
        num_cpr = group[group['recruit_type'] == 'CPR'].shape[0]
        
        group_sorted_onz = group.sort_values(by='true_overall_onz', ascending=False).head(6)
        onz_score = 0.0; weights_onz = [1.0, 0.8, 0.31, 0.08, 0.01, 0.005]
        for i, (_, recruit) in enumerate(group_sorted_onz.iterrows()):
            if i < len(weights_onz): onz_score += recruit['true_overall_onz'] * weights_onz[i]
        
        nspn_score = group['points_nspn'].mean() if not group.empty and group['points_nspn'].notna().any() else 0.0
        storms_score = group['points_storms'].sum() if not group.empty else 0.0
        twentyfour_eight_score = group['points_248sports'].sum() if not group.empty else 0.0
        
        summary_list.append({
            'season': effective_season_to_calc, 'team_tid': int(team_tid),
            'num_recruits': len(group),
            'avg_recruit_ovr': group['recruit_ovr'].mean() if not group.empty and group['recruit_ovr'].notna().any() else 0.0,
            'avg_numeric_rank': group['numeric_rank'].dropna().mean() if not group.empty and group['numeric_rank'].notna().any() else np.nan,
            'num_5_star': num_5_star, 'num_4_star': num_4_star, 'num_3_star': num_3_star,
            'num_2_star': num_2_star, 'num_1_star': num_1_star, 'num_hs_unranked': num_hs_unranked,
            'num_gt': num_gt, 'num_juco': num_juco, 'num_cpr': num_cpr,
            'score_onz': onz_score, 'score_nspn': nspn_score,
            'score_storms': storms_score, 'score_248sports': twentyfour_eight_score
            # KTV score will be calculated after this initial summary_df is formed
        })
    
    team_summary_df = pd.DataFrame(summary_list)

    if team_summary_df.empty:
        print(f"No team recruiting summaries generated for season {effective_season_to_calc} before KTV calculation.")
        return pd.DataFrame(columns=expected_summary_cols)

    # --- Calculate KTV Score using Z-Scores ---
    service_score_cols = ['score_onz', 'score_nspn', 'score_storms', 'score_248sports']
    z_score_cols = []

    for col in service_score_cols:
        if col in team_summary_df.columns:
            mean_score = team_summary_df[col].mean()
            std_score = team_summary_df[col].std()
            z_col_name = f'z_{col}'
            if std_score == 0 or pd.isna(std_score): # Handle case where all scores are same or all NaN
                team_summary_df[z_col_name] = 0.0
            else:
                team_summary_df[z_col_name] = (team_summary_df[col] - mean_score) / std_score
            z_score_cols.append(z_col_name)
        else:
            print(f"Warning: Service score column {col} not found for KTV Z-score calculation.")

    if not z_score_cols: # If no service scores were available to make Z-scores
        team_summary_df['score_ktv'] = 0.0
    else:
        team_summary_df['ktv_avg_z'] = team_summary_df[z_score_cols].mean(axis=1, skipna=True)
        # Rescale KTV_Avg_Z_Score
        mean_ktv_avg_z = team_summary_df['ktv_avg_z'].mean()
        std_ktv_avg_z = team_summary_df['ktv_avg_z'].std()

        if std_ktv_avg_z == 0 or pd.isna(std_ktv_avg_z):
            team_summary_df['score_ktv'] = config.KTV_DESIRED_MEAN
        else:
            team_summary_df['score_ktv'] = config.KTV_DESIRED_MEAN + \
                ((team_summary_df['ktv_avg_z'] - mean_ktv_avg_z) / std_ktv_avg_z) * config.KTV_DESIRED_STD_DEV
        
        # Clean up intermediate Z-score columns if desired
        # team_summary_df.drop(columns=z_score_cols + ['ktv_avg_z'], inplace=True, errors='ignore')
    
    # Ensure all expected columns are present before returning
    for col in expected_summary_cols:
        if col not in team_summary_df.columns:
            if col == 'avg_numeric_rank': # This one can be NaN
                 team_summary_df[col] = np.nan
            else:
                 team_summary_df[col] = 0.0 # Default other missing to 0.0 or 0

    print(f"Calculated team recruiting summaries (including KTV) for {len(team_summary_df)} teams for season {effective_season_to_calc}.")
    return team_summary_df[expected_summary_cols] # Return with expected columns ordered
