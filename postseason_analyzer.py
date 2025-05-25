# postseason_analyzer.py
import pandas as pd
import numpy as np

def process_postseason_data(raw_df, teams_df, data_season):
    """
    Processes raw postseason results data from CSV.
    - Renames columns.
    - Adds season column.
    - Maps team names to team_tid.
    - Cleans seed and result data.
    """
    if raw_df.empty:
        print("WARNING: Raw postseason DataFrame is empty.")
        return pd.DataFrame()
    if teams_df.empty:
        print("WARNING: Teams DataFrame is empty. Cannot map team TIDs for postseason data.")
        return pd.DataFrame()
    if data_season is None:
        print("ERROR: data_season not provided for processing postseason data.")
        return pd.DataFrame()

    df = raw_df.copy()
    
    # Standardize column names based on user's CSV example: Event,Seed,Team,Result
    df.rename(columns={
        'Event': 'event_type',
        'Seed': 'seed',
        'Team': 'team_name_str',
        'Result': 'result'
    }, inplace=True)

    required_cols = ['event_type', 'seed', 'team_name_str', 'result']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"ERROR: Postseason CSV data missing essential columns after rename: {missing}")
        return pd.DataFrame()

    df['season'] = int(data_season)
    
    # Clean string columns
    df['event_type'] = df['event_type'].astype(str).str.strip().str.upper()
    df['team_name_str'] = df['team_name_str'].astype(str).str.strip()
    df['result'] = df['result'].astype(str).str.strip()

    # Map team_name_str to team_tid using teams_df (region is often the team name in your exports)
    team_name_to_tid_map = {}
    for _, row in teams_df.iterrows():
        tid = row['tid']
        # Prioritize matching against 'region' as it's often the display name
        if pd.notna(row['region']):
            team_name_to_tid_map[str(row['region']).strip().lower()] = tid
        # Fallback to 'full_name'
        if pd.notna(row['full_name']) and str(row['full_name']).strip().lower() not in team_name_to_tid_map:
             team_name_to_tid_map[str(row['full_name']).strip().lower()] = tid
        # Fallback to 'abbrev'
        if pd.notna(row['abbrev']) and str(row['abbrev']).strip().lower() not in team_name_to_tid_map:
            team_name_to_tid_map[str(row['abbrev']).strip().lower()] = tid
    
    def map_team_id(name_str):
        if pd.isna(name_str) or str(name_str).strip() == "": return np.nan
        normalized_name = str(name_str).strip().lower()
        return team_name_to_tid_map.get(normalized_name, np.nan)

    df['team_tid'] = df['team_name_str'].apply(map_team_id)
    
    unmapped_teams = df[df['team_tid'].isna()]['team_name_str'].unique()
    if len(unmapped_teams) > 0:
        print(f"WARNING: Could not map the following team names from Postseason CSV to tids: {list(unmapped_teams)}")
    
    df.dropna(subset=['team_tid'], inplace=True) # Remove rows where team_tid could not be mapped
    if df.empty:
        print("WARNING: No valid postseason entries after team mapping.")
        return pd.DataFrame()
        
    df['team_tid'] = df['team_tid'].astype(int)

    # Clean seed - convert to integer, allow for NaNs if seed is missing or not applicable
    df['seed'] = pd.to_numeric(df['seed'], errors='coerce').astype('Int64') # Pandas nullable integer

    # Filter for known event types if necessary, e.g., NT, NIT
    df = df[df['event_type'].isin(['NT', 'NIT'])] # Or other known types

    final_cols = ['season', 'team_tid', 'event_type', 'seed', 'result']
    df_processed = df[final_cols].copy()
    
    print(f"Processed {len(df_processed)} valid postseason entries.")
    return df_processed
