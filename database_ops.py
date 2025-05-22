# database_ops.py
import sqlite3
import pandas as pd
import numpy as np
import config

DB_PATH = config.DATABASE_FILE_PATH

def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        print(f"DATABASE ERROR: Could not connect to database at {DB_PATH}: {e}")
    return conn

def create_tables_if_not_exist():
    conn = get_db_connection()
    if conn is None:
        print("DATABASE ERROR: No connection for create_tables_if_not_exist.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Teams (
            team_tid INTEGER PRIMARY KEY, cid INTEGER, abbrev TEXT UNIQUE,
            name TEXT, region TEXT, full_name TEXT
        );""")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS GameTeamStats (
            gid INTEGER, season INTEGER, team_tid INTEGER, opponent_tid INTEGER,
            team_abbrev TEXT, opponent_abbrev TEXT, 
            location TEXT, overtimes INTEGER, is_national_playoffs BOOLEAN,
            is_conf_tournament BOOLEAN, team_game_num_in_season INTEGER,
            team_score_official INTEGER, opponent_score_official INTEGER,
            designated_home_tid INTEGER, designated_away_tid INTEGER,
            team_pts REAL, team_poss REAL, team_fgm REAL, team_fga REAL, team_fgm3 REAL, team_fga3 REAL,
            team_ftm REAL, team_fta REAL, team_oreb REAL, team_dreb REAL, team_tov REAL, team_ast REAL,
            team_stl REAL, team_blk REAL, team_pf REAL, team_min REAL,
            opponent_pts REAL, opp_poss REAL, opponent_fgm REAL, opponent_fga REAL, opponent_fgm3 REAL, opponent_fga3 REAL,
            opponent_ftm REAL, opponent_fta REAL, opponent_oreb REAL, opponent_dreb REAL, opponent_tov REAL,
            opponent_ast REAL, opponent_stl REAL, opponent_blk REAL, opponent_pf REAL,
            raw_oe REAL, raw_de REAL,
            off_efg_pct REAL, off_tov_pct REAL, off_orb_pct REAL, off_ft_rate REAL,
            def_efg_pct REAL, def_tov_pct REAL, def_opp_orb_pct REAL, def_ft_rate REAL,
            team_drb_pct REAL, team_2p_pct REAL, opp_2p_pct REAL,
            team_3p_pct REAL, opp_3p_pct REAL, team_3p_rate REAL, opp_3p_rate REAL,
            win INTEGER, loss INTEGER,
            PRIMARY KEY (gid, season, team_tid) 
        );""")

        # --- EXPANDED SeasonTeamSummaries Schema ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS SeasonTeamSummaries (
            season INTEGER, 
            team_tid INTEGER, 
            team_abbrev TEXT, 
            cid INTEGER, 
            games_played INTEGER, 
            wins INTEGER, 
            losses INTEGER, 
            win_pct REAL,
            
            /* Season Aggregated Raw Totals (from groupby in calculate_season_team_aggregates) */
            team_pts REAL, team_poss REAL, 
            team_fgm REAL, team_fga REAL, team_fgm3 REAL, team_fga3 REAL, 
            team_ftm REAL, team_fta REAL, team_oreb REAL, team_dreb REAL, 
            team_tov REAL, team_ast REAL, team_stl REAL, team_blk REAL, team_pf REAL,
            opponent_pts REAL, opp_poss REAL, /* Note: opp_poss name from game_team_stats_df */
            opponent_fgm REAL, opponent_fga REAL, opponent_fgm3 REAL, opponent_fga3 REAL,
            opponent_ftm REAL, opponent_fta REAL, opponent_oreb REAL, opponent_dreb REAL,
            opponent_tov REAL, opponent_ast REAL, opponent_stl REAL, opponent_blk REAL, opponent_pf REAL,

            /* Calculated Season Raw Rates */
            raw_oe REAL, raw_de REAL, raw_em REAL, avg_tempo REAL,
            off_efg_pct REAL, off_tov_pct REAL, off_orb_pct REAL, off_ft_rate REAL, 
            def_efg_pct REAL, def_tov_pct REAL, def_opp_orb_pct REAL, def_ft_rate REAL, 
            team_drb_pct REAL, 
            team_2p_pct REAL, opp_2p_pct REAL, 
            team_3p_pct REAL, opp_3p_pct REAL, 
            team_3p_rate REAL, opp_3p_rate REAL, 
            
            /* Luck */
            expected_wins_adj REAL, luck_adj REAL,
            
            /* RPI/SOS */
            rpi_wp REAL, owp REAL, oowp REAL, rpi REAL, sos_bcs REAL,
            
            /* Adjusted Efficiencies */
            adj_o REAL, adj_d REAL, adj_em REAL,
            
            /* Adjusted SOS */
            avg_opp_adj_o REAL, avg_opp_adj_d REAL, avg_opp_adj_em REAL, avg_nonconf_opp_adj_em REAL,
            
            /* Ranking and Quadrants */
            rank_adj_em INTEGER,
            q1_w INTEGER, q1_l INTEGER, q2_w INTEGER, q2_l INTEGER,
            q3_w INTEGER, q3_l INTEGER, q4_w INTEGER, q4_l INTEGER,
            
            /* WAB */
            wab REAL,
            
            PRIMARY KEY (season, team_tid)
        );
        """)
        conn.commit()
        print("Database tables checked/created (SeasonTeamSummaries schema expanded).")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR during table creation: {e}")
    finally:
        if conn:
            conn.close()

def save_teams_df_to_db(teams_df):
    if teams_df.empty:
        print("Teams DataFrame is empty, nothing to save to DB.")
        return
    conn = get_db_connection()
    if conn is None: return
    try:
        df_to_save = teams_df.rename(columns={'tid': 'team_tid'})
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Teams)")
        table_cols = [info[1] for info in cursor.fetchall()]
        df_cols_in_table = [col for col in df_to_save.columns if col in table_cols]
        df_to_save_filtered = df_to_save[df_cols_in_table]
        df_to_save_filtered.to_sql('Teams', conn, if_exists='replace', index=False, chunksize=1000)
        print(f"Saved/Replaced {len(df_to_save_filtered)} teams in the database.")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR saving teams_df: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR saving teams_df: {ex}")
    finally:
        if conn:
            conn.close()

def save_game_team_stats_to_db(game_team_stats_df):
    # ... (This function remains the same as the last version, it saves columns present in GameTeamStats table) ...
    if game_team_stats_df.empty:
        print("GameTeamStats DataFrame is empty, nothing to save to DB.")
        return
    conn = get_db_connection()
    if conn is None: return
    cursor = conn.cursor()
    try:
        df_to_save = game_team_stats_df.copy()
        if 'win' not in df_to_save.columns or 'loss' not in df_to_save.columns:
            df_to_save['team_score_official'] = pd.to_numeric(df_to_save.get('team_score_official'), errors='coerce').fillna(0)
            df_to_save['opponent_score_official'] = pd.to_numeric(df_to_save.get('opponent_score_official'), errors='coerce').fillna(0)
            df_to_save['win'] = (df_to_save['team_score_official'] > df_to_save['opponent_score_official']).astype(int)
            df_to_save['loss'] = (df_to_save['team_score_official'] < df_to_save['opponent_score_official']).astype(int)
        cursor.execute("PRAGMA table_info(GameTeamStats)")
        table_cols_info = cursor.fetchall()
        table_cols = [info[1] for info in table_cols_info]
        df_cols_in_table_order = [col for col in table_cols if col in df_to_save.columns]
        df_to_save_filtered = df_to_save[df_cols_in_table_order]
        cols_filtered_str = ', '.join(f'"{col}"' for col in df_to_save_filtered.columns)
        placeholders_filtered_str = ', '.join(['?'] * len(df_to_save_filtered.columns))
        sql = f"INSERT OR IGNORE INTO GameTeamStats ({cols_filtered_str}) VALUES ({placeholders_filtered_str})"
        num_inserted = 0
        num_attempted = 0
        # print(f"DEBUG DB_SAVE_GAMES: Attempting to insert {len(df_to_save_filtered)} rows. Using columns: {df_to_save_filtered.columns.tolist()}") # Can be verbose
        tuples_to_insert = [tuple(x) for x in df_to_save_filtered.to_numpy()]
        if tuples_to_insert and len(tuples_to_insert[0]) != len(df_to_save_filtered.columns):
             print(f"ERROR DB_SAVE_GAMES: Mismatch between tuple length {len(tuples_to_insert[0])} and column count {len(df_to_save_filtered.columns)}")
             if conn: conn.close() # Close connection before returning
             return
        for i, row_tuple in enumerate(tuples_to_insert):
            num_attempted += 1
            # if i < 2: print(f"DEBUG DB_SAVE_GAMES: Tuple {i}: {row_tuple}") # Can be verbose
            try:
                cursor.execute(sql, row_tuple)
                if cursor.rowcount > 0: num_inserted += 1
            except sqlite3.Error as e:
                print(f"DATABASE ERROR inserting game row (gid,season,tid might be {row_tuple[:3]}): {e}")
        conn.commit()
        print(f"Attempted to save {num_attempted} game-team rows. Newly inserted: {num_inserted}.")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR in save_game_team_stats_to_db: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR in save_game_team_stats_to_db: {ex}")
    finally:
        if conn:
            conn.close()

def save_season_summary_to_db(season_team_summary_df, current_season):
    if season_team_summary_df.empty:
        print("Season summary DataFrame is empty, nothing to save to DB.")
        return
    if current_season is None:
        print("ERROR: current_season is None, cannot save season summary.")
        return

    conn = get_db_connection()
    if conn is None: return
    cursor = conn.cursor()
    try:
        season_val = int(current_season)
        cursor.execute("DELETE FROM SeasonTeamSummaries WHERE season = ?", (season_val,))
        conn.commit()
        # print(f"Deleted existing summary for season {season_val}. ({cursor.rowcount} rows affected by delete)") # Can be verbose
        
        df_to_save = season_team_summary_df.copy()
        
        cursor.execute("PRAGMA table_info(SeasonTeamSummaries)")
        table_cols_info = cursor.fetchall()
        table_cols_db = [info[1] for info in table_cols_info] # Columns defined in DB
        
        # Ensure all columns defined in the DB table are present in the DataFrame.
        # If a column from table_cols_db is missing in df_to_save, add it with a default.
        for tc in table_cols_db:
            if tc not in df_to_save.columns:
                print(f"WARNING DB_SAVE_SUMMARY: Column '{tc}' defined in DB but missing in season_summary_df. Adding as NaN/0 before save.")
                # Determine a sensible default based on typical column content
                if any(kw in tc for kw in ['_w', '_l', 'games_played', 'wins', 'losses', 'rank_adj_em', 'cid', 'season', 'team_tid', 'overtimes', 'is_national_playoffs', 'is_conf_tournament', 'team_game_num_in_season', 'team_score_official', 'opponent_score_official']):
                    df_to_save[tc] = 0
                else: # For most REAL valued stats, NaN is fine, or 0.0
                    df_to_save[tc] = np.nan
        
        # Filter DataFrame to only include columns that are in the table, in table order
        # This ensures no extra columns from DataFrame are attempted to be saved if not in DB schema.
        df_cols_in_table_order = [col for col in table_cols_db if col in df_to_save.columns]
        df_to_save_filtered = df_to_save[df_cols_in_table_order]

        print(f"DEBUG DB_SAVE_SUMMARY: Columns being saved to SeasonTeamSummaries: {df_to_save_filtered.columns.tolist()}")
        if not df_to_save_filtered.empty:
             # Convert boolean columns to int (0 or 1) for SQLite if they exist and are still bool
             # Note: This step might be redundant if booleans are already handled by to_sql or earlier conversions.
             # for bool_col in ['is_national_playoffs', 'is_conf_tournament']:
             #     if bool_col in df_to_save_filtered.columns and df_to_save_filtered[bool_col].dtype == 'bool':
             #         df_to_save_filtered[bool_col] = df_to_save_filtered[bool_col].astype(int)
             
             df_to_save_filtered.to_sql('SeasonTeamSummaries', conn, if_exists='append', index=False, chunksize=1000)
             print(f"Saved {len(df_to_save_filtered)} team summaries for season {season_val} to the database.")
        else:
             print("DEBUG DB_SAVE_SUMMARY: df_to_save_filtered was empty (likely due to column mismatch or no data). Nothing to save.")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR saving season_summary_df for season {current_season}: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR saving season_summary_df for season {current_season}: {ex}")
    finally:
        if conn:
            conn.close()

def load_all_game_stats_for_season(season_to_load):
    # ... (This function remains the same as the last version) ...
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()
    df = pd.DataFrame()
    try:
        season_val = 0
        if season_to_load is not None:
            try: season_val = int(season_to_load)
            except (ValueError, TypeError): print(f"ERROR DB_LOAD: season_to_load '{season_to_load}' invalid."); return df
        else: print(f"ERROR DB_LOAD: season_to_load is None."); return df
        query = "SELECT * FROM GameTeamStats WHERE season = ?"
        # print(f"DEBUG DB_LOAD: Attempting to load games for season: {season_val} (type: {type(season_val)})")
        # print(f"DEBUG DB_LOAD: Executing query: {query} with params ({season_val},)")
        df = pd.read_sql_query(query, conn, params=(season_val,))
        print(f"Loaded {len(df)} game-team stat rows for season {season_val} from database.")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR loading game stats for season {season_to_load}: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR loading game stats for season {season_to_load}: {ex}")
    finally:
        if conn:
            conn.close()
    return df
