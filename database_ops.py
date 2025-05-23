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

def create_tables_if_not_exist(): # THIS IS THE FUNCTION
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

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Recruits (
            recruit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            effective_season INTEGER NOT NULL,      
            recruiting_class_year INTEGER NOT NULL, 
            team_tid INTEGER,                       
            recruit_name TEXT NOT NULL,
            position TEXT,
            original_rank_str TEXT,                 
            numeric_rank INTEGER,                   
            recruit_type TEXT,                      
            star_rating INTEGER,                    
            recruit_ovr INTEGER,
            true_overall_onz REAL,
            points_nspn REAL,
            points_storms REAL,
            points_248sports REAL,
            FOREIGN KEY (team_tid) REFERENCES Teams(team_tid),
            UNIQUE(recruiting_class_year, recruit_name, team_tid) 
        );""")

        cursor.execute(""" DROP TABLE IF EXISTS SeasonTeamSummaries; """)
        cursor.execute("""
        CREATE TABLE SeasonTeamSummaries (
            season INTEGER, team_tid INTEGER, team_abbrev TEXT, cid INTEGER, 
            games_played INTEGER, wins INTEGER, losses INTEGER, win_pct REAL,
            team_pts REAL, team_poss REAL, team_fgm REAL, team_fga REAL, team_fgm3 REAL, team_fga3 REAL, 
            team_ftm REAL, team_fta REAL, team_oreb REAL, team_dreb REAL, 
            team_tov REAL, team_ast REAL, team_stl REAL, team_blk REAL, team_pf REAL,
            opponent_pts REAL, opp_poss REAL, opponent_fgm REAL, opponent_fga REAL, opponent_fgm3 REAL, opponent_fga3 REAL,
            opponent_ftm REAL, opponent_fta REAL, opponent_oreb REAL, opponent_dreb REAL,
            opponent_tov REAL, opponent_ast REAL, opponent_stl REAL, opponent_blk REAL, opponent_pf REAL,
            raw_oe REAL, raw_de REAL, raw_em REAL, avg_tempo REAL,
            off_efg_pct REAL, off_tov_pct REAL, off_orb_pct REAL, off_ft_rate REAL, 
            def_efg_pct REAL, def_tov_pct REAL, def_opp_orb_pct REAL, def_ft_rate REAL, 
            team_drb_pct REAL, team_2p_pct REAL, opp_2p_pct REAL, 
            team_3p_pct REAL, opp_3p_pct REAL, team_3p_rate REAL, opp_3p_rate REAL, 
            expected_wins_adj REAL, luck_adj REAL,
            rpi_wp REAL, owp REAL, oowp REAL, rpi REAL, sos_bcs REAL,
            adj_o REAL, adj_d REAL, adj_em REAL,
            avg_opp_adj_o REAL, avg_opp_adj_d REAL, avg_opp_adj_em REAL, avg_nonconf_opp_adj_em REAL,
            rank_adj_em INTEGER,
            q1_w INTEGER, q1_l INTEGER, q2_w INTEGER, q2_l INTEGER,
            q3_w INTEGER, q3_l INTEGER, q4_w INTEGER, q4_l INTEGER,
            wab REAL, 
            num_recruits INTEGER, avg_recruit_ovr REAL, avg_numeric_rank REAL,
            num_5_star INTEGER, num_4_star INTEGER, num_3_star INTEGER,
            num_2_star INTEGER, num_1_star INTEGER, num_hs_unranked INTEGER, 
            num_gt INTEGER, num_juco INTEGER, num_cpr INTEGER,
            score_onz REAL, score_nspn REAL, score_storms REAL, score_248sports REAL,
            score_ktv REAL, 
            PRIMARY KEY (season, team_tid)
        );""")
        conn.commit()
        print("Database tables checked/created (Recruits added, SeasonTeamSummaries expanded for all recruit counts and KTV).")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR during table creation: {e}")
    finally:
        if conn: conn.close()

def save_teams_df_to_db(teams_df):
    if teams_df.empty: print("Teams DataFrame is empty, nothing to save to DB."); return
    conn = get_db_connection();
    if conn is None: return
    try:
        df_to_save = teams_df.rename(columns={'tid': 'team_tid'})
        cursor = conn.cursor(); cursor.execute("PRAGMA table_info(Teams)")
        table_cols = [info[1] for info in cursor.fetchall()]
        df_cols_in_table = [col for col in df_to_save.columns if col in table_cols]
        df_to_save_filtered = df_to_save[df_cols_in_table]
        df_to_save_filtered.to_sql('Teams', conn, if_exists='replace', index=False, chunksize=1000)
        print(f"Saved/Replaced {len(df_to_save_filtered)} teams in the database.")
    except Exception as e: print(f"ERROR saving teams_df: {e}")
    finally:
        if conn: conn.close()

def save_game_team_stats_to_db(game_team_stats_df):
    if game_team_stats_df.empty: print("GameTeamStats DataFrame is empty, nothing to save to DB."); return
    conn = get_db_connection();
    if conn is None: return
    cursor = conn.cursor()
    try:
        df_to_save = game_team_stats_df.copy()
        if 'win' not in df_to_save.columns or 'loss' not in df_to_save.columns:
            df_to_save['team_score_official'] = pd.to_numeric(df_to_save.get('team_score_official'), errors='coerce').fillna(0)
            df_to_save['opponent_score_official'] = pd.to_numeric(df_to_save.get('opponent_score_official'), errors='coerce').fillna(0)
            df_to_save['win'] = (df_to_save['team_score_official'] > df_to_save['opponent_score_official']).astype(int)
            df_to_save['loss'] = (df_to_save['team_score_official'] < df_to_save['opponent_score_official']).astype(int)
        cursor.execute("PRAGMA table_info(GameTeamStats)"); table_cols_info = cursor.fetchall()
        table_cols = [info[1] for info in table_cols_info]
        df_cols_in_table_order = [col for col in table_cols if col in df_to_save.columns]
        df_to_save_filtered = df_to_save[df_cols_in_table_order]
        cols_filtered_str = ', '.join(f'"{col}"' for col in df_to_save_filtered.columns)
        placeholders_filtered_str = ', '.join(['?'] * len(df_to_save_filtered.columns))
        sql = f"INSERT OR IGNORE INTO GameTeamStats ({cols_filtered_str}) VALUES ({placeholders_filtered_str})"
        num_inserted = 0; num_attempted = 0
        tuples_to_insert = [tuple(x) for x in df_to_save_filtered.to_numpy()]
        if tuples_to_insert and len(tuples_to_insert[0]) != len(df_to_save_filtered.columns):
             print(f"ERROR DB_SAVE_GAMES: Tuple/Column mismatch"); conn.close(); return
        for row_tuple in tuples_to_insert:
            num_attempted += 1
            try:
                cursor.execute(sql, row_tuple)
                if cursor.rowcount > 0: num_inserted += 1
            except sqlite3.Error as e: print(f"DATABASE ERROR inserting game row: {e}")
        conn.commit()
        print(f"Attempted to save {num_attempted} game-team rows. Newly inserted: {num_inserted}.")
    except Exception as e: print(f"GENERAL ERROR in save_game_team_stats_to_db: {e}")
    finally:
        if conn: conn.close()

def save_processed_recruits(processed_recruits_df, class_year_being_processed):
    if processed_recruits_df.empty:
        print(f"Processed recruits DataFrame for class {class_year_being_processed} is empty, nothing to save.")
        return
    conn = get_db_connection();
    if conn is None: return
    cursor = conn.cursor()
    try:
        df_to_save = processed_recruits_df.copy()
        if 'recruiting_class_year' not in df_to_save.columns:
            df_to_save['recruiting_class_year'] = class_year_being_processed
        print(f"Deleting existing recruits for class year {class_year_being_processed} before insert...")
        cursor.execute("DELETE FROM Recruits WHERE recruiting_class_year = ?", (int(class_year_being_processed),)) # Ensure int
        conn.commit()
        print(f"Deleted {cursor.rowcount} old recruits for class year {class_year_being_processed}.")
        cursor.execute("PRAGMA table_info(Recruits)"); table_cols = [info[1] for info in cursor.fetchall()]
        df_cols_in_table = [col for col in table_cols if col in df_to_save.columns and col != 'recruit_id']
        df_to_save_filtered = df_to_save[df_cols_in_table]
        if not df_to_save_filtered.empty:
            df_to_save_filtered.to_sql('Recruits', conn, if_exists='append', index=False, chunksize=500)
            print(f"Saved {len(df_to_save_filtered)} processed recruits for class {class_year_being_processed} to database.")
        else: print(f"No columns in processed_recruits_df matched Recruits table schema for saving (class {class_year_being_processed}). Current df_to_save columns: {df_to_save.columns.tolist()}")
    except Exception as e: print(f"ERROR saving processed recruits for class {class_year_being_processed}: {e}")
    finally:
        if conn: conn.close()

def save_season_summary_to_db(season_team_summary_df, current_season):
    if season_team_summary_df.empty: print("Season summary DataFrame is empty, nothing to save."); return
    if current_season is None: print("ERROR: current_season is None for save_season_summary."); return
    conn = get_db_connection();
    if conn is None: return
    cursor = conn.cursor()
    try:
        season_val = int(current_season)
        cursor.execute("DELETE FROM SeasonTeamSummaries WHERE season = ?", (season_val,))
        conn.commit()
        df_to_save = season_team_summary_df.copy()
        cursor.execute("PRAGMA table_info(SeasonTeamSummaries)"); table_cols_info = cursor.fetchall()
        table_cols_db = [info[1] for info in table_cols_info]
        for tc in table_cols_db:
            if tc not in df_to_save.columns:
                is_count_like = any(kw in tc for kw in ['_w', '_l', 'games_played', 'wins', 'losses', 'rank_adj_em', 'cid', 'season', 'team_tid', 'num_'])
                df_to_save[tc] = 0 if is_count_like else np.nan
        df_cols_in_table_order = [col for col in table_cols_db if col in df_to_save.columns]
        df_to_save_filtered = df_to_save[df_cols_in_table_order]
        # print(f"DEBUG DB_SAVE_SUMMARY: Columns being saved to SeasonTeamSummaries: {df_to_save_filtered.columns.tolist()}") # Optional debug
        if not df_to_save_filtered.empty:
             df_to_save_filtered.to_sql('SeasonTeamSummaries', conn, if_exists='append', index=False, chunksize=1000)
             print(f"Saved {len(df_to_save_filtered)} team summaries for season {season_val} to DB.")
    except Exception as e: print(f"ERROR saving season_summary_df for season {current_season}: {e}")
    finally:
        if conn: conn.close()

def load_all_game_stats_for_season(season_to_load):
    conn = get_db_connection();
    if conn is None: return pd.DataFrame()
    df = pd.DataFrame()
    try:
        season_val = int(season_to_load) if season_to_load is not None else None
        if season_val is None: print(f"ERROR DB_LOAD: season_to_load is None."); return df
        query = "SELECT * FROM GameTeamStats WHERE season = ?"
        df = pd.read_sql_query(query, conn, params=(season_val,))
        print(f"Loaded {len(df)} game-team stat rows for season {season_val} from database.")
    except Exception as e: print(f"GENERAL ERROR loading game stats for season {season_to_load}: {e}")
    finally:
        if conn: conn.close()
    return df

def load_all_recruits_for_effective_season(effective_season_to_load):
    conn = get_db_connection();
    if conn is None: return pd.DataFrame()
    df = pd.DataFrame()
    try:
        season_val = int(effective_season_to_load)
        query = "SELECT * FROM Recruits WHERE effective_season = ?"
        df = pd.read_sql_query(query, conn, params=(season_val,))
        print(f"Loaded {len(df)} recruits for effective_season {season_val} from database.")
    except Exception as e: print(f"Error loading recruits for effective_season {effective_season_to_load} from database: {e}")
    finally:
        if conn: conn.close()
    return df
