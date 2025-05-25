# database_ops.py
import sqlite3
import pandas as pd
import numpy as np
import config

DB_PATH = config.DATABASE_FILE_PATH

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        print(f"DATABASE ERROR: Could not connect to database at {DB_PATH}: {e}")
    return conn

def create_tables_if_not_exist():
    """Creates all necessary tables in the SQLite database if they don't already exist."""
    conn = get_db_connection()
    if conn is None:
        print("DATABASE ERROR: No connection for create_tables_if_not_exist.")
        return

    try:
        cursor = conn.cursor()
        # Teams Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Teams (
            team_tid INTEGER PRIMARY KEY, cid INTEGER, abbrev TEXT UNIQUE,
            name TEXT, region TEXT, full_name TEXT
        );""")
        
        # GameTeamStats Table - UPDATED for future games & predictions
        cursor.execute("""DROP TABLE IF EXISTS GameTeamStats;""")
        cursor.execute("""
        CREATE TABLE GameTeamStats ( 
            gid INTEGER, season INTEGER, team_tid INTEGER, opponent_tid INTEGER,
            team_abbrev TEXT, opponent_abbrev TEXT, 
            location TEXT, overtimes INTEGER, 
            is_played BOOLEAN NOT NULL DEFAULT FALSE,      /* NEW */
            game_date TEXT,                                /* NEW (can be NULL) */
            game_time TEXT,                                /* NEW (can be NULL) */
            is_national_playoffs BOOLEAN,
            is_conf_tournament BOOLEAN, 
            team_game_num_in_season INTEGER,
            team_score_official INTEGER,          /* NULL for future games */
            opponent_score_official INTEGER,      /* NULL for future games */
            designated_home_tid INTEGER, 
            designated_away_tid INTEGER,
            /* Actual stats - will be NULL for future games initially */
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
            win INTEGER, loss INTEGER,             /* NULL for future games */
            game_quadrant TEXT,                   /* Can be calculated for future games too */
            /* Prediction columns - will be NULL for played games */
            pred_win_prob_team REAL,                    
            pred_margin_team REAL,                      
            pred_score_team INTEGER,                    
            pred_score_opponent INTEGER,                
            PRIMARY KEY (gid, season, team_tid) 
        );""")

        # Recruits Table
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

        # Coach Tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Coaches (
            coach_id INTEGER PRIMARY KEY AUTOINCREMENT,
            coach_name TEXT UNIQUE NOT NULL
        );""")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS CoachAssignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            coach_id INTEGER NOT NULL,
            team_tid INTEGER NOT NULL,
            season INTEGER NOT NULL,
            FOREIGN KEY (coach_id) REFERENCES Coaches(coach_id),
            FOREIGN KEY (team_tid) REFERENCES Teams(team_tid),
            UNIQUE (team_tid, season) 
        );""")
        
        # SeasonTeamSummaries Table
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
            score_ktv REAL, nt_seed INTEGER, nt_result TEXT, nit_seed INTEGER, nit_result TEXT, 
            PRIMARY KEY (season, team_tid)
        );""")

        # Coach Stat Tables
        cursor.execute("DROP TABLE IF EXISTS CoachSeasonStats;")
        cursor.execute("""
        CREATE TABLE CoachSeasonStats (
            coach_id INTEGER NOT NULL, coach_name TEXT, season INTEGER NOT NULL,
            team_tid INTEGER NOT NULL, team_abbrev TEXT,
            games_coached INTEGER, wins INTEGER, losses INTEGER, win_pct REAL,
            q1_w INTEGER, q1_l INTEGER, q2_w INTEGER, q2_l INTEGER,
            q3_w INTEGER, q3_l INTEGER, q4_w INTEGER, q4_l INTEGER,
            team_adj_em REAL, team_rank_adj_em INTEGER,
            team_raw_oe REAL, team_raw_de REAL, team_avg_tempo REAL,
            num_recruits INTEGER, avg_recruit_ovr REAL, 
            num_5_star INTEGER, num_4_star INTEGER, num_3_star INTEGER,
            num_2_star INTEGER, num_1_star INTEGER, num_hs_unranked INTEGER,
            num_gt INTEGER, num_juco INTEGER, num_cpr INTEGER,
            score_onz REAL, score_nspn REAL, score_storms REAL, score_248sports REAL, score_ktv REAL,
            PRIMARY KEY (coach_id, season, team_tid),
            FOREIGN KEY (coach_id) REFERENCES Coaches(coach_id),
            FOREIGN KEY (team_tid) REFERENCES Teams(team_tid)
        );""")
        
        cursor.execute("DROP TABLE IF EXISTS CoachCareerStats;")
        cursor.execute("""
        CREATE TABLE CoachCareerStats (
            coach_id INTEGER PRIMARY KEY, coach_name TEXT, 
            seasons_coached INTEGER, teams_coached_count INTEGER,
            total_games_coached INTEGER, total_wins INTEGER, total_losses INTEGER, career_win_pct REAL,
            career_q1_w INTEGER, career_q1_l INTEGER, career_q2_w INTEGER, career_q2_l INTEGER,
            career_q3_w INTEGER, career_q3_l INTEGER, career_q4_w INTEGER, career_q4_l INTEGER,
            career_avg_team_adj_em REAL, career_avg_team_rank REAL,
            career_total_recruits INTEGER, career_avg_recruit_ovr_of_classes REAL,
            career_total_5_stars INTEGER, career_total_4_stars INTEGER, career_total_3_stars INTEGER,
            career_total_2_stars INTEGER, career_total_1_stars INTEGER, career_total_hs_unranked INTEGER,
            career_total_gt INTEGER, career_total_juco INTEGER, career_total_cpr INTEGER,
            career_avg_score_onz REAL, career_avg_score_nspn REAL,
            career_avg_score_storms REAL, career_avg_score_248sports REAL, career_avg_score_ktv REAL,
            FOREIGN KEY (coach_id) REFERENCES Coaches(coach_id)
        );""")
        
        cursor.execute("""DROP TABLE IF EXISTS CoachHeadToHeadStats;""")
        cursor.execute("""
        CREATE TABLE CoachHeadToHeadStats (
            season INTEGER NOT NULL,
            coach1_id INTEGER NOT NULL,
            coach2_id INTEGER NOT NULL, 
            coach1_wins INTEGER DEFAULT 0,
            coach2_wins INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0, 
            PRIMARY KEY (season, coach1_id, coach2_id),
            FOREIGN KEY (coach1_id) REFERENCES Coaches(coach_id),
            FOREIGN KEY (coach2_id) REFERENCES Coaches(coach_id),
            CHECK (coach1_id < coach2_id) 
        );""")
        conn.commit()
        cursor.execute("""DROP TABLE IF EXISTS PostseasonResults;""")
        cursor.execute("""
        CREATE TABLE PostseasonResults (
            postseason_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            team_tid INTEGER NOT NULL,
            event_type TEXT NOT NULL, -- 'NT' or 'NIT'
            seed INTEGER,             -- Can be NULL
            result TEXT,              -- e.g., 'Champion', 'Final Four', 'Round 1'
            FOREIGN KEY (team_tid) REFERENCES Teams(team_tid),
            UNIQUE (season, team_tid, event_type) 
        );""")
        conn.commit()
        print("Database tables checked/created (PostseasonResults table added).")
    except sqlite3.Error as e:
        print(f"DATABASE ERROR during table creation: {e}")
    finally:
        if conn: conn.close()

def _get_or_create_coach_id(coach_name, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT coach_id FROM Coaches WHERE coach_name = ?", (coach_name,))
    data = cursor.fetchone()
    if data: return data[0]
    else:
        try:
            cursor.execute("INSERT INTO Coaches (coach_name) VALUES (?)", (coach_name,)); return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT coach_id FROM Coaches WHERE coach_name = ?", (coach_name,)); data=cursor.fetchone()
            return data[0] if data else None
        except sqlite3.Error as e: print(f"DB ERROR in _get_or_create_coach_id for {coach_name}: {e}"); return None

def save_coach_data(processed_coach_assignments_df):
    if processed_coach_assignments_df.empty: print("Processed coach assignments DataFrame is empty."); return
    conn = get_db_connection();
    if conn is None: return
    try:
        with conn:
            cursor = conn.cursor()
            seasons_in_csv = processed_coach_assignments_df['season'].unique().tolist()
            if not seasons_in_csv: print("No seasons in processed coach assignments."); return
            for season_to_clear in seasons_in_csv:
                cursor.execute("DELETE FROM CoachAssignments WHERE season = ?", (int(season_to_clear),))
            num_inserted = 0
            for _, row in processed_coach_assignments_df.iterrows():
                coach_name = str(row['coach_name']).strip(); team_tid = int(row['team_tid']); season = int(row['season'])
                if not coach_name: continue
                coach_id = _get_or_create_coach_id(coach_name, conn)
                if coach_id is not None:
                    try:
                        cursor.execute("INSERT INTO CoachAssignments (coach_id, team_tid, season) VALUES (?, ?, ?)", (coach_id, team_tid, season))
                        num_inserted += 1
                    except sqlite3.IntegrityError: print(f"WARN: UNIQUE constraint fail for coach assign: C:{coach_id} T:{team_tid} S:{season}.")
                    except sqlite3.Error as e: print(f"DB ERROR inserting coach assignment ({coach_name}, {team_tid}, {season}): {e}")
            print(f"Attempted {len(processed_coach_assignments_df)} coach assignments. Inserted/Replaced: {num_inserted}.")

    except Exception as e: print(f"GENERAL ERROR in save_coach_data: {e}")
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

# --- MODIFIED save_game_team_stats_to_db ---
def save_game_team_stats_to_db(all_games_for_season_df, season_to_clear_and_insert):
    """
    Saves all games (played and future, potentially with predictions) for a given season.
    Deletes ALL existing games for that season first, then inserts all new rows.
    """
    if season_to_clear_and_insert is None:
        print("ERROR: season_to_clear_and_insert is None. Cannot save game stats.")
        return
    
    season_val = int(season_to_clear_and_insert)

    conn = get_db_connection()
    if conn is None: return
    
    try:
        with conn:
            cursor = conn.cursor()
            
            print(f"Deleting existing game data for season {season_val} from GameTeamStats before insert...")
            cursor.execute("DELETE FROM GameTeamStats WHERE season = ?", (season_val,))
            print(f"Deleted {cursor.rowcount} old game-team rows for season {season_val}.")

            if all_games_for_season_df.empty:
                print(f"Input all_games_for_season_df was empty for season {season_val}. No new games to insert after delete.")
                return

            df_to_save = all_games_for_season_df.copy()
            
            bool_cols = ['is_played', 'is_national_playoffs', 'is_conf_tournament']
            for b_col in bool_cols:
                if b_col in df_to_save.columns:
                    df_to_save[b_col] = df_to_save[b_col].fillna(0).astype(int)
            
            int_cols_can_be_na = ['overtimes', 'team_score_official', 'opponent_score_official',
                                  'pred_score_team', 'pred_score_opponent', 'win', 'loss',
                                  'team_game_num_in_season']
            for i_col in int_cols_can_be_na:
                if i_col in df_to_save.columns:
                    df_to_save[i_col] = pd.to_numeric(df_to_save[i_col], errors='coerce').astype('Int64')

            int_cols_must_be_int = ['gid', 'season', 'team_tid', 'opponent_tid',
                                    'designated_home_tid', 'designated_away_tid']
            for col in int_cols_must_be_int:
                 if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce').fillna(0).astype(int)

            cursor.execute("PRAGMA table_info(GameTeamStats)")
            table_cols = [info[1] for info in cursor.fetchall()]
            df_cols_in_table_order = [col for col in table_cols if col in df_to_save.columns]
            df_to_save_filtered = df_to_save[df_cols_in_table_order]
            
            if df_to_save_filtered.empty or not df_cols_in_table_order:
                print(f"WARNING: No matching columns for GameTeamStats save or filtered DF is empty. DF cols: {df_to_save.columns.tolist()}, Table cols: {table_cols}")
                return

            df_to_save_filtered.to_sql('GameTeamStats', conn, if_exists='append', index=False, chunksize=1000)
            print(f"Saved {len(df_to_save_filtered)} game-team rows for season {season_val} to GameTeamStats.")

    except sqlite3.Error as e:
        print(f"DATABASE ERROR in save_game_team_stats_to_db for season {season_val}: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR in save_game_team_stats_to_db for season {season_val}: {ex}")
    finally:
        if conn: conn.close()

def save_processed_recruits(processed_recruits_df, class_year_being_processed):
    if processed_recruits_df.empty:
        print(f"INFO DB_OPS: Processed recruits DataFrame for class {class_year_being_processed} is empty. Performing delete for this class year only.")
        conn_del = get_db_connection()
        if conn_del:
            try:
                with conn_del: # Ensures commit/rollback
                    cursor_del = conn_del.cursor()
                    class_year_val_del = int(class_year_being_processed)
                    cursor_del.execute("DELETE FROM Recruits WHERE recruiting_class_year = ?", (class_year_val_del,))
                    print(f"INFO DB_OPS: Deleted {cursor_del.rowcount} existing recruits for class year {class_year_val_del} (as input df was empty).")
            except Exception as e_del:
                print(f"ERROR DB_OPS: during delete in save_processed_recruits (empty df case): {e_del}")
            finally:
                if conn_del: conn_del.close()
        return

    conn = get_db_connection()
    if conn is None: return
    
    try:
        with conn: # Use context manager for commit/rollback
            cursor = conn.cursor()
            df_to_save = processed_recruits_df.copy()

            # Fallback: Ensure recruiting_class_year is present and correct type
            if 'recruiting_class_year' not in df_to_save.columns:
                print(f"DEBUG DB_OPS: 'recruiting_class_year' column missing from DataFrame in save_processed_recruits. Adding it with value: {class_year_being_processed}")
                df_to_save['recruiting_class_year'] = int(class_year_being_processed)
            else:
                df_to_save['recruiting_class_year'] = pd.to_numeric(df_to_save['recruiting_class_year'], errors='coerce').fillna(int(class_year_being_processed)).astype(int)

            # Fallback: Ensure effective_season is present and correct type
            if 'effective_season' not in df_to_save.columns:
                print(f"DEBUG DB_OPS: 'effective_season' column missing. Calculating as recruiting_class_year + 1.")
                df_to_save['effective_season'] = df_to_save['recruiting_class_year'] + 1
            else:
                df_to_save['effective_season'] = pd.to_numeric(df_to_save['effective_season'], errors='coerce').fillna(df_to_save['recruiting_class_year'] + 1).astype(int)
            
            # Ensure recruit_name is not null (should be handled by analyzer, but good check)
            if 'recruit_name' in df_to_save.columns:
                if df_to_save['recruit_name'].isna().any():
                    print(f"WARNING DB_OPS: Found NaN in 'recruit_name' before saving recruits. Count: {df_to_save['recruit_name'].isna().sum()}. Filling with 'Unknown Recruit'.")
                    df_to_save['recruit_name'] = df_to_save['recruit_name'].fillna("Unknown Recruit").astype(str)
                else:
                    df_to_save['recruit_name'] = df_to_save['recruit_name'].astype(str)
            else:
                print("ERROR DB_OPS: 'recruit_name' column MISSING. Cannot save recruits.")
                return


            class_year_val = int(class_year_being_processed)
            print(f"Deleting existing recruits for class year {class_year_val} before insert...")
            cursor.execute("DELETE FROM Recruits WHERE recruiting_class_year = ?", (class_year_val,))
            print(f"Deleted {cursor.rowcount} old recruits for class year {class_year_val}.")
            
            cursor.execute("PRAGMA table_info(Recruits)")
            table_cols_info = cursor.fetchall()
            table_cols = [info[1] for info in table_cols_info]
            
            # Filter DataFrame columns to only those in the DB table and ensure essential NOT NULL columns are definitely there
            essential_db_cols = ['recruiting_class_year', 'effective_season', 'recruit_name']
            df_cols_in_table = [col for col in table_cols if col in df_to_save.columns and col != 'recruit_id']
            
            # Ensure essential columns are in the filtered list if they were in df_to_save
            for ess_col in essential_db_cols:
                if ess_col in df_to_save.columns and ess_col not in df_cols_in_table and ess_col in table_cols:
                    df_cols_in_table.append(ess_col)
            df_cols_in_table = list(dict.fromkeys(df_cols_in_table)) # Remove duplicates, preserve order

            df_to_save_filtered = df_to_save[df_cols_in_table]

            print("DEBUG DB_OPS: df_to_save_filtered before to_sql (head):")
            print(df_to_save_filtered.head())
            print("DEBUG DB_OPS: df_to_save_filtered columns:", df_to_save_filtered.columns.tolist())
            print("DEBUG DB_OPS: df_to_save_filtered dtypes:\n", df_to_save_filtered.dtypes)
            
            not_null_cols_schema = ['effective_season', 'recruiting_class_year', 'recruit_name']
            for nn_col in not_null_cols_schema:
                if nn_col in df_to_save_filtered.columns:
                    if df_to_save_filtered[nn_col].isna().any():
                        print(f"WARNING DB_OPS: Column '{nn_col}' in df_to_save_filtered contains NaNs before to_sql, count: {df_to_save_filtered[nn_col].isna().sum()}. This might violate NOT NULL if not handled by SQLite types (INTEGER can be NULL).")
                else:
                     print(f"ERROR DB_OPS: NOT NULL Column '{nn_col}' is MISSING from df_to_save_filtered.")
                     return # Stop if essential NOT NULL column is completely missing

            if not df_to_save_filtered.empty:
                df_to_save_filtered.to_sql('Recruits', conn, if_exists='append', index=False, chunksize=500)
                print(f"Saved {len(df_to_save_filtered)} processed recruits for class {class_year_val} to database.")
            else:
                print("INFO DB_OPS: df_to_save_filtered is empty. No recruits saved.")

    except sqlite3.Error as e:
        print(f"DATABASE ERROR saving processed recruits for class {class_year_being_processed}: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR saving processed recruits for class {class_year_being_processed}: {ex}")
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
        cursor.execute("DELETE FROM SeasonTeamSummaries WHERE season = ?", (season_val,)); conn.commit()
        df_to_save = season_team_summary_df.copy()
        cursor.execute("PRAGMA table_info(SeasonTeamSummaries)"); table_cols_info = cursor.fetchall()
        table_cols_db = [info[1] for info in table_cols_info]
        for tc in table_cols_db:
            if tc not in df_to_save.columns:
                is_count_like = any(kw in tc for kw in ['_w','_l','games_played','wins','losses','rank_adj_em','cid','season','team_tid','num_'])
                df_to_save[tc] = 0 if is_count_like else np.nan
        df_cols_in_table_order = [col for col in table_cols_db if col in df_to_save.columns]
        df_to_save_filtered = df_to_save[df_cols_in_table_order]
        if not df_to_save_filtered.empty:
             df_to_save_filtered.to_sql('SeasonTeamSummaries', conn, if_exists='append', index=False, chunksize=1000)
             print(f"Saved {len(df_to_save_filtered)} team summaries for season {season_val} to DB.")
    except Exception as e: print(f"ERROR saving season_summary_df: {e}")
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
    except Exception as e: print(f"GENERAL ERROR loading game stats: {e}")
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
    except Exception as e: print(f"Error loading recruits: {e}")
    finally:
        if conn: conn.close()
    return df

def load_coach_assignments_from_db():
    conn = get_db_connection();
    if conn is None: return pd.DataFrame()
    try:
        query = "SELECT ca.season, ca.team_tid, ca.coach_id, c.coach_name FROM CoachAssignments ca JOIN Coaches c ON ca.coach_id = c.coach_id"
        df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} coach assignments (with names) from database.")
        return df
    except Exception as e: print(f"Error loading coach assignments: {e}"); return pd.DataFrame()
    finally:
        if conn: conn.close()

def load_coaches_from_db():
    conn = get_db_connection();
    if conn is None: return pd.DataFrame()
    try:
        query = "SELECT coach_id, coach_name FROM Coaches"; df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} coaches from database."); return df
    except Exception as e: print(f"Error loading coaches: {e}"); return pd.DataFrame()
    finally:
        if conn: conn.close()

def save_coach_season_stats(coach_season_stats_df):
    if coach_season_stats_df.empty: print("Coach season stats DataFrame is empty, nothing to save."); return
    conn = get_db_connection();
    if conn is None: return
    try:
        with conn:
            cursor = conn.cursor()
            seasons_in_df = []
            if 'season' in coach_season_stats_df.columns: seasons_in_df = coach_season_stats_df['season'].unique()
            if len(seasons_in_df) > 0:
                for season_val in seasons_in_df:
                    cursor.execute("DELETE FROM CoachSeasonStats WHERE season = ?", (int(season_val),))
            cursor.execute("PRAGMA table_info(CoachSeasonStats)"); table_cols = [info[1] for info in cursor.fetchall()]
            df_to_save = coach_season_stats_df.copy()
            for tc in table_cols:
                if tc not in df_to_save.columns:
                    is_count_like = any(kw in tc for kw in ['_w','_l','games_coached','wins','losses','num_']) or tc in ['coach_id','season','team_tid','team_rank_adj_em']
                    df_to_save[tc] = 0 if is_count_like else np.nan
            df_to_save_filtered = df_to_save[[col for col in table_cols if col in df_to_save.columns]]
            if not df_to_save_filtered.empty:
                df_to_save_filtered.to_sql('CoachSeasonStats', conn, if_exists='append', index=False, chunksize=500)
                print(f"Saved {len(df_to_save_filtered)} coach season stats to DB.")
    except Exception as ex: print(f"GENERAL ERROR saving coach season stats: {ex}")
    finally:
        if conn: conn.close()

def save_coach_career_stats(coach_career_stats_df):
    if coach_career_stats_df.empty: print("Coach career stats DataFrame is empty, nothing to save."); return
    conn = get_db_connection();
    if conn is None: return
    try:
        with conn:
            cursor = conn.cursor()
            coach_ids_in_df = []
            if 'coach_id' in coach_career_stats_df.columns: coach_ids_in_df = coach_career_stats_df['coach_id'].unique().tolist()
            if coach_ids_in_df:
                placeholders = ','.join('?' for _ in coach_ids_in_df)
                cursor.execute(f"DELETE FROM CoachCareerStats WHERE coach_id IN ({placeholders})", coach_ids_in_df)
            cursor.execute("PRAGMA table_info(CoachCareerStats)"); table_cols = [info[1] for info in cursor.fetchall()]
            df_to_save = coach_career_stats_df.copy()
            for tc in table_cols:
                if tc not in df_to_save.columns:
                    is_count_like = any(kw in tc for kw in ['seasons_coached','teams_coached_count','total_games_coached','total_wins','total_losses','career_q','career_total_']) or tc == 'coach_id'
                    df_to_save[tc] = 0 if is_count_like else np.nan
            df_to_save_filtered = df_to_save[[col for col in table_cols if col in df_to_save.columns]]
            if not df_to_save_filtered.empty:
                df_to_save_filtered.to_sql('CoachCareerStats', conn, if_exists='append', index=False, chunksize=500)
                print(f"Saved/Updated {len(df_to_save_filtered)} coach career stats to DB.")
    except Exception as ex: print(f"GENERAL ERROR saving coach career stats: {ex}")
    finally:
        if conn: conn.close()

def save_coach_head_to_head_stats(coach_h2h_df):
    if coach_h2h_df.empty:
        print("Coach H2H stats DataFrame is empty, nothing to save.")
        return
    conn = get_db_connection()
    if conn is None: return
    seasons_in_df = []
    if 'season' in coach_h2h_df.columns:
        seasons_in_df = coach_h2h_df['season'].unique().tolist()
    try:
        with conn:
            cursor = conn.cursor()
            if seasons_in_df:
                for season_val in seasons_in_df:
                    cursor.execute("DELETE FROM CoachHeadToHeadStats WHERE season = ?", (int(season_val),))
            
            cursor.execute("PRAGMA table_info(CoachHeadToHeadStats)")
            table_cols = [info[1] for info in cursor.fetchall()]
            df_to_save = coach_h2h_df.copy()
            for tc in table_cols:
                if tc not in df_to_save.columns:
                    is_count_like = any(kw in tc for kw in ['wins', 'games_played']) or '_id' in tc or tc == 'season'
                    df_to_save[tc] = 0 if is_count_like else np.nan
            
            df_cols_in_table_order = [col for col in table_cols if col in df_to_save.columns]
            df_to_save_filtered = df_to_save[df_cols_in_table_order] # Ensure correct order and only existing cols
            
            if not df_to_save_filtered.empty:
                df_to_save_filtered.to_sql('CoachHeadToHeadStats', conn, if_exists='append', index=False, chunksize=500)
                print(f"Saved {len(df_to_save_filtered)} coach H2H stats to the database for seasons: {seasons_in_df}.")
            else:
                print(f"No columns in coach_h2h_df matched CoachHeadToHeadStats table after filtering. Nothing saved. DF cols: {df_to_save.columns.tolist()}, Table cols: {table_cols}")

    except sqlite3.Error as e:
        print(f"DATABASE ERROR saving coach H2H stats: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR saving coach H2H stats: {ex}")
    finally:
        if conn: conn.close()

def load_season_summary_for_display(season_to_load=None):
    """
    Loads the SeasonTeamSummaries table for display.
    If season_to_load is None, it tries to load the most recent season.
    Sorts by adj_em by default.
    Adds formatted Q1-Q4 record strings, rank, AND coach_name.
    Returns a tuple: (DataFrame, actual_season_loaded)
    """
    conn = get_db_connection()
    if conn is None:
        print("ERROR DB_OPS: No connection for load_season_summary_for_display.")
        return pd.DataFrame(), None

    df_summary = pd.DataFrame()
    actual_season_loaded = season_to_load
    
    try:
        if actual_season_loaded is None:
            s_df = pd.read_sql_query("SELECT DISTINCT season FROM SeasonTeamSummaries ORDER BY season DESC LIMIT 1", conn)
            if not s_df.empty:
                actual_season_loaded = int(s_df['season'].iloc[0])
            else:
                print("INFO DB_OPS: No seasons found in SeasonTeamSummaries to display.")
                return pd.DataFrame(), None
        
        if actual_season_loaded is None:
             print("INFO DB_OPS: Could not determine a season to load for display.")
             return pd.DataFrame(), None

        actual_season_loaded = int(actual_season_loaded)
        
        # Query to get season summaries and join with coach info
        # Using LEFT JOINs so teams without coaches are still included
        query = f"""
        SELECT ss.*, c.coach_name,ca.coach_id
        FROM SeasonTeamSummaries ss
        LEFT JOIN CoachAssignments ca ON ss.season = ca.season AND ss.team_tid = ca.team_tid
        LEFT JOIN Coaches c ON ca.coach_id = c.coach_id
        WHERE ss.season = ?
        ORDER BY ss.adj_em DESC
        """
        df_summary = pd.read_sql_query(query, conn, params=(actual_season_loaded,))
        
        if not df_summary.empty:
            print(f"Loaded {len(df_summary)} team summaries (with coach names) for season {actual_season_loaded} for display.")
            for i in range(1, 5): # Q-Record Formatting
                q_w_col, q_l_col, q_rec_col = f'q{i}_w', f'q{i}_l', f'Q{i}_Record'
                if q_w_col in df_summary.columns and q_l_col in df_summary.columns:
                    df_summary[q_w_col] = pd.to_numeric(df_summary[q_w_col], errors='coerce').fillna(0).astype(int)
                    df_summary[q_l_col] = pd.to_numeric(df_summary[q_l_col], errors='coerce').fillna(0).astype(int)
                    df_summary[q_rec_col] = df_summary[q_w_col].astype(str) + "-" + df_summary[q_l_col].astype(str)
                else: df_summary[q_rec_col] = "0-0"
            
            if 'adj_em' in df_summary.columns: # Rank Calculation
                df_summary['adj_em'] = pd.to_numeric(df_summary['adj_em'], errors='coerce')
                if df_summary['adj_em'].notna().any():
                    df_summary['rank'] = df_summary['adj_em'].rank(method='min', ascending=False).fillna(0).astype(int)
                else: df_summary['rank'] = 0
            else: df_summary['rank'] = 0
            
            if 'coach_name' not in df_summary.columns: # Ensure column exists even if all are NULL from join
                df_summary['coach_name'] = None
            df_summary['coach_name'] = df_summary['coach_name'].fillna("N/A")

        else:
            print(f"INFO DB_OPS: No summary data found for season {actual_season_loaded}.")
            
    except Exception as e:
        print(f"ERROR DB_OPS: Error loading season summary for display: {e}")
        df_summary = pd.DataFrame()
        actual_season_loaded = target_season # Revert to initially targeted season on error for label
        if actual_season_loaded is None and conn:
            try:
                s_df_fallback = pd.read_sql_query("SELECT DISTINCT season FROM SeasonTeamSummaries ORDER BY season DESC LIMIT 1", conn)
                if not s_df_fallback.empty: actual_season_loaded = int(s_df_fallback['season'].iloc[0])
            except: pass # Ignore error in fallback during error handling
    finally:
        if conn: conn.close()
    return df_summary, actual_season_loaded
def save_postseason_results(processed_postseason_df, season_to_clear):
    """Saves processed postseason results. Deletes for the given season then appends."""
    if processed_postseason_df.empty:
        print(f"Processed postseason DataFrame for season {season_to_clear} is empty. Only performing delete.")
    if season_to_clear is None:
        print("ERROR: season_to_clear is None. Cannot save postseason results.")
        return

    conn = get_db_connection()
    if conn is None: return
    
    try:
        with conn:
            cursor = conn.cursor()
            season_val = int(season_to_clear)
            
            print(f"Deleting existing postseason results for season {season_val}...")
            cursor.execute("DELETE FROM PostseasonResults WHERE season = ?", (season_val,))
            print(f"Deleted {cursor.rowcount} old postseason results for season {season_val}.")

            if not processed_postseason_df.empty:
                df_to_save = processed_postseason_df.copy()
                
                # Ensure DataFrame columns match table schema before saving
                cursor.execute("PRAGMA table_info(PostseasonResults)")
                table_cols = [info[1] for info in cursor.fetchall()]
                # 'postseason_id' is autoincrement, so don't include it if df doesn't have it
                df_cols_to_save = [col for col in table_cols if col in df_to_save.columns and col != 'postseason_id']
                df_to_save_filtered = df_to_save[df_cols_to_save]

                if not df_to_save_filtered.empty:
                    df_to_save_filtered.to_sql('PostseasonResults', conn, if_exists='append', index=False, chunksize=100)
                    print(f"Saved {len(df_to_save_filtered)} postseason results for season {season_val} to DB.")
                else:
                    print("WARNING: Postseason DataFrame became empty after filtering for DB columns.")
            else: # If input df was empty, we only performed the delete
                 pass


    except sqlite3.Error as e:
        print(f"DATABASE ERROR saving postseason results for season {season_to_clear}: {e}")
    except Exception as ex:
        print(f"GENERAL ERROR saving postseason results for season {season_to_clear}: {ex}")
    finally:
        if conn: conn.close()
def load_postseason_results_for_season(season_to_load):
    """Loads postseason results for a specific season."""
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()
    df = pd.DataFrame()
    try:
        season_val = int(season_to_load)
        # --- CORRECTED QUERY ---
        query = "SELECT season, team_tid, event_type, seed, result FROM PostseasonResults WHERE season = ?"
        df = pd.read_sql_query(query, conn, params=(season_val,))
        if not df.empty:
            print(f"Loaded {len(df)} postseason result rows for season {season_val} (with season column).")
        else:
            print(f"No postseason results found for season {season_val}.")
    except Exception as e:
        print(f"Error loading postseason results for season {season_to_load} from database: {e}")
    finally:
        if conn: conn.close()
    return df
def load_team_season_details(team_tid_to_load, season_to_load):
    conn = get_db_connection()
    if conn is None:
        print("ERROR DB_OPS: No connection for load_team_season_details.")
        return None, None

    target_season_for_return = None
    team_details_dict = None
    try:
        season_val = int(season_to_load)
        tid_val = int(team_tid_to_load)
        target_season_for_return = season_val

        query = f"""
        SELECT 
            ss.*, 
            t.name AS team_table_name,
            t.region AS team_table_region,
            t.full_name AS team_table_full_name,
            c.coach_name, 
            ca.coach_id
        FROM SeasonTeamSummaries ss
        JOIN Teams t ON ss.team_tid = t.team_tid
        LEFT JOIN CoachAssignments ca ON ss.season = ca.season AND ss.team_tid = ca.team_tid
        LEFT JOIN Coaches c ON ca.coach_id = c.coach_id
        WHERE ss.season = ? AND ss.team_tid = ?
        """
        details_df = pd.read_sql_query(query, conn, params=(season_val, tid_val))
        
        if not details_df.empty:
            print(f"INFO DB_OPS: Found data for team {tid_val}, season {season_val}. Processing...")
            
            team_detail_series = details_df.iloc[0].copy()
            
            # Convert Series to dict first, then work with dict values
            team_details_dict = team_detail_series.to_dict()

            # Ensure desired name fields are present
            team_details_dict['full_name'] = team_details_dict.get('team_table_full_name', team_details_dict.get('team_abbrev', 'Unknown Team'))
            team_details_dict['name'] = team_details_dict.get('team_table_name', team_details_dict.get('team_abbrev', 'Unknown Team'))
            team_details_dict['region'] = team_details_dict.get('team_table_region', '')

            # Coach name handling
            coach_name_val = team_details_dict.get('coach_name')
            team_details_dict['coach_name'] = "N/A" if pd.isna(coach_name_val) else str(coach_name_val)
            
            # Coach ID handling
            coach_id_val = team_details_dict.get('coach_id')
            if pd.isna(coach_id_val):
                team_details_dict['coach_id'] = None
            else:
                try: team_details_dict['coach_id'] = int(coach_id_val)
                except (ValueError, TypeError): team_details_dict['coach_id'] = None

            # Format Quadrant Records
            for i in range(1, 5):
                q_w_col, q_l_col, q_rec_col = f'q{i}_w', f'q{i}_l', f'Q{i}_Record'
                
                w_val = team_details_dict.get(q_w_col)
                l_val = team_details_dict.get(q_l_col)
                
                # Convert to numeric, default to 0 if NaN or conversion error
                w = 0
                if pd.notna(w_val):
                    try: w = int(pd.to_numeric(w_val))
                    except (ValueError, TypeError): w = 0 # default if not convertible
                
                l = 0
                if pd.notna(l_val):
                    try: l = int(pd.to_numeric(l_val))
                    except (ValueError, TypeError): l = 0 # default if not convertible
                
                team_details_dict[q_rec_col] = f"{w}-{l}"
            
            # Rank
            rank_val = team_details_dict.get('rank_adj_em')
            if pd.notna(rank_val):
                try: team_details_dict['rank_adj_em'] = int(rank_val)
                except (ValueError, TypeError): team_details_dict['rank_adj_em'] = 0
            else: # If rank_val is None/NaN or key doesn't exist
                team_details_dict['rank_adj_em'] = 0
            
            print(f"INFO DB_OPS: Successfully processed details for team {tid_val}, season {season_val}.")
            return team_details_dict, target_season_for_return
        else:
            print(f"INFO DB_OPS: No summary data found for team {tid_val}, season {season_val} in DB query.")
            return None, target_season_for_return
            
    except Exception as e:
        print(f"ERROR DB_OPS: Error in load_team_season_details for T:{team_tid_to_load} S:{season_to_load}: {e}")
        final_season_label = season_to_load if season_to_load is not None else "Unknown"
        return None, final_season_label
    finally:
        if conn: conn.close()

    return None, season_to_load if season_to_load is not None else None
def load_team_game_log(team_tid_to_load, season_to_load):
    """
    Loads the game log for a specific team and season.
    Includes actual results for played games and predictions for future games.
    """
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()

    game_log_df = pd.DataFrame()
    try:
        season_val = int(season_to_load)
        tid_val = int(team_tid_to_load)
        
        # Query GameTeamStats
        # Ensure all necessary columns including prediction columns are selected
        query = f"""
        SELECT gid, season, team_tid, opponent_tid, team_abbrev, opponent_abbrev, 
               location, overtimes, is_played, game_date, game_time, 
               is_national_playoffs, is_conf_tournament, team_game_num_in_season,
               team_score_official, opponent_score_official, win, loss, game_quadrant,
               pred_win_prob_team, pred_margin_team, pred_score_team, pred_score_opponent
        FROM GameTeamStats
        WHERE season = ? AND team_tid = ?
        ORDER BY team_game_num_in_season ASC 
        """
        # Sorting by team_game_num_in_season will give chronological order
        # Or sort by game_date if it's reliably populated and sortable
        
        game_log_df = pd.read_sql_query(query, conn, params=(season_val, tid_val))
        
        if not game_log_df.empty:
            print(f"Loaded {len(game_log_df)} games for team {tid_val}, season {season_val}.")
            # Ensure boolean columns are treated as boolean for template logic if needed
            for bool_col in ['is_played', 'is_national_playoffs', 'is_conf_tournament']:
                if bool_col in game_log_df.columns:
                    game_log_df[bool_col] = game_log_df[bool_col].astype(bool)
        else:
            print(f"No game log data found for team {tid_val}, season {season_val}.")
            
    except Exception as e:
        print(f"Error loading game log for team {team_tid_to_load}, season {season_to_load}: {e}")
    finally:
        if conn: conn.close()
    return game_log_df


# --- Function to load historical summary for a single team ---
def load_team_historical_summary(team_tid_to_load):
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()

    history_df = pd.DataFrame()
    try:
        tid_val = int(team_tid_to_load)
        query = f"""
        SELECT 
            ss.season, ss.wins, ss.losses, ss.win_pct, 
            ss.adj_em, ss.rank_adj_em, 
            ss.nt_result, ss.nit_result, ss.nt_seed, ss.nit_seed,
            c.coach_name, ca.coach_id 
        FROM SeasonTeamSummaries ss
        LEFT JOIN CoachAssignments ca ON ss.season = ca.season AND ss.team_tid = ca.team_tid
        LEFT JOIN Coaches c ON ca.coach_id = c.coach_id
        WHERE ss.team_tid = ?
        ORDER BY ss.season DESC
        """
        history_df = pd.read_sql_query(query, conn, params=(tid_val,))
        
        if not history_df.empty:
            print(f"Loaded {len(history_df)} historical seasons for team {tid_val}.")
            
            # Fill NaNs for display consistency
            for col in ['nt_result', 'nit_result', 'coach_name']:
                if col not in history_df.columns: history_df[col] = "N/A" # Add column if missing
                history_df[col] = history_df[col].fillna("N/A")
            
            for col in ['nt_seed', 'nit_seed', 'coach_id']: # Ensure these are Int64 to handle potential pd.NA
                 if col not in history_df.columns: history_df[col] = pd.NA
                 history_df[col] = pd.to_numeric(history_df[col], errors='coerce').astype('Int64')
        else:
            print(f"No historical summary data found for team {tid_val}.")
            
    except Exception as e:
        print(f"Error loading historical summary for team {team_tid_to_load}: {e}")
    finally:
        if conn: conn.close()
    return history_df
def load_team_historical_summary(team_tid_to_load):
    """
    Loads key historical season summaries for a specific team, including coach for each season.
    """
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()

    history_df = pd.DataFrame()
    try:
        tid_val = int(team_tid_to_load)
        
        # Query to get season summaries and join with coach info
        query = f"""
        SELECT 
            ss.season, ss.wins, ss.losses, ss.win_pct, 
            ss.adj_em, ss.rank_adj_em, 
            ss.nt_result, ss.nit_result, ss.nt_seed, ss.nit_seed,
            c.coach_name
        FROM SeasonTeamSummaries ss
        LEFT JOIN CoachAssignments ca ON ss.season = ca.season AND ss.team_tid = ca.team_tid
        LEFT JOIN Coaches c ON ca.coach_id = c.coach_id
        WHERE ss.team_tid = ?
        ORDER BY ss.season DESC
        """
        history_df = pd.read_sql_query(query, conn, params=(tid_val,))
        
        if not history_df.empty:
            print(f"Loaded {len(history_df)} historical seasons for team {tid_val}.")
            # Fill NaNs for display consistency
            for col in ['nt_result', 'nit_result', 'coach_name']: # Added coach_name
                if col in history_df.columns:
                    history_df[col] = history_df[col].fillna("N/A")
            for col in ['nt_seed', 'nit_seed']: # Seeds to 0 or keep as nullable int
                 if col in history_df.columns:
                     history_df[col] = pd.to_numeric(history_df[col], errors='coerce').astype('Int64').fillna(pd.NA) # Use pd.NA for nullable int

        else:
            print(f"No historical summary data found for team {tid_val}.")
            
    except Exception as e:
        print(f"Error loading historical summary for team {team_tid_to_load}: {e}")
    finally:
        if conn: conn.close()
    return history_df
def load_coach_info(coach_id_to_load):
    """Loads basic information for a specific coach."""
    conn = get_db_connection()
    if conn is None: return None
    coach_info = None
    try:
        cid_val = int(coach_id_to_load)
        query = "SELECT coach_id, coach_name FROM Coaches WHERE coach_id = ?"
        coach_df = pd.read_sql_query(query, conn, params=(cid_val,))
        if not coach_df.empty:
            coach_info = coach_df.iloc[0].to_dict()
            print(f"Loaded info for coach_id {cid_val}: {coach_info.get('coach_name')}")
        else:
            print(f"No coach found with coach_id {cid_val}")
    except Exception as e:
        print(f"Error loading info for coach_id {coach_id_to_load}: {e}")
    finally:
        if conn: conn.close()
    return coach_info

def load_coach_career_stats_by_id(coach_id_to_load):
    """Loads career stats for a specific coach."""
    conn = get_db_connection()
    if conn is None: return None
    career_stats = None
    try:
        cid_val = int(coach_id_to_load)
        query = "SELECT * FROM CoachCareerStats WHERE coach_id = ?"
        career_df = pd.read_sql_query(query, conn, params=(cid_val,))
        if not career_df.empty:
            career_stats = career_df.iloc[0].to_dict()
            # Convert potential float NaNs for integer counts to 0 for display
            count_cols = [col for col in career_df.columns if 'total_' in col or 'career_q' in col or '_appearances' in col or '_championships' in col or 'seasons_coached' in col or 'teams_coached_count' in col]
            for col in count_cols:
                if col in career_stats and pd.isna(career_stats[col]):
                    career_stats[col] = 0
                elif col in career_stats: # Ensure integer type if not NaN
                    career_stats[col] = int(career_stats[col])

            print(f"Loaded career stats for coach_id {cid_val}")
        else:
            print(f"No career stats found for coach_id {cid_val}")
    except Exception as e:
        print(f"Error loading career stats for coach_id {coach_id_to_load}: {e}")
    finally:
        if conn: conn.close()
    return career_stats

def load_coach_seasons_stats_by_id(coach_id_to_load):
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()

    seasons_df = pd.DataFrame()
    try:
        cid_val = int(coach_id_to_load)
        
        # Select specific columns from SeasonTeamSummaries to avoid duplicates after JOIN
        # and alias them if their base names might already exist in CoachSeasonStats (though they shouldn't for these specific stats)
        query = f"""
        SELECT 
            css.*, 
            sts.avg_opp_adj_em AS team_avg_opp_adj_em, /* Alias to avoid potential conflict if css had it */
            sts.luck_adj AS team_luck_adj,
            sts.wab AS team_wab,
            sts.num_cpr AS team_num_cpr,
            sts.nt_seed, sts.nt_result, 
            sts.nit_seed, sts.nit_result
        FROM CoachSeasonStats css
        LEFT JOIN SeasonTeamSummaries sts 
            ON css.season = sts.season AND css.team_tid = sts.team_tid
        WHERE css.coach_id = ? 
        ORDER BY css.season DESC
        """
        seasons_df = pd.read_sql_query(query, conn, params=(cid_val,))
        
        if not seasons_df.empty:
            print(f"Loaded {len(seasons_df)} seasonal records (joined) for coach_id {cid_val}")
            
            # Columns from CoachSeasonStats for Q-Record formatting
            for i in range(1, 5):
                q_w_col, q_l_col, q_rec_col = f'q{i}_w', f'q{i}_l', f'Q{i}_Record_CSS'
                if q_w_col in seasons_df.columns and q_l_col in seasons_df.columns:
                    seasons_df[q_w_col] = pd.to_numeric(seasons_df[q_w_col], errors='coerce').fillna(0).astype(int)
                    seasons_df[q_l_col] = pd.to_numeric(seasons_df[q_l_col], errors='coerce').fillna(0).astype(int)
                    seasons_df[q_rec_col] = seasons_df[q_w_col].astype(str) + "-" + seasons_df[q_l_col].astype(str)
                else:
                    seasons_df[q_rec_col] = "0-0"

            # Fill NaNs for the joined columns (now aliased if necessary)
            # These are team-level stats for the season the coach was there
            team_level_text_cols = ['nt_result', 'nit_result']
            team_level_nullable_int_cols = ['nt_seed', 'nit_seed']
            team_level_float_cols = ['team_avg_opp_adj_em', 'team_luck_adj', 'team_wab']
            team_level_count_cols = ['team_num_cpr']


            for col in team_level_text_cols:
                if col in seasons_df.columns: seasons_df[col] = seasons_df[col].fillna("N/A")
                else: seasons_df[col] = "N/A"
            for col in team_level_nullable_int_cols:
                if col in seasons_df.columns: seasons_df[col] = pd.to_numeric(seasons_df[col], errors='coerce').astype('Int64')
                else: seasons_df[col] = pd.NA # Use pd.NA for missing Int64
            for col in team_level_float_cols:
                if col in seasons_df.columns: seasons_df[col] = pd.to_numeric(seasons_df[col], errors='coerce') # Keep as float, allow NaN
                else: seasons_df[col] = np.nan
            for col in team_level_count_cols:
                if col in seasons_df.columns: seasons_df[col] = pd.to_numeric(seasons_df[col], errors='coerce').fillna(0).astype(int)
                else: seasons_df[col] = 0
        else:
            print(f"No seasonal stats found for coach_id {cid_val}")
            
    except sqlite3.Error as sql_e:
        print(f"DATABASE SQL ERROR in load_coach_seasons_stats_by_id for coach_id {coach_id_to_load}: {sql_e}")
    except Exception as e:
        print(f"GENERAL ERROR loading seasonal stats for coach_id {coach_id_to_load}: {e}")
    finally:
        if conn: conn.close()
    return seasons_df
def load_all_coaches_with_enhanced_career_stats():
    """
    Loads all coaches with their existing career stats from CoachCareerStats,
    and then calculates additional career postseason metrics and total GTs
    by processing their associated seasonal stats from CoachSeasonStats (joined with SeasonTeamSummaries).
    """
    conn = get_db_connection()
    if conn is None:
        print("ERROR DB_OPS: No connection for load_all_coaches_with_enhanced_career_stats.")
        return []

    coaches_career_list = []
    try:
        # Step 1: Get all coaches and their existing career stats
        # Using LEFT JOIN to ensure all coaches from Coaches table are included,
        # even if they don't have an entry in CoachCareerStats yet.
        query_career = """
        SELECT 
            c.coach_id, 
            c.coach_name, 
            ccs.seasons_coached,
            ccs.teams_coached_count,
            ccs.total_games_coached,
            ccs.total_wins,
            ccs.total_losses,
            ccs.career_win_pct,
            ccs.career_q1_w, ccs.career_q1_l, ccs.career_q2_w, ccs.career_q2_l,
            ccs.career_q3_w, ccs.career_q3_l, ccs.career_q4_w, ccs.career_q4_l,
            ccs.career_avg_team_adj_em,
            ccs.career_avg_team_rank,
            ccs.career_total_recruits,
            ccs.career_avg_recruit_ovr_of_classes,
            ccs.career_total_5_stars, ccs.career_total_4_stars, ccs.career_total_3_stars, /* Added 3 stars */
            ccs.career_avg_score_ktv
            /* Add other existing fields from CoachCareerStats as needed */
        FROM Coaches c
        LEFT JOIN CoachCareerStats ccs ON c.coach_id = ccs.coach_id
        ORDER BY c.coach_name;
        """
        all_coaches_career_df = pd.read_sql_query(query_career, conn)
        
        if all_coaches_career_df.empty:
            print("INFO DB_OPS: No coaches found in Coaches table.")
            return []

        # Step 2: Get all relevant seasonal data for all coaches in one go
        # This seasonal data should include team's postseason results and num_gt
        query_seasonal = """
        SELECT 
            css.coach_id, 
            css.num_gt,        /* From CoachSeasonStats */
            sts.nt_result,     /* From SeasonTeamSummaries */
            sts.nit_result     /* From SeasonTeamSummaries */
        FROM CoachSeasonStats css
        LEFT JOIN SeasonTeamSummaries sts 
            ON css.season = sts.season AND css.team_tid = sts.team_tid
        """
        all_seasonal_stats_df = pd.read_sql_query(query_seasonal, conn)

        if all_seasonal_stats_df.empty:
            print("WARNING DB_OPS: No seasonal stats found in CoachSeasonStats to calculate detailed career postseason/GT metrics.")
        
        coaches_career_list = all_coaches_career_df.to_dict(orient='records')

        for coach_dict in coaches_career_list:
            coach_id = coach_dict.get('coach_id')
            if coach_id is None: continue # Should not happen if joining from Coaches table

            # Initialize new calculated career counters
            coach_dict['career_nt_appearances_calc'] = 0
            coach_dict['career_nit_appearances_calc'] = 0
            coach_dict['career_nt_champs_calc'] = 0
            coach_dict['career_final_fours_calc'] = 0
            coach_dict['career_championship_games_calc'] = 0 # NT Championship Games
            coach_dict['career_total_gt_calc'] = 0

            if not all_seasonal_stats_df.empty:
                coach_seasons_data = all_seasonal_stats_df[all_seasonal_stats_df['coach_id'] == coach_id]
                
                if not coach_seasons_data.empty:
                    # Calculate total grad transfers
                    coach_dict['career_total_gt_calc'] = int(pd.to_numeric(coach_seasons_data['num_gt'], errors='coerce').fillna(0).sum())
                    
                    # Calculate postseason achievements
                    for _, season_row in coach_seasons_data.iterrows():
                        nt_res = season_row.get('nt_result')
                        nit_res = season_row.get('nit_result')

                        if nt_res and nt_res != 'N/A' and nt_res != 0: # Check for 0 if it was a fillna value
                            coach_dict['career_nt_appearances_calc'] += 1
                            if nt_res == 'Champion':
                                coach_dict['career_nt_champs_calc'] += 1
                            if nt_res in ['Champion', 'Championship Game']: # Assuming 'Championship Game' means runner-up or made it there
                                coach_dict['career_championship_games_calc'] += 1
                            if nt_res in ['Champion', 'Championship Game', 'Final Four']:
                                coach_dict['career_final_fours_calc'] += 1
                            # Elite 8s can be added similarly if result strings match
                        
                        if nit_res and nit_res != 'N/A' and nit_res != 0:
                            coach_dict['career_nit_appearances_calc'] += 1
                            # Add NIT Champion count if needed, same way as NT champs
            
            # Ensure existing career stats (that might be NaN from LEFT JOIN) have defaults
            # These are from the CoachCareerStats table itself
            existing_career_counts = [
                'seasons_coached', 'teams_coached_count', 'total_games_coached',
                'total_wins', 'total_losses',
                'career_q1_w', 'career_q1_l', 'career_q2_w', 'career_q2_l',
                'career_q3_w', 'career_q3_l', 'career_q4_w', 'career_q4_l',
                'career_total_recruits', 'career_total_5_stars', 'career_total_4_stars', 'career_total_3_stars'
            ]
            existing_career_floats = [
                'career_win_pct', 'career_avg_team_adj_em', 'career_avg_team_rank',
                'career_avg_recruit_ovr_of_classes', 'career_avg_score_ktv'
            ]

            for col in existing_career_counts:
                if pd.isna(coach_dict.get(col)):
                    coach_dict[col] = 0
            for col in existing_career_floats:
                if pd.isna(coach_dict.get(col)):
                    coach_dict[col] = 0.0 # Or np.nan if you prefer for averages

        print(f"INFO DB_OPS: Loaded and enhanced career stats for {len(coaches_career_list)} coaches.")

    except sqlite3.Error as sql_e:
        print(f"DATABASE SQL ERROR in load_all_coaches_with_enhanced_career_stats: {sql_e}")
    except Exception as e:
        print(f"GENERAL ERROR in load_all_coaches_with_enhanced_career_stats: {e}")
    finally:
        if conn: conn.close()
    
    return coaches_career_list
def load_recruiting_rankings_for_class_year(class_year_to_load, default_sort_col='score_ktv', sort_ascending=False):
    """
    Loads team summaries for the effective playing season corresponding to a 
    recruiting class year, focusing on recruiting metrics and including coach info.
    Returns a DataFrame sorted by the specified recruiting score and the effective_season loaded.
    """
    conn = get_db_connection()
    if conn is None:
        print("ERROR DB_OPS: No connection for load_recruiting_rankings_for_class_year.")
        return pd.DataFrame(), None

    recruiting_rankings_df = pd.DataFrame()
    effective_season_loaded = None

    if class_year_to_load is None:
        print("INFO DB_OPS: No class_year_to_load specified for recruiting rankings.")
        # Optionally, determine the latest possible class year based on max(effective_season) - 1
        # For now, returning empty if no specific class year.
        return pd.DataFrame(), None
        
    try:
        class_year = int(class_year_to_load)
        effective_season_loaded = class_year + 1 # Recruits from class X play in season X+1

        # Select all relevant columns from SeasonTeamSummaries and join for coach name
        # Ensure all recruiting columns (num_X_star, num_gt, num_juco, num_cpr, score_X) are in SeasonTeamSummaries
        query = f"""
        SELECT 
            ss.season, ss.team_tid, ss.team_abbrev, ss.cid,
            c.coach_name, ca.coach_id, /* Coach info for that effective season */
            ss.num_recruits, ss.avg_recruit_ovr, 
            ss.num_5_star, ss.num_4_star, ss.num_3_star, ss.num_2_star, ss.num_1_star,
            ss.num_gt, ss.num_juco, ss.num_cpr,
            ss.score_ktv, ss.score_onz, ss.score_nspn, ss.score_storms, ss.score_248sports
            /* Add any other columns from SeasonTeamSummaries you might want on the recruiting rank page */
        FROM SeasonTeamSummaries ss
        LEFT JOIN CoachAssignments ca ON ss.season = ca.season AND ss.team_tid = ca.team_tid
        LEFT JOIN Coaches c ON ca.coach_id = c.coach_id
        WHERE ss.season = ? 
        """
        
        # Sorting logic
        if default_sort_col and isinstance(default_sort_col, str):
            # Validate sort_col against typical SeasonTeamSummaries columns to prevent SQL injection
            # For simplicity here, assuming default_sort_col is safe.
            # In a production system, validate it against a list of allowed sort columns.
            query += f" ORDER BY ss.{default_sort_col} {'ASC' if sort_ascending else 'DESC'}"
        else: # Default sort if col is invalid or None
            query += " ORDER BY ss.score_ktv DESC"


        recruiting_rankings_df = pd.read_sql_query(query, conn, params=(effective_season_loaded,))
        
        if not recruiting_rankings_df.empty:
            print(f"INFO DB_OPS: Loaded {len(recruiting_rankings_df)} team recruiting summaries for class year {class_year} (effective season {effective_season_loaded}).")
            
            # Add a rank column based on the sort
            # If default_sort_col was valid and used for sorting
            if default_sort_col in recruiting_rankings_df.columns:
                recruiting_rankings_df['rank'] = recruiting_rankings_df[default_sort_col].rank(method='min', ascending=sort_ascending).astype(int)
            else: # Fallback rank if sort column wasn't valid (should not happen if validated)
                recruiting_rankings_df['rank'] = range(1, len(recruiting_rankings_df) + 1)

            # Ensure coach_name exists and fill NaNs
            if 'coach_name' not in recruiting_rankings_df.columns:
                recruiting_rankings_df['coach_name'] = "N/A"
            else:
                recruiting_rankings_df['coach_name'] = recruiting_rankings_df['coach_name'].fillna("N/A")
            
            if 'coach_id' not in recruiting_rankings_df.columns:
                recruiting_rankings_df['coach_id'] = None
            else:
                recruiting_rankings_df['coach_id'] = pd.to_numeric(recruiting_rankings_df['coach_id'], errors='coerce').astype('Int64')

        else:
            print(f"INFO DB_OPS: No recruiting summary data found for class year {class_year} (effective season {effective_season_loaded}).")
            
    except sqlite3.Error as sql_e:
        print(f"DATABASE SQL ERROR in load_recruiting_rankings_for_class_year (class {class_year_to_load}): {sql_e}")
        effective_season_loaded = None # Indicate failure
    except Exception as e:
        print(f"GENERAL ERROR in load_recruiting_rankings_for_class_year (class {class_year_to_load}): {e}")
        effective_season_loaded = None # Indicate failure
    finally:
        if conn: conn.close()
        
    return recruiting_rankings_df, effective_season_loaded
