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
    conn = get_db_connection();
    if conn is None: return pd.DataFrame()
    df = pd.DataFrame();
    try:
        target_season = season_to_load
        if target_season is None:
            s_df = pd.read_sql_query("SELECT DISTINCT season FROM SeasonTeamSummaries ORDER BY season DESC LIMIT 1", conn)
            if not s_df.empty: target_season = int(s_df['season'].iloc[0])
            else: print("No seasons found in SeasonTeamSummaries to display."); return pd.DataFrame()
        if target_season is None: print("Could not determine a season to load for display."); return pd.DataFrame()
        target_season = int(target_season)
        query = "SELECT * FROM SeasonTeamSummaries WHERE season = ? ORDER BY adj_em DESC"
        df = pd.read_sql_query(query, conn, params=(target_season,))
        if not df.empty:
            print(f"Loaded {len(df)} team summaries for season {target_season} for display.")
            for i in range(1, 5):
                q_w,q_l,q_rec = f'q{i}_w',f'q{i}_l',f'Q{i}_Record'
                if q_w in df.columns and q_l in df.columns: df[q_rec] = df[q_w].astype(str) + "-" + df[q_l].astype(str)
                else: df[q_rec] = "0-0"
            if 'adj_em' in df.columns: df['rank'] = df['adj_em'].rank(method='min', ascending=False).astype(int)
            else: df['rank'] = 0
        else: print(f"No summary data found for season {target_season}.")
    except Exception as e: print(f"Error loading season summary for display: {e}")
    finally:
        if conn: conn.close()
    return df
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
