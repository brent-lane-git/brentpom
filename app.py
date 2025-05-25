# app.py
from flask import Flask, render_template, request, redirect, url_for, abort
import database_ops
import config
import os
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)

# --- Load Conference Lookup Map on App Startup ---
cid_to_conf_abbrev_map = {}
try:
    if hasattr(config, 'DATA_DIR'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir_path = config.DATA_DIR
        if not os.path.isabs(data_dir_path):
            data_dir_path = os.path.join(base_dir, data_dir_path)
        
        conference_mapping_file_path = os.path.join(data_dir_path, 'conference_mapping.json')
        if os.path.exists(conference_mapping_file_path):
            with open(conference_mapping_file_path, 'r') as f:
                str_cid_map = json.load(f)
                cid_to_conf_abbrev_map = {int(k): v for k, v in str_cid_map.items()}
            print(f"INFO APP: Conference mapping loaded successfully from {conference_mapping_file_path}")
        else:
            print(f"WARNING APP: Conference mapping file not found at '{conference_mapping_file_path}'. CIDs will be displayed as numbers.")
    else:
        print("WARNING APP: config.DATA_DIR not defined. Cannot load conference mapping.")
except Exception as e:
    print(f"ERROR APP: Could not load or parse conference_mapping.json: {e}")
# --- End Conference Lookup ---

# --- Load Team Name (Region) Lookup Map on App Startup ---
tid_to_school_name_map = {}
conn_teams_init = None # Initialize connection variable outside try
try:
    conn_teams_init = database_ops.get_db_connection()
    if conn_teams_init:
        teams_for_map_df = pd.read_sql_query("SELECT team_tid, region FROM Teams", conn_teams_init)
        if not teams_for_map_df.empty:
            tid_to_school_name_map = pd.Series(
                teams_for_map_df.region.values,
                index=teams_for_map_df.team_tid
            ).to_dict()
            print(f"INFO APP: Team name (region) lookup loaded successfully with {len(tid_to_school_name_map)} entries.")
        else:
            print("WARNING APP: No teams found in database to create team name lookup.")
    else:
        print("ERROR APP: No DB connection to initially load team names for lookup.")
except Exception as e:
    print(f"ERROR APP: Could not load team names for lookup during app startup: {e}")
finally:
    if conn_teams_init:
        conn_teams_init.close()
# --- END Team Name Lookup ---


def get_common_template_context(active_tab_name, specific_season_or_year=None, is_class_year=False):
    conn_temp = database_ops.get_db_connection()
    available_years = []
    latest_year = None
    
    if conn_temp:
        try:
            if is_class_year:
                query = "SELECT DISTINCT effective_season FROM Recruits ORDER BY effective_season DESC"
                try:
                    years_df = pd.read_sql_query(query, conn_temp)
                    if not years_df.empty:
                        valid_effective_seasons = [s for s in years_df['effective_season'].tolist() if pd.notna(s)]
                        available_years = sorted(list(set(s -1 for s in valid_effective_seasons if pd.notna(s))), reverse=True)
                except pd.io.sql.DatabaseError:
                     pass
                if not available_years:
                    seasons_df = pd.read_sql_query("SELECT DISTINCT season FROM SeasonTeamSummaries ORDER BY season DESC", conn_temp)
                    if not seasons_df.empty:
                         valid_seasons = [s for s in seasons_df['season'].tolist() if pd.notna(s)]
                         available_years = sorted(list(set(s -1 for s in valid_seasons if pd.notna(s))), reverse=True)
            else:
                seasons_df = pd.read_sql_query("SELECT DISTINCT season FROM SeasonTeamSummaries ORDER BY season DESC", conn_temp)
                if not seasons_df.empty:
                    available_years = [s for s in seasons_df['season'].tolist() if pd.notna(s)]

            if available_years:
                latest_year = available_years[0]
        except Exception as e:
            print(f"Error fetching available seasons/years: {e}")
        finally:
            if conn_temp: conn_temp.close()

    current_year_to_display = specific_season_or_year if specific_season_or_year is not None else latest_year
    if current_year_to_display is None:
        current_year_to_display = "N/A"
    
    return {
        "all_available_seasons": available_years,
        "current_season_displayed": current_year_to_display,
        "cid_to_conf_abbrev_map": cid_to_conf_abbrev_map,
        "tid_to_school_name_map": tid_to_school_name_map,
        "active_tab": active_tab_name,
        "now": {'year': datetime.now().year}
    }

@app.route('/')
@app.route('/rankings/')
@app.route('/rankings/<int:season_year>')
def rankings(season_year=None):
    common_context = get_common_template_context('teams', season_year)
    
    target_season_for_load = season_year
    if target_season_for_load is None and common_context["all_available_seasons"] and \
       common_context["all_available_seasons"][0] != "N/A":
        target_season_for_load = common_context["all_available_seasons"][0]
            
    team_summaries_df, season_data_actually_loaded_for = database_ops.load_season_summary_for_display(target_season_for_load)
    
    if season_data_actually_loaded_for is not None:
        common_context["current_season_displayed"] = season_data_actually_loaded_for
    elif target_season_for_load is not None:
         common_context["current_season_displayed"] = target_season_for_load
    else:
        common_context["current_season_displayed"] = "N/A"

    summary_data = []
    if not team_summaries_df.empty:
        summary_data = team_summaries_df.to_dict(orient='records')
    
    return render_template('rankings.html',
                           summaries=summary_data,
                           title=f"BrentPom Rankings - Season {common_context['current_season_displayed']}",
                           **common_context)

@app.route('/team/<int:team_tid>/')
@app.route('/team/<int:team_tid>/<int:season_year>')
def team_page(team_tid, season_year=None):
    common_context = get_common_template_context('teams', season_year)

    target_season_for_data_load = season_year
    if target_season_for_data_load is None and common_context["all_available_seasons"] and \
       common_context["all_available_seasons"][0] != "N/A":
        target_season_for_data_load = common_context["all_available_seasons"][0]
    
    team_details = None
    game_log_df = pd.DataFrame()
    historical_summary_df = pd.DataFrame()
    # Use the global tid_to_school_name_map for initial title, fallback to abbrev if needed
    team_abbrev_for_title = tid_to_school_name_map.get(team_tid, "Unknown Team")

    # If still "Unknown Team", try to get actual abbrev from DB for a better title
    if team_abbrev_for_title == "Unknown Team":
        conn_abbrev = database_ops.get_db_connection()
        if conn_abbrev:
            try:
                abbrev_query_df = pd.read_sql_query(f"SELECT abbrev FROM Teams WHERE team_tid = {team_tid}", conn_abbrev)
                if not abbrev_query_df.empty:
                    team_abbrev_for_title = abbrev_query_df['abbrev'].iloc[0]
            except Exception as e:
                print(f"Error fetching team abbrev for title fallback for team_tid {team_tid}: {e}")
            finally:
                if conn_abbrev: conn_abbrev.close()

    if target_season_for_data_load and target_season_for_data_load != "N/A":
        team_details, season_info_from_db_call = database_ops.load_team_season_details(team_tid, target_season_for_data_load)
        
        if season_info_from_db_call is not None:
            common_context["current_season_displayed"] = season_info_from_db_call
        else:
            common_context["current_season_displayed"] = target_season_for_data_load
            
        if team_details:
            team_abbrev_for_title = team_details.get('team_abbrev', team_abbrev_for_title)
            game_log_df = database_ops.load_team_game_log(team_tid, common_context["current_season_displayed"])
    else:
        common_context["current_season_displayed"] = "N/A"
            
    historical_summary_df = database_ops.load_team_historical_summary(team_tid)
    
    game_log_data = [] if game_log_df.empty else game_log_df.to_dict(orient='records')
    
    historical_summary_data = []
    if not historical_summary_df.empty:
        temp_historical_list = historical_summary_df.to_dict(orient='records')
        for history_item in temp_historical_list:
            if 'coach_id' not in history_item or pd.isna(history_item.get('coach_id')):
                history_item['coach_id'] = None
            else:
                try: history_item['coach_id'] = int(history_item['coach_id'])
                except: history_item['coach_id'] = None
            if 'coach_name' not in history_item or pd.isna(history_item.get('coach_name')):
                 history_item['coach_name'] = "N/A"
            historical_summary_data.append(history_item)

    key_games_list = [] # Populate this if you re-add key games logic
    if not game_log_df.empty:
        played_games = game_log_df[game_log_df['is_played'] == True].copy()
        if not played_games.empty:
            quadrant_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
            if 'game_quadrant' in played_games.columns:
                played_games_with_valid_q = played_games[played_games['game_quadrant'].isin(quadrant_order.keys())].copy()
                if not played_games_with_valid_q.empty:
                    played_games_with_valid_q.loc[:, 'quad_sort_key'] = played_games_with_valid_q['game_quadrant'].map(quadrant_order)
                    if 'win' in played_games_with_valid_q.columns:
                        played_games_with_valid_q.loc[:, 'win_for_sort'] = pd.to_numeric(played_games_with_valid_q['win'], errors='coerce').fillna(0).astype(int)
                    else: played_games_with_valid_q.loc[:, 'win_for_sort'] = 0
                    key_games_df = played_games_with_valid_q.sort_values(by=['quad_sort_key', 'win_for_sort'], ascending=[True, False]).head(8)
                    if not key_games_df.empty: key_games_list = key_games_df.to_dict(orient='records')

    page_title_season_part = common_context['current_season_displayed']
    if page_title_season_part == "N/A" and not historical_summary_data:
        page_title_season_part = "Info"

    page_title = f"{team_abbrev_for_title} - {page_title_season_part} - BrentPom"
    if team_details is None and common_context['current_season_displayed'] != "N/A":
        page_title = f"{team_abbrev_for_title} (No Data for {page_title_season_part}) - BrentPom"
    elif team_details is None and not historical_summary_data and team_abbrev_for_title != "Unknown Team":
         page_title = f"{team_abbrev_for_title} - No Data Available - BrentPom"

    return render_template('team_page.html',
                           team_details=team_details,
                           game_log=game_log_data,
                           historical_summary=historical_summary_data,
                           key_games=key_games_list,
                           team_abbrev_for_title=team_abbrev_for_title,
                           title=page_title,
                           **common_context)

@app.route('/conferences/')
@app.route('/conferences/<int:season_year>')
def conferences_page(season_year=None): # This is the main CONFERENCES LIST page
    common_context = get_common_template_context('conferences', season_year)
    
    target_season_for_load = season_year
    if target_season_for_load is None and common_context["all_available_seasons"] and \
       common_context["all_available_seasons"][0] != "N/A":
        target_season_for_load = common_context["all_available_seasons"][0]
    
    conference_summary_list = []
    if target_season_for_load and target_season_for_load != "N/A":
        # Load all team summaries for the season to perform conference aggregations
        all_team_summaries_df, loaded_season = database_ops.load_season_summary_for_display(target_season_for_load)
        
        if loaded_season is not None: # Ensure common_context reflects the season data is for
            common_context["current_season_displayed"] = loaded_season

        if not all_team_summaries_df.empty:
            print(f"INFO APP: Calculating conference aggregates for season {loaded_season}...")
            # Ensure necessary columns are numeric for aggregation
            cols_to_numeric = ['wins', 'losses', 'games_played', 'adj_em', 'adj_o', 'adj_d', 'avg_opp_adj_em']
            for col in cols_to_numeric:
                if col in all_team_summaries_df.columns:
                    all_team_summaries_df[col] = pd.to_numeric(all_team_summaries_df[col], errors='coerce')

            # Group by conference ID (cid)
            # Before grouping, ensure 'cid' is not NaN, replace with a placeholder if necessary
            all_team_summaries_df['cid'] = all_team_summaries_df['cid'].fillna(-999) # Use a known placeholder for "Independent" or unassigned

            conference_groups = all_team_summaries_df.groupby('cid')
            
            for conf_id, group_df in conference_groups:
                if conf_id == -999 and len(group_df) == 0 : # Skip if only placeholder and no teams
                    continue

                conf_abbrev = common_context["cid_to_conf_abbrev_map"].get(conf_id, f"Conf ID {conf_id}")
                conf_name = conf_abbrev # Ideally, load full conference name too if available in a map
                
                # If you have a cid_to_conf_full_name_map:
                # conf_name = cid_to_conf_full_name_map.get(conf_id, conf_abbrev)

                total_wins = group_df['wins'].sum()
                total_losses = group_df['losses'].sum()
                total_games = group_df['games_played'].sum() # Or total_wins + total_losses
                
                conf_win_pct = 0.0
                if total_games > 0:
                    conf_win_pct = total_wins / total_games
                
                avg_adjem = group_df['adj_em'].mean()
                avg_adjo = group_df['adj_o'].mean()
                avg_adjd = group_df['adj_d'].mean()
                avg_adjsos = group_df['avg_opp_adj_em'].mean() # This is the AdjEM SOS

                conference_summary_list.append({
                    'cid': conf_id,
                    'conf_name': conf_name, # Will display abbrev for now
                    'conf_abbrev': conf_abbrev,
                    'total_wins': int(total_wins),
                    'total_losses': int(total_losses),
                    'conf_win_pct': conf_win_pct,
                    'avg_adjem': avg_adjem,
                    'avg_adjo': avg_adjo,
                    'avg_adjd': avg_adjd,
                    'avg_adjsos': avg_adjsos
                })
            
            # Sort conferences by average AdjEM (optional)
            conference_summary_list = sorted(conference_summary_list, key=lambda x: x.get('avg_adjem', -float('inf')), reverse=True)
            print(f"INFO APP: Calculated aggregates for {len(conference_summary_list)} conferences.")
        else:
            print(f"INFO APP: No team summaries found for season {target_season_for_load} to aggregate conference stats.")

    return render_template('conferences_list.html',
                            page_title=f"Conference Overview - Season {common_context['current_season_displayed']}",
                            conference_summaries=conference_summary_list,
                            **common_context)

@app.route('/coaches/')
@app.route('/coaches/<int:season_year>/') # Add optional season year
@app.route('/coaches/')
def coaches_list_page():
    common_context = get_common_template_context('coaches', None)

    # Use the new function to get enhanced career stats
    all_coaches_data = database_ops.load_all_coaches_with_enhanced_career_stats()

    return render_template('coaches_list.html',
                            page_title="All Coaches",
                            coaches=all_coaches_data, # This list now has the calculated fields
                            **common_context)
@app.route('/coach/<int:coach_id>/')
# @app.route('/coach/<int:coach_id>/<int:season_year>/') # We can make season optional for initial view
def coach_detail_page(coach_id, season_year=None): # season_year is optional here
    # Determine a season for context, e.g., latest available or latest for this coach
    # For now, get_common_template_context handles defaulting to overall latest season if season_year is None
    common_context = get_common_template_context('coaches', season_year)
    
    coach_info = database_ops.load_coach_info(coach_id)
    career_stats = database_ops.load_coach_career_stats_by_id(coach_id)
    seasonal_stats_df = database_ops.load_coach_seasons_stats_by_id(coach_id)

    coach_name_for_title = coach_info.get('coach_name', f"Coach ID {coach_id}") if coach_info else f"Coach ID {coach_id}"

    seasonal_stats_data = []
    if not seasonal_stats_df.empty:
        seasonal_stats_data = seasonal_stats_df.to_dict(orient='records')

    # If a specific season_year was passed and is valid, use it for context
    # Otherwise, current_season_displayed from common_context will be the latest general season
    display_season_context = season_year if season_year else common_context['current_season_displayed']


    return render_template('coach_detail.html',
                            coach_info=coach_info, # dict or None
                            career_stats=career_stats, # dict or None
                            seasonal_stats=seasonal_stats_data, # list of dicts
                            title=f"Coach Profile: {coach_name_for_title}",
                            # Pass the specific season context if relevant for this page
                            # For example, if you want the season selector to default to a coach's most recent season
                            current_season_displayed_context=display_season_context,
                            **common_context)
@app.route('/recruiting/')
@app.route('/recruiting/<int:class_year>')
def recruiting_page(class_year=None):
    # Use is_class_year=True for get_common_template_context
    common_context = get_common_template_context('recruiting', class_year, is_class_year=True)
    
    target_class_year_for_load = class_year
    # If no class_year specified, and available_class_years exist, default to the latest one
    if target_class_year_for_load is None and common_context["all_available_seasons"] and \
       common_context["all_available_seasons"][0] != "N/A":
        target_class_year_for_load = common_context["all_available_seasons"][0]

    recruiting_rankings_data = []
    effective_season_for_data = "N/A"

    if target_class_year_for_load and target_class_year_for_load != "N/A":
        recruiting_df, effective_season_loaded = database_ops.load_recruiting_rankings_for_class_year(target_class_year_for_load)
        if not recruiting_df.empty:
            recruiting_rankings_data = recruiting_df.to_dict(orient='records')
        if effective_season_loaded is not None:
            effective_season_for_data = effective_season_loaded
            # Update context for display (header might show playing season or class year based on tab)
            common_context["current_season_displayed"] = target_class_year_for_load # Keep this as class year for the label
    else:
        print(f"INFO APP: No target class year for recruiting page.")


    page_title = f"Recruiting Rankings - Class of {common_context['current_season_displayed']}"
    if not recruiting_rankings_data:
         print(f"INFO APP: No recruiting data to display for class {target_class_year_for_load} (effective season {effective_season_for_data}).")
        
    return render_template('recruiting_rankings.html', # NEW TEMPLATE
                           page_title=page_title,
                           recruiting_summaries=recruiting_rankings_data,
                           effective_season=effective_season_for_data, # To link team/coach for correct playing year
                           **common_context)
@app.route('/conference/<int:conf_id>/')
@app.route('/conference/<int:conf_id>/<int:season_year>')
def conference_detail_page(conf_id, season_year=None):
    common_context = get_common_template_context('conferences', season_year) # Keep tab active on 'conferences'

    target_season_for_load = season_year
    if target_season_for_load is None and common_context["all_available_seasons"] and \
       common_context["all_available_seasons"][0] != "N/A":
        target_season_for_load = common_context["all_available_seasons"][0]

    # Get conference name/abbrev for the title
    conference_name_display = common_context["cid_to_conf_abbrev_map"].get(conf_id, f"Conference ID {conf_id}")
    
    team_summaries_df, season_data_actually_loaded_for = database_ops.load_season_summary_for_display(target_season_for_load)

    # Update current_season_displayed in common_context based on what was actually loaded or attempted
    if season_data_actually_loaded_for is not None:
        common_context["current_season_displayed"] = season_data_actually_loaded_for
    elif target_season_for_load is not None:
         common_context["current_season_displayed"] = target_season_for_load
    else: # Neither specified nor found
        common_context["current_season_displayed"] = "N/A"


    filtered_summaries = []
    if not team_summaries_df.empty:
        # Ensure 'cid' is integer for proper comparison
        team_summaries_df['cid'] = pd.to_numeric(team_summaries_df['cid'], errors='coerce').fillna(-999).astype(int)
        conference_teams_df = team_summaries_df[team_summaries_df['cid'] == conf_id]
        if not conference_teams_df.empty:
            filtered_summaries = conference_teams_df.to_dict(orient='records')
        else:
            print(f"INFO APP: No teams found for conference ID {conf_id} in season {common_context['current_season_displayed']}.")
    else:
        if target_season_for_load is not None and target_season_for_load != "N/A":
             print(f"INFO APP: No team summaries loaded for season {target_season_for_load} to filter for conference {conf_id}.")
    
    page_title = f"{conference_name_display} Rankings - Season {common_context['current_season_displayed']}"

    return render_template('rankings.html',  # Reuse the main rankings template!
                           summaries=filtered_summaries,
                           title=page_title,
                           # Pass a flag or specific title to let rankings.html know it's a conference view
                           is_conference_view=True,
                           conference_name_for_view=conference_name_display,
                           **common_context)

if __name__ == '__main__':
    print(f"Attempting to run Flask app. Database expected at: {config.DATABASE_FILE_PATH}")
    if hasattr(config, 'DATABASE_FILE_PATH') and not os.path.exists(config.DATABASE_FILE_PATH):
        print(f"WARNING APP: Database file not found at {config.DATABASE_FILE_PATH}. Ensure main_processor.py has run and config.py is correct.")
    elif not hasattr(config, 'DATABASE_FILE_PATH'):
        print("WARNING APP: config.DATABASE_FILE_PATH is not defined.")
        
    app.run(debug=True, port=5001)
