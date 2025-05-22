# data_transformer.py
import pandas as pd
from collections import defaultdict # For easy counting

def create_team_lookup(zengm_data):
    """
    Extracts team information from ZenGM JSON data and creates lookup tables.

    Args:
        zengm_data (dict): The loaded ZenGM league data.

    Returns:
        tuple: (teams_df, tid_to_abbrev, abbrev_to_tid)
               - teams_df (pd.DataFrame): DataFrame with columns ['tid', 'cid', 'region', 'name', 'abbrev', 'full_name'].
               - tid_to_abbrev (dict): Mapping from team ID (tid) to abbreviation.
               - abbrev_to_tid (dict): Mapping from team abbreviation to team ID (tid).
    """
    if not zengm_data or 'teams' not in zengm_data:
        print("ERROR: 'teams' array not found in zengm_data or zengm_data is None.")
        return pd.DataFrame(), {}, {}

    teams_list = []
    for team_data in zengm_data.get('teams', []):
        tid = team_data.get('tid')
        cid = team_data.get('cid') # Conference ID
        region = team_data.get('region', '')
        name = team_data.get('name', '')
        abbrev = team_data.get('abbrev')
        
        if tid is None or abbrev is None:
            # print(f"WARNING: Team data missing tid or abbrev, skipping: {team_data}") # Can be verbose
            continue
            
        teams_list.append({
            'tid': tid,
            'cid': cid,
            'region': region,
            'name': name,
            'abbrev': abbrev,
            'full_name': f"{region} {name}".strip()
        })

    teams_df = pd.DataFrame(teams_list)
    if teams_df.empty:
        # print("WARNING: No team data processed into DataFrame.") # Can be verbose
        return teams_df, {}, {}

    # Check for duplicate abbreviations, which could cause issues if not handled.
    if teams_df['abbrev'].duplicated().any():
        print(f"WARNING: Duplicate abbreviations found in team data: {teams_df[teams_df['abbrev'].duplicated()]['abbrev'].tolist()}")
        # Pandas default for to_dict on Series with duplicate index is to raise error or take last.
        # We should ensure unique abbrevs if they are primary keys for lookup.
        # For now, this just creates the dicts.
        
    try:
        tid_to_abbrev = pd.Series(teams_df.abbrev.values, index=teams_df.tid).to_dict()
        abbrev_to_tid = pd.Series(teams_df.tid.values, index=teams_df.abbrev).to_dict()
    except Exception as e:
        print(f"ERROR creating team lookup dictionaries, possibly due to duplicate abbreviations: {e}")
        # Fallback or handle duplicates if necessary. For now, let it fail if duplicates cause issues here.
        # A more robust way might be to group by abbrev/tid and pick first/last or error out.
        return pd.DataFrame(), {}, {}

    
    print(f"Processed {len(teams_df)} teams into lookup.")
    return teams_df, tid_to_abbrev, abbrev_to_tid

def extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev_lookup):
    """
    Extracts game details and SUMMED player stats from the ZenGM JSON 'games' array.
    Each game will result in two rows in the output DataFrame, one for each team.
    Home team for non-playoff, non-conference tournament games is determined by the 
    first team in game_data['teams'].
    Conference tournament games are identified if playoffs=false and game_num >= 35 for either team.
    """
    if not zengm_data or 'games' not in zengm_data:
        print("ERROR: 'games' array not found in zengm_data or zengm_data is None.")
        return pd.DataFrame()

    processed_game_team_stats_list = []
    
    # Stores game played count for each team in each season: (season, tid) -> count
    team_season_game_counts = defaultdict(int)

    # Sort games by gid to process them chronologically for game counting
    # Assumes gid is a reliable chronological indicator
    all_games_in_json = sorted(zengm_data.get('games', []), key=lambda g: g.get('gid', float('inf')))
    if not all_games_in_json:
        print("WARNING: No games found in zengm_data['games'] to process.")
        return pd.DataFrame()
        
    print(f"INFO: Processing {len(all_games_in_json)} games, sorted by gid.")

    for game_data in all_games_in_json: # Iterate over sorted games
        gid = game_data.get('gid')
        season = game_data.get('season')
        is_national_playoffs = game_data.get('playoffs', False) # True for national tournament

        won_info = game_data.get('won', {})
        lost_info = game_data.get('lost', {})
        winner_tid = won_info.get('tid')
        winner_score_official = won_info.get('pts')
        loser_tid = lost_info.get('tid')
        loser_score_official = lost_info.get('pts')

        if None in [gid, season, winner_tid, winner_score_official, loser_tid, loser_score_official]:
            # print(f"WARNING: Game data (gid {gid}) missing some core fields (winner/loser info). Skipping.") # Can be verbose
            continue
        
        if 'teams' not in game_data or len(game_data['teams']) != 2:
            # print(f"WARNING: Game data (gid {gid}) does not have 2 teams in its 'teams' array. Skipping.") # Can be verbose
            continue

        # Team TIDs based on their order in the 'teams' array
        # Per user discovery: game_data['teams'][0] is the "designated" home team in non-playoff, non-conf_tourney games
        designated_home_tid_for_game = game_data['teams'][0].get('tid') # Team at index 0
        designated_away_tid_for_game = game_data['teams'][1].get('tid') # Team at index 1

        if designated_home_tid_for_game is None or designated_away_tid_for_game is None:
            # print(f"WARNING: Game data (gid {gid}) team objects missing tids. Skipping game.") # Can be verbose
            continue
        
        # Increment game counts for this game (count includes current game being processed)
        # These counts are specific to this season for each team
        game_count_team_designated_home = team_season_game_counts.get((season, designated_home_tid_for_game), 0) + 1
        team_season_game_counts[(season, designated_home_tid_for_game)] = game_count_team_designated_home
        
        game_count_team_designated_away = team_season_game_counts.get((season, designated_away_tid_for_game), 0) + 1
        team_season_game_counts[(season, designated_away_tid_for_game)] = game_count_team_designated_away

        # Determine if it's a conference tournament game based on user's rule
        is_conf_tournament_game = False
        if not is_national_playoffs: # Only consider if not already national playoffs
            # Check if EITHER team is playing their 35th+ game this season
            if game_count_team_designated_home >= 35 or game_count_team_designated_away >= 35:
                is_conf_tournament_game = True
        
        # Determine actual game home TID for location assignment (None if neutral)
        game_true_home_tid = None
        if not is_national_playoffs and not is_conf_tournament_game:
            game_true_home_tid = designated_home_tid_for_game # First team in 'teams' array is home

        stats_for_tid = {} # To store summed stats for each tid in this game
        for team_obj in game_data['teams']:
            current_tid = team_obj.get('tid')
            if current_tid is None: continue

            # Define keys for stats we need to sum from player objects
            # JSON snippet used 'tp' for 3PM and 'tpa' for 3PA.
            player_stat_keys_to_sum = ['pts', 'fg', 'fga', 'tp', 'tpa', 'ft', 'fta', 'orb', 'drb', 'ast', 'tov', 'stl', 'blk', 'pf', 'min']
            stat_rename_map = {
                'fg': 'fgm', 'fga': 'fga',
                'tp': 'fgm3', 'tpa': 'fga3', # three-pointers made/attempted
                'ft': 'ftm', 'fta': 'fta',   # free throws made/attempted
                'orb': 'oreb', 'drb': 'dreb', # offensive/defensive rebounds
                'ast': 'ast', 'tov': 'tov',
                'stl': 'stl', 'blk': 'blk',
                'pf': 'pf', 'min':'min', 'pts':'pts'
            }
            summed_team_stats = {renamed_key: 0 for renamed_key in stat_rename_map.values()}

            for player_data in team_obj.get('players', []):
                for original_key, renamed_key in stat_rename_map.items():
                    # Ensure the original_key exists in player_data, otherwise default to 0
                    stat_value = player_data.get(original_key, 0)
                    if pd.notna(stat_value): # Check if not NaN before trying to convert/add
                        try:
                            numeric_value = float(stat_value)
                            if pd.notna(numeric_value): # Ensure float conversion didn't result in NaN (e.g. from empty string)
                                summed_team_stats[renamed_key] += numeric_value
                        except (ValueError, TypeError):
                            # print(f"Warning: Non-numeric player stat {original_key}='{stat_value}' for pid {player_data.get('pid')} in gid {gid}. Skipping this stat for this player.")
                            pass # Ignore non-convertible non-numeric player stats for this specific stat
            
            stats_for_tid[current_tid] = summed_team_stats
            # Store official score as well for this tid
            if current_tid == winner_tid:
                stats_for_tid[current_tid]['score_official'] = winner_score_official
            elif current_tid == loser_tid:
                stats_for_tid[current_tid]['score_official'] = loser_score_official
            else: # Should not happen if tids in teams array match winner/loser tids
                  # Or if a game has no winner/loser explicitly (e.g. a tie, though rare in BBGM exports)
                stats_for_tid[current_tid]['score_official'] = team_obj.get('pts') # Fallback to team obj pts


        # Create the two rows for this game, one for each team's perspective
        for team_idx_in_json_array, team_obj_in_game in enumerate(game_data['teams']):
            team_tid_for_row = team_obj_in_game.get('tid')
            # The opponent is the other team in game_data['teams']
            opponent_tid_for_row = game_data['teams'][1 - team_idx_in_json_array].get('tid')

            if team_tid_for_row is None or opponent_tid_for_row is None or \
               team_tid_for_row not in stats_for_tid or opponent_tid_for_row not in stats_for_tid:
                # print(f"Warning: Summed stats missing for gid {gid}, team {team_tid_for_row} or opp {opponent_tid_for_row}. Skipping row creation.")
                continue

            team_stats_dict = stats_for_tid[team_tid_for_row]
            opponent_stats_dict = stats_for_tid[opponent_tid_for_row]
            
            location_for_this_team = 'Error_Loc_Assignment' # Default before assignment
            if is_national_playoffs or is_conf_tournament_game:
                location_for_this_team = 'Neutral'
            else: # Regular season, not conference tournament
                if team_tid_for_row == game_true_home_tid: # game_true_home_tid is from earlier non-playoff, non-conf-tourney logic
                    location_for_this_team = 'Home'
                # team_tid_for_row could also be game_true_away_tid if game_true_home_tid is not None
                elif game_true_home_tid is not None and team_tid_for_row != game_true_home_tid:
                    location_for_this_team = 'Away'
                else: # Should not be hit if game_true_home_tid is correctly one of the participants
                    location_for_this_team = 'Unknown_H/A_Logic_Issue'
            
            this_team_game_num = team_season_game_counts.get((season, team_tid_for_row))

            entry = {
                'gid': gid,
                'season': season,
                'is_national_playoffs': is_national_playoffs,
                'is_conf_tournament': is_conf_tournament_game,
                'team_game_num_in_season': this_team_game_num,
                'team_tid': team_tid_for_row,
                'team_abbrev': tid_to_abbrev_lookup.get(team_tid_for_row, 'N/A'),
                'opponent_tid': opponent_tid_for_row,
                'opponent_abbrev': tid_to_abbrev_lookup.get(opponent_tid_for_row, 'N/A'),
                'team_score_official': team_stats_dict.get('score_official'),
                'opponent_score_official': opponent_stats_dict.get('score_official'),
                'location': location_for_this_team,
                # designated_home_tid is always the first team in the JSON 'teams' array for that game
                'designated_home_tid': designated_home_tid_for_game,
                'designated_away_tid': designated_away_tid_for_game,
                # Add all summed stats for the team (prefix with 'team_')
                **{f"team_{key}": val for key, val in team_stats_dict.items() if key != 'score_official'},
                # Add all summed stats for the opponent (prefix with 'opponent_')
                **{f"opponent_{key}": val for key, val in opponent_stats_dict.items() if key != 'score_official'}
            }
            processed_game_team_stats_list.append(entry)

    games_expanded_df = pd.DataFrame(processed_game_team_stats_list)
    
    # Final check for location assignment issues
    unknown_loc_count = 0
    if 'location' in games_expanded_df.columns:
        unknown_loc_count = games_expanded_df[games_expanded_df['location'].isin(['Error_Loc_Assignment', 'Unknown_H/A_Logic_Issue']) | games_expanded_df['location'].isna()].shape[0]
    if unknown_loc_count > 0:
        print(f"WARNING: {unknown_loc_count} team-game rows had an 'Error_Loc_Assignment', 'Unknown_H/A_Logic_Issue', or missing location. Review H/A logic.")

    print(f"Processed {len(all_games_in_json)} distinct games from JSON, resulting in {len(games_expanded_df)} team-game rows.")
    return games_expanded_df
