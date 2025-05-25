# data_transformer.py
import pandas as pd
from collections import defaultdict
import numpy as np

def create_team_lookup(zengm_data):
    """
    Extracts team information from ZenGM JSON data and creates lookup tables.
    """
    if not zengm_data or 'teams' not in zengm_data:
        print("ERROR: 'teams' array not found in zengm_data or zengm_data is None.")
        return pd.DataFrame(), {}, {}

    teams_list = []
    for team_data in zengm_data.get('teams', []):
        tid = team_data.get('tid')
        cid = team_data.get('cid')
        region = team_data.get('region', '')
        name = team_data.get('name', '')
        abbrev = team_data.get('abbrev')
        
        if tid is None or abbrev is None:
            continue
            
        teams_list.append({
            'tid': tid, 'cid': cid, 'region': region, 'name': name,
            'abbrev': abbrev, 'full_name': f"{region} {name}".strip()
        })

    teams_df = pd.DataFrame(teams_list)
    if teams_df.empty:
        return teams_df, {}, {}

    if teams_df['abbrev'].duplicated().any():
        print(f"WARNING: Duplicate abbreviations found: {teams_df[teams_df['abbrev'].duplicated()]['abbrev'].tolist()}")
        
    try:
        unique_teams_for_tid_lookup = teams_df.drop_duplicates(subset=['tid'], keep='first')
        unique_teams_for_abbrev_lookup = teams_df.drop_duplicates(subset=['abbrev'], keep='first')
        
        tid_to_abbrev = pd.Series(unique_teams_for_tid_lookup.abbrev.values, index=unique_teams_for_tid_lookup.tid).to_dict()
        abbrev_to_tid = pd.Series(unique_teams_for_abbrev_lookup.tid.values, index=unique_teams_for_abbrev_lookup.abbrev).to_dict()
    except Exception as e:
        print(f"ERROR creating team lookup dictionaries: {e}")
        return pd.DataFrame(), {}, {}
    
    print(f"Processed {len(teams_df)} teams into lookup.")
    return teams_df, tid_to_abbrev, abbrev_to_tid

def extract_game_data_and_stats_from_json(zengm_data, tid_to_abbrev_lookup, current_season_year):
    # ... (initial checks, active_season, stat_rename_map, all_played_gids_in_season as in response #103) ...
    if not zengm_data: return pd.DataFrame()
    if current_season_year is None: return pd.DataFrame()
    active_season = int(current_season_year)
    processed_games_list = []
    team_season_game_counts_played_only = defaultdict(int)
    stat_rename_map = {'fg':'fgm', 'fga':'fga', 'tp':'fgm3', 'tpa':'fga3', 'ft':'ftm', 'fta':'fta', 'orb':'oreb', 'drb':'dreb', 'ast':'ast', 'tov':'tov', 'stl':'stl', 'blk':'blk', 'pf':'pf', 'min':'min', 'pts':'pts'}
    all_stat_cols_renamed = list(stat_rename_map.values())
    per_game_calculated_stat_keys = ['poss', 'raw_oe', 'raw_de', 'off_efg_pct', 'off_tov_pct', 'off_orb_pct', 'off_ft_rate', 'def_efg_pct', 'def_tov_pct', 'def_opp_orb_pct', 'def_ft_rate', 'team_drb_pct', 'team_2p_pct', 'opp_2p_pct', 'team_3p_pct', 'opp_3p_pct', 'team_3p_rate', 'opp_3p_rate']
    prediction_cols = ['pred_win_prob_team', 'pred_margin_team', 'pred_score_team', 'pred_score_opponent']
    all_played_gids_in_season = set()

    # --- Process PLAYED games ---
    if 'games' in zengm_data:
        played_games_in_json_for_season = sorted(
            [g for g in zengm_data.get('games', []) if g.get('season') == active_season],
            key=lambda g: g.get('gid', float('inf'))
        )
        for game_data in played_games_in_json_for_season:
            gid = game_data.get('gid')
            all_played_gids_in_season.add(gid)
            # ... (extract winner_tid, loser_tid, scores, team_objects, designated_home/away, tournament flags etc. as before) ...
            is_national_playoffs = game_data.get('playoffs', False)
            won_info = game_data.get('won', {}); lost_info = game_data.get('lost', {})
            winner_tid = won_info.get('tid'); winner_score = won_info.get('pts')
            loser_tid = lost_info.get('tid'); loser_score = lost_info.get('pts')
            if None in [gid, winner_tid, winner_score, loser_tid, loser_score] or 'teams' not in game_data or len(game_data['teams']) != 2: continue
            team_objects = game_data['teams']
            designated_home_tid = team_objects[0].get('tid'); designated_away_tid = team_objects[1].get('tid')
            if designated_home_tid is None or designated_away_tid is None: continue
            team_season_game_counts_played_only[(active_season, designated_home_tid)] +=1
            team_season_game_counts_played_only[(active_season, designated_away_tid)] +=1
            is_conf_tournament = game_data.get('type') == 'conferenceTournament'
            game_true_home_tid = None
            if not is_national_playoffs and not is_conf_tournament: game_true_home_tid = designated_home_tid
            stats_for_tid_in_game = {}
            for team_obj in team_objects:
                current_tid = team_obj.get('tid');
                if current_tid is None: continue
                summed_stats = {new_key: 0.0 for new_key in stat_rename_map.values()}
                for p_stat in team_obj.get('players', []):
                    for p_key, new_key in stat_rename_map.items():
                        try: summed_stats[new_key] += float(p_stat.get(p_key, 0))
                        except (ValueError, TypeError): pass
                stats_for_tid_in_game[current_tid] = summed_stats
                stats_for_tid_in_game[current_tid]['score_official'] = winner_score if current_tid == winner_tid else loser_score

            for i, team_obj in enumerate(team_objects):
                team_tid = team_obj.get('tid')
                opp_tid = team_objects[1-i].get('tid')
                if team_tid is None or opp_tid is None or team_tid not in stats_for_tid_in_game or opp_tid not in stats_for_tid_in_game: continue
                loc = 'Neutral'
                if game_true_home_tid is not None: loc = 'Home' if team_tid == game_true_home_tid else 'Away'
                
                team_actual_score = stats_for_tid_in_game[team_tid]['score_official']
                opp_actual_score = stats_for_tid_in_game[opp_tid]['score_official']

                entry = {
                    'gid': gid, 'season': active_season, 'team_tid': team_tid, 'opponent_tid': opp_tid,
                    'team_abbrev': tid_to_abbrev_lookup.get(team_tid, "N/A"),
                    'opponent_abbrev': tid_to_abbrev_lookup.get(opp_tid, "N/A"),
                    'location': loc, 'overtimes': game_data.get('overtimes', 0),
                    'is_national_playoffs': is_national_playoffs,
                    'is_conf_tournament': is_conf_tournament,
                    'team_score_official': team_actual_score,
                    'opponent_score_official': opp_actual_score,
                    'designated_home_tid': designated_home_tid,
                    'designated_away_tid': designated_away_tid,
                    'is_played': True,
                    'game_date': game_data.get('day'), 'game_time': None,
                    'win': 1 if team_actual_score > opp_actual_score else 0,  # Explicitly add win/loss
                    'loss': 1 if team_actual_score < opp_actual_score else 0, # Explicitly add win/loss
                    'game_quadrant': np.nan
                }
                for stat_key_original, stat_val in stats_for_tid_in_game[team_tid].items():
                    if stat_key_original != 'score_official': entry[f"team_{stat_key_original}"] = stat_val
                for stat_key_original, stat_val in stats_for_tid_in_game[opp_tid].items():
                    if stat_key_original != 'score_official': entry[f"opponent_{stat_key_original}"] = stat_val
                for stat_col in per_game_calculated_stat_keys:
                    if stat_col not in entry: entry[stat_col] = np.nan
                for pred_col in prediction_cols: entry[pred_col] = np.nan
                processed_games_list.append(entry)

    # --- Process FUTURE games from 'schedule' array ---
    if 'schedule' in zengm_data:
        for scheduled_game in zengm_data.get('schedule', []):
            gid = scheduled_game.get('gid')
            if gid is None or (gid in all_played_gids_in_season): continue
            home_tid = scheduled_game.get('homeTid'); away_tid = scheduled_game.get('awayTid')
            if home_tid is None or away_tid is None: continue
            game_date_future = scheduled_game.get('day', scheduled_game.get('date')); game_time_future = scheduled_game.get('time', None)
            s_type = str(scheduled_game.get('type', '')).lower()
            is_playoffs_future = 'playoffs' in s_type or 'national' in s_type
            is_conf_tourney_future = 'conference' in s_type and 'tournament' in s_type
            location_future_home = 'Neutral' if is_playoffs_future or is_conf_tourney_future else 'Home'
            location_future_away = 'Neutral' if is_playoffs_future or is_conf_tourney_future else 'Away'
            if scheduled_game.get('neutralSite', False) is True: location_future_home = 'Neutral'; location_future_away = 'Neutral'

            for participant_perspective in [(home_tid, away_tid, location_future_home), (away_tid, home_tid, location_future_away)]:
                team_tid, opp_tid, loc = participant_perspective
                entry = {
                    'gid': gid, 'season': active_season, 'team_tid': team_tid, 'opponent_tid': opp_tid,
                    'team_abbrev': tid_to_abbrev_lookup.get(team_tid, "N/A"),
                    'opponent_abbrev': tid_to_abbrev_lookup.get(opp_tid, "N/A"),
                    'location': loc, 'overtimes': 0, 'is_played': False,
                    'game_date': game_date_future, 'game_time': game_time_future,
                    'is_national_playoffs': is_playoffs_future, 'is_conf_tournament': is_conf_tourney_future,
                    'team_score_official': np.nan, 'opponent_score_official': np.nan,
                    'designated_home_tid': home_tid, 'designated_away_tid': away_tid,
                    'win': np.nan, 'loss': np.nan, # Explicitly NaN for future games
                    'game_quadrant': np.nan
                }
                for sk_orig, sk_renamed in stat_rename_map.items():
                    entry[f"team_{sk_renamed}"] = np.nan; entry[f"opponent_{sk_renamed}"] = np.nan
                for pgc_stat_key in per_game_calculated_stat_keys: entry[pgc_stat_key] = np.nan
                for pred_col in prediction_cols: entry[pred_col] = np.nan
                processed_games_list.append(entry)

    if not processed_games_list: return pd.DataFrame()
    all_games_df = pd.DataFrame(processed_games_list)

    # Finalize team_game_num_in_season
    sort_keys = ['season', 'team_tid']
    if 'game_date' in all_games_df.columns and pd.to_numeric(all_games_df['game_date'], errors='coerce').notna().any():
        all_games_df['sort_key_day'] = pd.to_numeric(all_games_df['game_date'], errors='coerce').fillna(float('inf'))
        sort_keys.extend(['is_played', 'sort_key_day', 'gid']) # is_played False (future) comes after True (played) if ascending=True
        all_games_df.sort_values(by=sort_keys, ascending=[True, True, False, True, True], inplace=True) # is_played False comes after True
        all_games_df.drop(columns=['sort_key_day'], inplace=True)
    else:
        sort_keys.extend(['is_played', 'gid'])
        all_games_df.sort_values(by=sort_keys, ascending=[True, True, False, True], inplace=True)
    
    all_games_df['team_game_num_in_season'] = all_games_df.groupby(['season', 'team_tid']).cumcount() + 1
    
    print(f"Finalized processing {len(all_games_df)} total team-game rows (played & future) for season {active_season}.")
    return all_games_df
