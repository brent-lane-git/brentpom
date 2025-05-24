# app.py
from flask import Flask, render_template, request
import database_ops # Our existing module to fetch data
import config       # To potentially get the current season or other defaults
import os

app = Flask(__name__)

# Determine the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Update config.DATABASE_FILE_PATH if it's relative to BASE_DIR and not already absolute
# This ensures database_ops uses the correct path when called from Flask.
# If config.DATABASE_FILE_PATH is already absolute or correctly relative from where it's defined, this might not be needed.
# However, it's safer if database_ops.DB_PATH relies on an absolute path defined in config.
# Assuming config.DATABASE_FILE_PATH is already correctly set up.


@app.route('/')
@app.route('/rankings')
@app.route('/rankings/<int:season>')
def rankings(season=None):
    # Determine the season to display
    active_season = season
    if active_season is None:
        # Try to get the latest season from gameAttributes in config or a default
        # This is a simplified way to get current season for now
        # In a more complex app, this might come from a settings table or most recent DB entry.
        if hasattr(config, 'CURRENT_PROCESSING_SEASON_FOR_WEB'): # You might add this to config
            active_season = config.CURRENT_PROCESSING_SEASON_FOR_WEB
        else:
            # Fallback: Try to get the most recent season from the database directly
            # This is handled by load_season_summary_for_display if season is None
            pass


    team_summaries_df = database_ops.load_season_summary_for_display(active_season)
    
    # If a specific season was requested but no data found, team_summaries_df might be empty
    # Get the actual season loaded if active_season was None initially
    if not team_summaries_df.empty and active_season is None:
        active_season = team_summaries_df['season'].iloc[0]
    elif team_summaries_df.empty and active_season is None:
        active_season = "N/A" # Or some default if no data at all

    # Convert DataFrame to a list of dictionaries for easier use in template
    summary_data = team_summaries_df.to_dict(orient='records')
    
    return render_template('rankings.html',
                           summaries=summary_data,
                           current_season=active_season,
                           title=f"BrentPom Rankings - Season {active_season}")

if __name__ == '__main__':
    # Make sure to run create_tables_if_not_exist() once if DB might be new
    # This is usually part of your main_processor.py, but for a standalone web app run:
    # database_ops.create_tables_if_not_exist()
    app.run(debug=True) # debug=True is helpful for development
