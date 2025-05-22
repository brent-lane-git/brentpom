# data_loader.py
import json
import pandas as pd
import ijson

def load_json_file(file_path):
    """
    Loads a large JSON file using ijson's iterative parsing to build the main object,
    and handles potential UTF-8 BOM.
    """
    data = {}
    try:
        # Open in text mode ('r') with 'utf-8-sig' encoding to handle BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            print(f"Attempting to load JSON with ijson (handling BOM): {file_path}")
            # ijson.kvitems can work with text IO objects
            data = dict(ijson.kvitems(f, ''))
        print(f"Successfully loaded and reconstructed JSON object using ijson: {file_path}")
        return data
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at '{file_path}'.")
    except ijson.JSONError as e:
        print(f"ERROR: ijson could not decode JSON from '{file_path}'. Error: {e}")
        print("This might indicate a genuine syntax error beyond the BOM, or an issue with the file structure for ijson.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading JSON '{file_path}' with ijson: {e}")
    return None

def load_csv_file(file_path, expected_cols=None):
    """Loads a CSV file into a pandas DataFrame."""
    # ... (rest of this function remains the same) ...
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV: {file_path}")
        if expected_cols:
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                print(f"WARNING: CSV '{file_path}' is missing expected columns: {', '.join(missing_cols)}.")
        return df
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: CSV file at '{file_path}' is empty.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading CSV '{file_path}': {e}")
    return None
