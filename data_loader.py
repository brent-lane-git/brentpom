# data_loader.py
import json
import pandas as pd
import ijson # For loading large JSON files

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
    except ijson.JSONError as e: # Catch ijson specific errors
        print(f"ERROR: ijson could not decode JSON from '{file_path}'. Error: {e}")
        print("This might indicate a genuine syntax error beyond the BOM, or an issue with the file structure for ijson.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading JSON '{file_path}' with ijson: {e}")
    return None

def load_csv_file(file_path, expected_cols=None):
    """
    Loads a generic CSV file into a pandas DataFrame.
    (This was originally for game stats CSV, now more general).
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV: {file_path}")
        if expected_cols:
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                print(f"WARNING: CSV '{file_path}' is missing some of the expected columns: {', '.join(missing_cols)}.")
        return df
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: CSV file at '{file_path}' is empty.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading CSV '{file_path}': {e}")
    return None

def load_recruiting_csv(file_path, selected_columns=None):
    """
    Loads the recruiting CSV file into a pandas DataFrame, optionally selecting specific columns.
    Args:
        file_path (str): The path to the recruiting CSV file.
        selected_columns (list, optional): Specific columns to load. If None, loads all.
                                           It's good practice to specify to catch errors early
                                           if CSV format changes.
    Returns:
        pandas.DataFrame: The loaded data as a DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, usecols=selected_columns)
        print(f"Successfully loaded Recruiting CSV: {file_path}")
        
        # Verify that all selected columns were actually found if usecols was effective
        if selected_columns:
            missing_sel_cols = [col for col in selected_columns if col not in df.columns]
            if missing_sel_cols: # This should ideally be caught by read_csv if usecols has non-existent col
                print(f"WARNING: Recruiting CSV is missing some of the specifically requested columns after load: {missing_sel_cols}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Recruiting CSV file not found at '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: Recruiting CSV file at '{file_path}' is empty.")
    except ValueError as ve:
        # This can be raised if columns in 'usecols' are not in the file
        print(f"ERROR: Recruiting CSV column issue (e.g., selected columns in 'usecols' not found in file, or other parsing error): {ve}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading Recruiting CSV '{file_path}': {e}")
    return None
def load_coach_csv(file_path, selected_columns=None):
    """
    Loads the coach assignment CSV file into a pandas DataFrame.
    Args:
        file_path (str): The path to the coach CSV file.
        selected_columns (list, optional): Specific columns to load.
    Returns:
        pandas.DataFrame: The loaded data as a DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, usecols=selected_columns, keep_default_na=False, na_filter=False) # Treat empty strings as empty strings, not NaN
        print(f"Successfully loaded Coach CSV: {file_path}")
        if selected_columns:
            missing_sel_cols = [col for col in selected_columns if col not in df.columns]
            if missing_sel_cols:
                print(f"WARNING: Coach CSV is missing some of the specifically requested columns after load: {missing_sel_cols}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Coach CSV file not found at '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: Coach CSV file at '{file_path}' is empty.")
    except ValueError as ve:
        print(f"ERROR: Coach CSV column issue (e.g., selected columns not found): {ve}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading Coach CSV '{file_path}': {e}")
    return None
def load_postseason_csv(file_path, selected_columns=None):
    """
    Loads the postseason results CSV file into a pandas DataFrame.
    """
    try:
        # Specify dtype for Seed to avoid mixed type issues if some are missing
        # but pandas usually handles this fine with read_csv if NaNs are present
        df = pd.read_csv(file_path, usecols=selected_columns, keep_default_na=True, na_filter=True)
        print(f"Successfully loaded Postseason CSV: {file_path}")
        if selected_columns:
            missing_sel_cols = [col for col in selected_columns if col not in df.columns]
            if missing_sel_cols:
                print(f"WARNING: Postseason CSV is missing some of the specifically requested columns after load: {missing_sel_cols}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Postseason CSV file not found at '{file_path}'.")
    except pd.errors.EmptyDataError:
        print(f"ERROR: Postseason CSV file at '{file_path}' is empty.")
    except ValueError as ve: # Catches issues like columns in usecols not in file
        print(f"ERROR: Postseason CSV column issue: {ve}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading Postseason CSV '{file_path}': {e}")
    return None
