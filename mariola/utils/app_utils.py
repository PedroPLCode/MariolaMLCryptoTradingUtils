import json
import pandas as pd
from utils.logger_utils import log

def save_data_to_csv(data, filename):
    """
    Saves the provided data to a CSV file.

    This function takes a pandas DataFrame and saves it to a specified file.
    It will not include the index in the saved CSV file by default.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be saved.
        filename (str): The name or path of the CSV file to save the data to.

    Returns:
        None: If the data is saved successfully, a message is printed confirming the save.
        If an error occurs during the save process, it will be caught and printed.

    Example:
        >>> save_data_to_csv(df, 'data.csv')
        Klines data saved to data.csv
    """
    
    if data is None:
        log(f"Error in save_data_to_csv. data id None.\n{data}")
        return None
        
    try:
        data.to_csv(filename, index=False)
        log(f"Klines data saved to {filename}")
        
    except Exception as e:
        log(e)
        return None


def load_data_from_csv(filename):
    """
    Loads data from a CSV file into a pandas DataFrame.

    This function reads the contents of a CSV file and loads it into a pandas DataFrame.
    It assumes the file exists and is properly formatted as a CSV.

    Parameters:
        filename (str): The name or path of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded DataFrame containing the data from the CSV file.
        If an error occurs, it will return None.

    Example:
        >>> df = load_data_from_csv('data.csv')
        Klines data loaded from data.csv
    """
    
    if not filename:
        log("Error in load_data_from_csv. Filemane not provided")
        return None
    
    try:
        df = pd.read_csv(filename)
        log(f"Klines data loaded from {filename}")
        
        return df
    
    except Exception as e:
        log(e)
        return None


def save_df_info(df, filename):
    """
    Saves basic information and the last 3 rows of a pandas DataFrame to a file.

    Args:
        df (pd.DataFrame): A pandas DataFrame whose information will be saved.
        filename (str): The name of the file where the DataFrame information will be saved.

    Returns:
        None: The function writes the information to the specified file.

    Notes:
        The function saves the following information to the file:
        - The column names of the DataFrame.
        - The number of columns in the DataFrame.
        - The last 3 rows of the DataFrame without the index.
    """
    try:
        if df is None or df.empty or not filename:
            raise ValueError("df and filename must be provided and cannot be None.")
        
        with open(filename, 'w') as f:
            f.write("Pandas DataFrame:\n")
            f.write(str(df.columns) + "\n\n")
            f.write("Number of Columns:\n")
            f.write(str(len(df.columns)) + "\n\n")
            f.write("Number of Rows:\n")
            f.write(str(len(df)) + "\n\n")
            f.write("Last 3 rows:\n")
            f.write(df.tail(3).to_string(index=False))
            log(f"DataFrame saved to {filename}")
            
    except Exception as e:
        log(e)
        return None
    
    
def extract_settings_data(settings_filename):
    """
    Extracts and returns settings data from a JSON file.

    This function reads the specified JSON file and parses its content into a Python dictionary. 
    Logs a message upon successfully loading the settings or logs an appropriate error message 
    and exits the program if an error occurs.

    Parameters:
        settings_filename (str): The path to the JSON settings file.

    Returns:
        dict: A dictionary containing the parsed settings data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON format.
        Exception: For any other unexpected errors.
    """
    try:
        with open(settings_filename, 'r') as f:
            settings_data = json.load(f)
            
        log(f"Successfully loaded settings from {settings_filename}")
        
        return settings_data
    
    except FileNotFoundError:
        log(f"Error: File {settings_filename} not found.")
        exit(1)
    except json.JSONDecodeError:
        log(f"Error: File {settings_filename} is not a valid JSON.")
        exit(1)
    except Exception as e:
        log(f"Unexpected error loading file {settings_filename}: {e}")
        exit(1)
        
        
def save_dataframe_with_info(dataframe, base_filename, stage_name):
    """
    Saves a DataFrame to a CSV file and logs information about the saved data.

    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame to be saved.
    - base_filename (str): The base filename used for saving the CSV file.
      The '_calculated' part in the filename is replaced with the `stage_name`.
    - stage_name (str): A descriptive name representing the current stage of processing.
      This is used to modify the filename (e.g., '_normalized', '_pca_analyzed').

    Saves the DataFrame to a CSV file and its associated information to an '.info' file.
    Logs the action of saving the files with the stage name included in the filename.

    Example:
    >>> save_dataframe_with_info(df, 'data_calculated.csv', 'normalized')
    Data saved to data_normalized.csv.
    """
    try:
        csv_filename = base_filename.replace('_calculated', f'_{stage_name}')
        info_filename = csv_filename.replace('csv', 'info')
        save_df_info(dataframe, info_filename)
        log(f"{stage_name.capitalize()} data saved to {csv_filename}.")
        
    except Exception as e:
        log(e)
        return None