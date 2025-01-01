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


def save_pandas_df_info(df, filename):
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
            f.write("Len Columns:\n")
            f.write(str(len(df.columns)) + "\n\n")
            f.write("Last 3 rows:\n")
            f.write(df.tail(3).to_string(index=False))
            log(f"DataFrame saved to {filename}")
            
    except Exception as e:
        log(e)
        return None