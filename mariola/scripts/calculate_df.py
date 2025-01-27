"""
MariolaCryptoTradingBot - DataFrame Calculation

This script provides functionality for calculating technical analysis parameters 
from historical cryptocurrency data fetched via Binance API. 
The script processes a CSV file with historical klines data and saves 
the processed DataFrame with additional parameters based on user-defined settings.

Functions:
    calculate_df() - Main function for loading, processing, and saving DataFrame data.
    
Requirements:
    - logger_utils: Functions for initializing and using a logger.
    - parser_utils: Functions for parsing command-line arguments.
    - df_utils: Function for preparing DataFrame (prepare_df).
    - app_utils: Helper functions for extracting settings and handling CSV files.

Usage:
    Run the script with two arguments:
    1. JSON settings file containing configuration for regression and classification.
    2. CSV file with historical klines data to process.
    
    Example:
        python calculate_df.py settings.json historical_data.csv

Author:
    PedroMolina

Last Update:
    2025-01-25
"""

import sys
from time import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger_utils import initialize_logger, log
from utils.parser_utils import get_parsed_arguments
from utils.df_utils import prepare_df
from utils.app_utils import (
    extract_settings_data, 
    load_data_from_csv,
    save_data_to_csv, 
    save_df_info
)

def calculate_df():
    """
    Main function to calculate technical analysis parameters for historical cryptocurrency data.

    The function performs the following steps:
        1. Parse command-line arguments to get the settings file and data file.
        2. Initialize the logger using the settings file.
        3. Extract settings data from the provided JSON settings file, including configuration for regression and classification.
        4. Load historical klines data from the specified CSV file.
        5. Process the data using the `prepare_df` function to calculate regression or classification parameters based on the settings.
        6. Save the processed DataFrame to a new CSV file and generate an info file.
        7. Log the completion of the process and the total time taken.

    Arguments:
        None (Relies on command-line arguments for settings and data files.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the CSV file containing historical klines data.

    Returns:
        None. This function saves the calculated DataFrame to a CSV file and generates an info file.

    Raises:
        Exception: If any step in the data processing pipeline fails, the error is logged.

    Example:
        python calculate_df.py settings.json historical_data.csv
    """
    start_time = time()

    settings_filename, data_filename = get_parsed_arguments(
        first_arg_string='Settings filename.json',
        second_arg_string='Klines full historical data filename.csv'
    )

    initialize_logger(settings_filename)
    log(f"Calculating DataFrame process starting.\n"
        f"Received filename arguments: {settings_filename} {data_filename}"
        )

    settings_data = extract_settings_data(settings_filename)
    settings = settings_data['settings']
    regression = settings_data['settings']['regression']
    classification = settings_data['settings']['classification']

    log(f"Load data from csv file.\n"
        f"Starting load_data_from_csv.\n"
        f"Filename: {data_filename}"
        )
    data_df = load_data_from_csv(data_filename)
    log(f"load_data_from_csv completed.")

    log(f"Prepare DataFrame.\n"
        f"Starting prepare_df.\n"
        f"Regression: {regression}\n"
        f"Classification: {classification}"
        )
    result_df = prepare_df(
        df=data_df, 
        regression=regression,
        classification=classification,
        settings=settings,
        training_mode=True
    )

    csv_filename = data_filename.replace('_fetched', '_calculated')
    info_filename = csv_filename.replace('csv', 'info')
    save_data_to_csv(result_df, csv_filename)
    save_df_info(result_df, info_filename)

    end_time = time()
    log(f"Calculating Technical Analysis parameters completed.\n"
        f"prepare_df completed and result_df saved to {csv_filename}.\n"
        f"{'Regression' if regression else 'Classification'}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
        )

if __name__ == "__main__":
    calculate_df()