"""
MariolaCryptoTradingBot - Fetching Historical Data

This script is responsible for fetching historical cryptocurrency data from an API, 
saving it as CSV files, and logging the process. The script uses a settings file to define 
a sequence of data fetch steps and supports a dry run mode for testing.

Functions:
    fetch_data() - Main function for fetching historical cryptocurrency data.
    
Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - api_utils: Contains functions for interacting with the API and fetching data.
    - app_utils: Provides utility functions for extracting settings, saving data, and generating info files.

Usage:
    Run the script from the command line:
        python fetch_data.py <settings_filename.json> <dry_run_mode>

    Example:
        python fetch_data.py settings.json no
        
Author:
    PedroMolina

Last Update:
    2025-01-25
"""

import sys
from time import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.api_utils import get_full_historical_klines
from utils.app_utils import (
    extract_settings_data, 
    save_data_to_csv, 
    save_df_info
)

def fetch_data():
    """
    Fetches historical cryptocurrency data based on settings from a JSON file.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and dry run mode.
        2. Initializes logging and extracts settings from the JSON file.
        3. Iterates through the fetch sequence defined in the settings file.
        4. Fetches historical data for each step and saves it as a CSV file.
        5. Logs the progress and handles errors gracefully.

    Arguments:
        None (Relies on command-line arguments for settings filename and dry run mode.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: Dry run mode (yes/no). If "yes", no data will be fetched or saved.

    Returns:
        None. This function logs the fetching process and saves data to CSV files.

    Example:
        python fetch_data.py settings.json no
    """
    start_time = time()

    settings_filename, dry_run = get_parsed_arguments(
        first_arg_string='Settings filename.json',
        second_arg_string='Dry run mode (yes/no)'
    )
    dry_run = (dry_run == 'yes')

    if dry_run:
        log("Dry run mode enabled. No data will be fetched or saved.")

    initialize_logger(settings_filename)
    log(f"Received arguments: "
        f"settings_filename={settings_filename}")

    settings_data = extract_settings_data(settings_filename)
    fetch_sequence = settings_data['fetch_sequence']
    total_steps = len(fetch_sequence)
    log(f"Total steps to fetch data: {total_steps}")

    log(f"Fetch and save data.\n"
        f"Starting sequence."
    )

    if dry_run:
        log("Dry run mode enabled. Settings file checked.")
        
    else:
        for i, (key, value) in enumerate(fetch_sequence.items(), start=1):
            step_name = key
            symbol = str(value['symbol'])
            interval = str(value['interval'])
            start_str = str(value['start_str'])
            
            log(f"Fetching data.\n"
                f"Step {i}/{total_steps} - step_name: {step_name}\n"
                f"symbol: {symbol}\n"
                f"interval: {interval}"
            )
            
            try:
                historical_klines = get_full_historical_klines(
                    symbol=symbol, 
                    interval=interval, 
                    start_str=start_str
                )
            except Exception as e:
                log(f"Error during data fetching for step {step_name}: {e}")
                continue

            csv_filename = f"mariola/data/df_{step_name}_fetched.csv"
            info_filename = csv_filename.replace('csv', 'info')
            save_data_to_csv(historical_klines, csv_filename)
            save_df_info(historical_klines, info_filename)
            log(f"historical_klines saved to {csv_filename}.")
            
            log(f"Fetch step completed.\n"
                f"Step {i}/{total_steps} - step_name: {step_name}\n"
                f"symbol: {symbol}\n"
                f"interval: {interval}"
            )
        
    end_time = time()

    log(f"Fetching historical data completed.\n"
        f"Total steps: {total_steps}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
    )
    
if __name__ == "__main__":
    fetch_data()