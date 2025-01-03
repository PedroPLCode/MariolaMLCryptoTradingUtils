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

def mariola_fetch():
    
    start_time = time()

    settings_filename, dry_run = get_parsed_arguments(
        first_arg_str='Settings filename.json',
        second_arg_string='Dry run mode (yes/no)'
    )
    dry_run = (dry_run == 'yes')

    if dry_run:
        log("Dry run mode enabled. No data will be fetched or saved.")

    initialize_logger(settings_filename)
    log(f"MariolaCryptoTradingBot. Fetch starting.\n"
        f"Received filename argument: {settings_filename}"
        )

    settings_data = extract_settings_data(settings_filename)
    fetch_sequence = settings_data['fetch_sequence']
    total_steps = len(fetch_sequence)
    log(f"Total steps to fetch data: {total_steps}")

    log(f"MariolaCryptoTradingBot. Fetch and save all data according to fetch sequence.\n"
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
            
            log(f"MariolaCryptoTradingBot. Fetching data.\n"
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

            csv_filename = f"data/df_{step_name}_fetched.csv"
            info_filename = csv_filename.replace('csv', 'info')
            save_data_to_csv(historical_klines, csv_filename)
            save_df_info(historical_klines, info_filename)
            log(f"MariolaCryptoTradingBot. historical_klines saved to {csv_filename}.")
            
            log(f"MariolaCryptoTradingBot. Fetch step completed.\n"
                f"Step {i}/{total_steps} - step_name: {step_name}\n"
                f"symbol: {symbol}\n"
                f"interval: {interval}"
                )
        
    end_time = time()

    log(f"MariolaCryptoTradingBot Fetching historical data completed.\n"
        f"Total steps: {total_steps}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
    )
    
if __name__ == "__main__":
    mariola_fetch()