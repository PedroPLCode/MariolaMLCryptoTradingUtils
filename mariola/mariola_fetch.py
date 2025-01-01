import argparse
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger_utils import initialize_logger, log
from utils.api_utils import get_full_historical_klines
from utils.app_utils import save_data_to_csv, save_pandas_df_info

parser = argparse.ArgumentParser(description="A script that accepts one argument.")
parser.add_argument('argument', 
                    type=str, 
                    help="A required argument. Settings filename.json"
                    )
args = parser.parse_args()
settings_filename = args.argument

initialize_logger(settings_filename)
log(f"MariolaCryptoTradingBot. Fetch starting.\n"
    f"Received filename argument: {settings_filename}"
    )

try:
    with open(settings_filename, 'r') as f:
        settings_data = json.load(f)
    log(f"Successfully loaded settings from {args.argument}")
except FileNotFoundError:
    log(f"Error: File {settings_filename} not found.")
    exit(1)
except json.JSONDecodeError:
    log(f"Error: File {settings_filename} is not a valid JSON.")
    exit(1)
except Exception as e:
    log(f"Unexpected error loading file {settings_filename}: {e}")
    exit(1)

fetch_sequence = settings_data['fetch_sequence']

total_steps = len(fetch_sequence)
log(f"Total steps to fetch data: {total_steps}")

log(f"MariolaCryptoTradingBot. Fetch and save all data according to fetch sequence.\n"
    f"Starting sequence."
    )

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
    
    historical_klines = get_full_historical_klines(
        symbol=symbol, 
        interval=interval, 
        start_str=start_str
        )

    csv_filename = f"data/df_{step_name}_fetched.csv"
    info_filename = csv_filename.replace('csv', 'info')
    save_data_to_csv(historical_klines, csv_filename)
    save_pandas_df_info(historical_klines, info_filename)
    log(f"MariolaCryptoTradingBot. historical_klines saved to {csv_filename}.")
    
    log(f"MariolaCryptoTradingBot. Fetch step completed.\n"
        f"Step {i}/{total_steps} - step_name: {step_name}\n"
        f"symbol: {symbol}\n"
        f"interval: {interval}"
        )
    
log(f"MariolaCryptoTradingBot. All fetch steps in sequence completed.")