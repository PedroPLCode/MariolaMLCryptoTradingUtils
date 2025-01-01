import argparse
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger_utils import initialize_logger, log
from utils.app_utils import load_data_from_csv, save_data_to_csv, save_pandas_df_info
from utils.calc_utils import prepare_df

parser = argparse.ArgumentParser(description="A script that accepts two arguments.")
parser.add_argument('first_argument', 
                    type=str, 
                    help="A required argument. Settings filename.json"
                    )
parser.add_argument('second_argument', 
                    type=str, 
                    help="A required argument. Klines full historical data filename.csv"
                    )

args = parser.parse_args()
settings_filename = args.first_argument
data_filename = args.second_argument
base_filename = data_filename.split('.')[0]

initialize_logger(settings_filename)
log(f"MariolaCryptoTradingBot. Calculating DataFrame process starting.\n"
      f"Received filename arguments: {args.first_argument} {args.second_argument}"
      )


try:
    with open(settings_filename, 'r') as f:
        settings_data = json.load(f)
    log(f"Successfully loaded settings from {settings_filename}")
except FileNotFoundError:
    log(f"Error: File {settings_filename} not found.")
    exit(1)
except json.JSONDecodeError:
    log(f"Error: File {settings_filename} is not a valid JSON.")
    exit(1)
except Exception as e:
    log(f"Unexpected error loading file {settings_filename}: {e}")
    exit(1)


settings=settings_data['settings']
regresion=settings_data['settings']['regresion']
clasification=settings_data['settings']['clasification']


log(f"MariolaCryptoTradingBot. Load data from csv file.\n"
      f"starting load_data_from_csv.\n"
      f"filename: {data_filename}"
      )
data_df = load_data_from_csv(data_filename)
log(f"MariolaCryptoTradingBot. load_data_from_csv completed.")


log(f"MariolaCryptoTradingBot. Prepare DataFrame.\n"
      f"starting prepare_df.\n"
      f"regresion: {regresion}\n"
      f"clasification: {clasification}"
      )
result_df = prepare_df(
    df=data_df, 
    regresion=regresion,
    clasification=clasification,
    settings=settings,
    training=True
    )
csv_filename = data_filename.replace('_fetched', '_calculated')
info_filename = csv_filename.replace('csv', 'info')
save_data_to_csv(result_df, csv_filename)
save_pandas_df_info(result_df, info_filename)
log(f"MariolaCryptoTradingBot. prepare_df completed and result_df saved to {csv_filename}.")