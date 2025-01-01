import argparse
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from keras.models import load_model
from utils.api_utils import get_klines
from utils.calc_utils import prepare_df
from utils.logger_utils import initialize_logger, log
from mariola_utils import (
    normalize_df, 
    handle_pca, 
    create_sequences
)

parser = argparse.ArgumentParser(description="A script that accepts two arguments.")
parser.add_argument('first_argument', 
                    type=str, 
                    help="A required argument. Settings filename.json"
                    )
parser.add_argument('second_argument', 
                    type=str, 
                    help="A required argument. Model filename.keras"
                    )

args = parser.parse_args()
settings_filename = args.first_argument
model_filename = args.second_argument
base_filename = model_filename.split('.')[0]

initialize_logger(settings_filename)
log(f"MariolaCryptoTradingBot. Prediction process starting.\n"
      f"Received filename arguments: {args.first_argument} {args.second_argument}"
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


settings=settings_data['settings']
symbol=settings_data['settings']['symbol']
interval=settings_data['settings']['interval']
lookback=settings_data['settings']['lookback']
regresion=settings_data['settings']['regresion']
clasification=settings_data['settings']['clasification']
result_marker=settings_data['settings']['result_marker']
window_size=settings_data['settings']['window_size']
lookback=settings_data['settings']['window_lookback']
test_size=settings_data['settings']['test_size']
random_state=settings_data['settings']['random_state']


log(f"MariolaCryptoTradingBot. Fetch actual {symbol} {interval} data.")
data_df = get_klines(
    symbol=symbol, 
    interval=interval, 
    lookback=lookback
    )
log(f"MariolaCryptoTradingBot. Fetch completed.")


log(f"MariolaCryptoTradingBot. Prepare DataFrame.\n"
      f"starting prepare_df.\n"
      f"regresion: {regresion}\n"
      f"clasification: {clasification}"
      )
result_df = prepare_df(
    df=data_df, 
    regresion=False,
    clasification=False
    )
log(f"MariolaCryptoTradingBot. prepare_df completed.")


log(f"MariolaCryptoTradingBot. Normalize data."
      f"starting normalize_df."
      )
df_normalized = normalize_df(
    result_df=result_df
    )
log(f"MariolaCryptoTradingBot. normalize_df completed.")


log(f"MariolaCryptoTradingBot. Principal Component Analysis.\n"
      f"starting handle_pca.\n"
      f"result_marker: {result_marker}"
      )
df_reduced = handle_pca(
    df_normalized=df_normalized, 
    result_df=result_df, 
    result_marker=None
    )
log(f"MariolaCryptoTradingBot. handle_pca completed.")


log(f"MariolaCryptoTradingBot. Create sequences.\n"
      f"starting normalize_df.\n"
      f"window_size: {window_size}\n"
      f"lookback: {lookback}"
      )
X, _ = create_sequences(
    df_reduced=df_reduced, 
    lookback=lookback, 
    window_size=window_size,
    result_marker=result_marker
    )
log(f"MariolaCryptoTradingBot. create_sequences completed.")


log(f"MariolaCryptoTradingBot. Load the saved model.")
loaded_model = load_model('model.keras')
log(f"MariolaCryptoTradingBot. Load completed.")


log(f"MariolaCryptoTradingBot. Prediction on new data.")
y_pred = loaded_model.predict(X)
log(f"MariolaCryptoTradingBot. Prediction completed.")


log(f"MariolaCryptoTradingBot. Converting the predictions to binary values (0 or 1).")
y_pred = (y_pred > 0.5)
log("Predictions:\n", y_pred[:10])