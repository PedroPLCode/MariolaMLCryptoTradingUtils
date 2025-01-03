import sys
from time import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tensorflow.keras.models import load_model
from utils.app_utils import extract_settings_data
from utils.api_utils import get_klines
from utils.calc_utils import prepare_df
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from mariola_utils import (
    normalize_df, 
    handle_pca, 
    create_sequences
)

def main():
    
    start_time = time()

    settings_filename, model_filename = get_parsed_arguments(
        first_arg_str='Settings filename.json',
        second_arg_string='Model filename.keras'
    )

    initialize_logger(settings_filename)
    log(f"MariolaCryptoTradingBot. Prediction process starting.\n"
        f"Received filename arguments: {settings_filename} {model_filename}"
        )

    settings_data = extract_settings_data(settings_filename)

    settings=settings_data['settings']
    symbol=settings_data['settings']['symbol']
    interval=settings_data['settings']['interval']
    lookback=settings_data['settings']['lookback']
    regresion=settings_data['settings']['regresion']
    clasification=settings_data['settings']['clasification']
    result_marker=settings_data['settings']['result_marker']
    window_size=settings_data['settings']['window_size']
    window_lookback=settings_data['settings']['window_lookback']

    log(f"MariolaCryptoTradingBot. Fetch actual {symbol} {interval} data.")
    data_df = get_klines(
        symbol=symbol, 
        interval=interval, 
        lookback=lookback,
        )
    log(f"MariolaCryptoTradingBot. Fetch completed.\n"
        f"symbol: {symbol}\n"
        f"interval: {interval}\n"
        f"lookback: {lookback}\n"
        f"len(data_df): {len(data_df)}"
        )

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
        training_mode=False
        )
    log(f"MariolaCryptoTradingBot. prepare_df completed.")

    log(f"MariolaCryptoTradingBot. Normalize data."
        f"starting normalize_df."
        )
    df_normalized = normalize_df(
        result_df=result_df,
        training_mode=False,
        result_marker=result_marker
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
        f"starting create_sequences.\n"
        f"len(df_reduced): {len(df_reduced)}\n"
        f"window_size: {window_size}\n"
        f"lookback: {window_lookback}\n"
        f"result_marker: {result_marker}"
        )
    X = create_sequences(
        df_reduced=df_reduced, 
        lookback=window_lookback, 
        window_size=window_size,
        result_marker=result_marker,
        training_mode=False
        )
    log(f"MariolaCryptoTradingBot. create_sequences completed.")

    log(f"MariolaCryptoTradingBot. Load the saved model.")
    loaded_model = load_model(model_filename)
    log(f"MariolaCryptoTradingBot. Load completed.")

    log(f"MariolaCryptoTradingBot. Prediction on new data.")
    y_pred = loaded_model.predict(X)
    log(f"MariolaCryptoTradingBot. Prediction completed.")

    log(f"MariolaCryptoTradingBot. Converting the predictions to binary values (0 or 1).")
    if clasification:
        y_pred = (y_pred > 0.5)
    log(f"Predictions ({'regresion' if regresion else 'clasification'}):")
    for i, val in enumerate(y_pred[-10:]):
        log(f"Index {len(y_pred) - 10 + i}: {val}")

    end_time = time()

    log(f"MariolaCryptoTradingBot. {'Regresion' if regresion else 'Clasification'} completed.\n"
        f"Prediction based on latest data: {y_pred[-1]}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
        )
    
if __name__ == "__main__":
    main()