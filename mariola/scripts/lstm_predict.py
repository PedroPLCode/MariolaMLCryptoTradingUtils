"""
MariolaMLCryptoTradingUtils - Predicting with LSTM Model

This script is responsible for making predictions using a pre-trained LSTM model.
It takes settings and model filenames as inputs, fetches data from Binance, processes it
by normalizing, performing PCA, and creating sequences, and then makes predictions based
on the trained LSTM model. The predictions can be for either regression or classification
tasks, depending on the settings. The result is logged throughout the process.

Functions:
    predict_lstm_model() - Main function for making predictions with the LSTM model.

Requirements:
    - app_utils: Provides functions for extracting settings data.
    - api_utils: Provides function to fetch kline data from Binance.
    - df_utils: Contains functions for preparing data frames.
    - parser_utils: Provides command-line argument parsing functions.
    - logger_utils: Handles logging and initialization of log messages.
    - ml_utils: Provides functions for normalizing data, handling PCA, and creating sequences.
    - tensorflow.keras.models: Provides functionality for loading the pre-trained LSTM model.

Usage:
    Run the script from the command line:
        python3 predict_lstm_model.py <settings_filename.json> <model_filename.keras>

    Example:
        python3 predict_lstm_model.py settings.json trained_model.keras

Author:
    PedroMolina

Last Update:
    2025-01-25
"""

import sys
from time import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tensorflow.keras.models import load_model
from utils.app_utils import extract_settings_data
from utils.api_utils import get_klines
from utils.df_utils import prepare_ml_df
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.plot_utils import visualise_model_prediction
from utils.ml_utils import normalize_df, handle_pca, create_sequences


def predict_lstm_model():
    """
    Makes predictions using a pre-trained LSTM model based on the latest data from Binance.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and model filename.
        2. Initializes logging and loads the settings from the JSON file.
        3. Fetches the latest data from Binance based on the provided symbol, interval, and lookback.
        4. Prepares the data for prediction by normalizing and performing PCA.
        5. Creates sequences of the data for input into the LSTM model.
        6. Loads the pre-trained LSTM model from the specified file.
        7. Makes predictions based on the prepared input data.
        8. Converts the predictions to binary values if the task is classification.
        9. Logs the results, including the latest prediction and the time taken for the process.

    Arguments:
        None (Relies on command-line arguments.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the Keras model file (.keras).

    Returns:
        None

    Example:
        python3 predict_lstm_model.py settings.json trained_model.keras
    """

    start_time = time()

    settings_filename, model_filename = get_parsed_arguments(
        first_arg_string="Settings filename.json",
        second_arg_string="Model filename.keras",
    )

    initialize_logger(settings_filename)
    log(
        f"Received arguments: "
        f"settings_filename={settings_filename}, "
        f"model_filenama={model_filename}"
    )

    settings_data = extract_settings_data(settings_filename)
    settings = settings_data["settings"]
    symbol = settings_data["settings"]["symbol"]
    interval = settings_data["settings"]["interval"]
    lookback = settings_data["settings"]["lookback"]
    regression = settings_data["settings"]["regression"]
    classification = settings_data["settings"]["classification"]
    result_marker = settings_data["settings"]["result_marker"]
    window_size = settings_data["settings"]["window_size"]
    window_lookback = settings_data["settings"]["window_lookback"]

    log(f"Fetch actual {symbol} {interval} data.")
    fetched_df = get_klines(
        symbol=symbol,
        interval=interval,
        lookback=lookback,
    )
    log(
        f"Fetch completed.\n"
        f"symbol: {symbol}\n"
        f"interval: {interval}\n"
        f"lookback: {lookback}\n"
        f"len(data_df): {len(fetched_df)}"
    )

    log(
        f"Prepare DataFrame.\n"
        f"starting prepare_df.\n"
        f"regression: {regression}\n"
        f"classification: {classification}"
    )
    calculated_df = prepare_ml_df(
        df=fetched_df,
        regression=regression,
        classification=classification,
        settings=settings,
        training_mode=False,
    )
    log(f"prepare_df completed.")

    log(f"Normalize data." f"starting normalize_df.")
    df_normalized = normalize_df(
        df=calculated_df, training_mode=False, result_marker=result_marker
    )
    log(f"normalize_df completed.")

    log(
        f"Principal Component Analysis.\n"
        f"starting handle_pca.\n"
        f"result_marker: {result_marker}"
    )
    df_reduced = handle_pca(
        df_normalized=df_normalized, loaded_df=calculated_df, result_marker=None
    )
    log(f"handle_pca completed.")

    log(
        f"Create sequences.\n"
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
        training_mode=False,
    )
    log(f"create_sequences completed.")

    log(f"Load the saved model.")
    loaded_model = load_model(model_filename)
    log(f"Load completed.")

    log(f"Prediction on new data.")
    y_pred = loaded_model.predict(X)
    log(f"Prediction completed.")

    log(f"Converting the predictions to binary values (0 or 1).")
    if classification:
        y_pred = y_pred > 0.5
    log(f"Predictions ({'regression' if regression else 'classification'}):")
    for i, val in enumerate(y_pred[-10:]):
        log(f"Index {len(y_pred) - 10 + i}: {val}")

    end_time = time()

    log(
        f"{'regression' if regression else 'classification'} completed.\n"
        f"Prediction based on latest data: {y_pred[-1]}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
    )

    visualise_model_prediction(y_pred)


if __name__ == "__main__":
    predict_lstm_model()
