"""
MariolaMLCryptoTradingUtils - Predicting with XGBoost Model

This script is responsible for making predictions using a pre-trained XGBoost model.
The process includes loading settings and the model, fetching and preparing data,
scaling features, and generating predictions.

Functions:
    predict_xgboost_model() - Main function for predicting using the XGBoost model.

Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - app_utils: Contains utility functions for extracting settings and handling app-related tasks.
    - sklearn: Provides functionality for feature scaling.
    - xgboost: Provides functionality for loading and using pre-trained models.

Usage:
    Run the script from the command line:
        python3 predict_xgboost_model.py <settings_filename.json> <model_filename.model>

    Example:
        python3 predict_xgboost_model.py settings.json xgboost_model.model

Author:
    PedroMolina

Last Update:
    2025-01-27
"""

import sys
from time import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from utils.app_utils import extract_settings_data
from utils.api_utils import get_klines
from utils.df_utils import prepare_ml_df
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.plot_utils import visualise_model_prediction


def predict_xgboost_model():
    """
    Predicts using a pre-trained XGBoost model and data fetched from an API.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and model filename.
        2. Loads configuration settings from the JSON file.
        3. Fetches the latest market data based on the configuration.
        4. Prepares the fetched data for prediction.
        5. Scales the data using StandardScaler.
        6. Loads the pre-trained XGBoost model.
        7. Makes predictions using the model on the prepared data.
        8. Logs the predictions and execution time.

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the pre-trained XGBoost model file.

    Returns:
        None

    Example:
        python3 predict_xgboost_model.py settings.json xgboost_model.model
    """
    start_time = time()

    settings_filename, model_filename = get_parsed_arguments(
        first_arg_string="Settings filename.json",
        second_arg_string="Model filename.model",
    )

    initialize_logger(settings_filename)
    log(
        f"Received arguments: "
        f"settings_filename={settings_filename}, "
        f"model_filename={model_filename}"
    )

    settings_data = extract_settings_data(settings_filename)

    settings = settings_data["settings"]
    symbol = settings["symbol"]
    interval = settings["interval"]
    lookback = settings["lookback"]
    regression = settings["regression"]
    classification = settings["classification"]
    result_marker = settings["result_marker"]

    log(f"Fetch actual {symbol} {interval} data.")
    fetched_df = get_klines(symbol=symbol, interval=interval, lookback=lookback)
    log(f"Fetch completed.")

    log(f"Prepare DataFrame.")
    calculated_df = prepare_ml_df(
        df=fetched_df,
        regression=regression,
        classification=classification,
        settings=settings,
        training_mode=False,
    )
    log(f"prepare_df completed.")

    log(f"Preparing features and target variable.")
    if result_marker not in calculated_df.columns:
        X = calculated_df.fillna(0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"Data preparation completed.")

    log(f"Scaling the data using StandardScaler.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log(f"Data scaling completed.")

    log(f"Load the saved XGBoost model.")
    model = xgb.Booster()
    model.load_model(model_filename)
    log(f"Load completed.")

    log(f"Prediction on new data.")
    if isinstance(X_scaled, tuple):
        X_scaled = X_scaled[0]
    X_scaled = X_scaled.reshape(X_scaled.shape[0], -1)
    dmatrix = xgb.DMatrix(X_scaled)
    y_pred = model.predict(dmatrix)
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
    predict_xgboost_model()
