"""
MariolaMLCryptoTradingUtils - Prediction Using Trained Random Forest Model

This script is responsible for making predictions using a pre-trained Random Forest model 
on new data. The process includes loading the trained model from a `.joblib` file, fetching 
the new data, preprocessing the features, performing predictions, and logging the results.

Functions:
    predict_with_rf_model() - Main function for predicting using the trained Random Forest model.

Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - app_utils: Contains utility functions for extracting settings and handling app-related tasks.
    - sklearn: Provides functionality for for loading the pre-trained Random Forest model.

Usage:
    Run the script from the command line:
        python3 predict_rf_model.py <settings_filename.json> <model_filename.joblib>

    Example:
        python3 predict_rf_model.py settings.json model_rf.joblib
        
Author:
    PedroMolina

Last Update:
    2025-01-25
"""
import sys
import joblib
import numpy as np
from time import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.preprocessing import StandardScaler
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.api_utils import get_klines
from utils.df_utils import prepare_ml_df
from utils.app_utils import extract_settings_data
from utils.plot_utils import visualise_model_prediction

def predict_with_rf_model():
    """
    Makes predictions using a pre-trained Random Forest model.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename, data filename, 
           and the model filename.
        2. Loads configuration settings from the JSON file.
        3. Loads new data from the CSV file and prepares the features.
        4. Loads the pre-trained Random Forest model from a .joblib file.
        5. Makes predictions on the new data.
        6. Evaluates the predictions and logs the results.
        7. Optionally, visualizes the predictions.

    Arguments:
        None (Relies on command-line arguments.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the pre-trained model (.joblib file).

    Returns:
        None

    Example:
        python3 predict_rf_model.py settings.json new_data.csv model_rf.joblib
    """

    start_time = time()

    settings_filename, model_filename = get_parsed_arguments(
        first_arg_string="Settings filename.json",
        second_arg_string="Trained model filename.joblib"
    )

    initialize_logger(settings_filename)
    log(f"Received arguments: "
        f"settings_filename={settings_filename}, "
        f"model_filename={model_filename}")

    settings_data = extract_settings_data(settings_filename)
    settings = settings_data['settings']
    symbol = settings['symbol']
    interval = settings['interval']
    lookback = settings['lookback']
    result_marker = settings['result_marker']
    regression = settings['regression']
    classification = settings['classification']

    log(f"Fetching actual {symbol} {interval} data.")
    fetched_df = get_klines(symbol=symbol, interval=interval, lookback=lookback)
    log(f"Data fetched. "
        f"Symbol: {symbol}, Interval: {interval}, Lookback: {lookback}, Data length: {len(fetched_df)}")

    log(f"Preparing DataFrame for prediction.")
    calculated_df = prepare_ml_df(
        df=fetched_df, 
        regression=regression,
        classification=classification,
        settings=settings,
        training_mode=False
    )
    log(f"Data preparation completed.")

    log(f"Loading pre-trained model from {model_filename}.")
    model = joblib.load(model_filename)
    log("Model loaded successfully.")

    log(f"Making predictions on new data.")
    if result_marker not in calculated_df.columns:
        X_new = calculated_df
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X_new)
    y_pred = model.predict(X_new)
    
    log(f"Predictions made. First 10 predictions:")
    for idx, pred in enumerate(y_pred[:10]):
        log(f"Prediction {idx + 1}: {pred:.4f}")
        
    end_time = time()
    
    log(f"{'Regression' if regression else 'Classification'} completed.\n"
        f"Prediction based on latest data: {y_pred[-1]}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
        )
    
    visualise_model_prediction(y_pred)

if __name__ == "__main__":
    predict_with_rf_model()