"""
MariolaMLCryptoTradingUtils - Training XGBoost Model

This script is responsible for training an XGBoost model (regression or classification)
using prepared data. The process includes loading CSV files with data and configuration,
preparing features, training the model, evaluating results, and saving the trained model.

Functions:
    train_xgboost_model() - Main function for training the XGBoost model.

Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - app_utils: Contains utility functions for extracting settings and handling app-related tasks.
    - sklearn: Provides preprocessing and evaluation functionality.
    - xgboost: Provides XGBoost regression and classification capabilities.

Usage:
    Run the script from the command line:
        python3 train_xgboost_model.py <settings_filename.json> <data_filename.csv>

    Example:
        python3 train_xgboost_model.py settings.json calculated_df.csv

Author:
    PedroMolina

Last Update:
    2025-01-25
"""

import sys
from time import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.plot_utils import visualise_model_performance
from utils.app_utils import extract_settings_data, load_data_from_csv


def train_xgboost_model():
    """
    Trains an XGBoost regression or classification model using data from a CSV file
    and settings from a JSON file.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and data filename.
        2. Loads configuration settings from the JSON file.
        3. Loads data from the CSV file.
        4. Prepares features and target variables.
        5. Splits data into training and testing sets.
        6. Handles missing or infinite values in data.
        7. Scales the data using StandardScaler.
        8. Creates and trains an XGBoost model.
        9. Evaluates the model's performance and logs the results.
       10. Saves the trained model to a .model file.

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the CSV data file.

    Returns:
        None

    Example:
        python3 train_xgboost_model.py settings.json calculated_df.csv
    """
    start_time = time()

    settings_filename, data_filename = get_parsed_arguments(
        first_arg_string="Settings filename.json",
        second_arg_string="Calculated and prepared data filename.csv",
    )

    initialize_logger(settings_filename)
    log(
        f"Received arguments: "
        f"settings_filename={settings_filename}, "
        f"data_filename={data_filename}"
    )

    settings_data = extract_settings_data(settings_filename)
    regression = settings_data["settings"]["regression"]
    classification = settings_data["settings"]["classification"]
    result_marker = settings_data["settings"]["result_marker"]
    test_size = settings_data["settings"]["test_size"]
    random_state = settings_data["settings"]["random_state"]

    log(
        f"Loading data from CSV file.\n"
        f"Starting load_data_from_csv.\n"
        f"Filename: {data_filename}"
    )
    loaded_df = load_data_from_csv(data_filename)
    log(f"Data loading completed.")

    if result_marker not in loaded_df.columns:
        log(f"Error: result_marker '{result_marker}' not found in DataFrame.")
        raise ValueError(f"result_marker '{result_marker}' not found in DataFrame.")

    log(f"Preparing features and target variable.")
    X = loaded_df.drop(columns=result_marker)
    y = loaded_df[result_marker]
    X = X.fillna(0)
    log(f"Data preparation completed.")

    log(f"Splitting the data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log(f"Data split completed.")

    log(f"Handle missing or infinite values in training and testing sets.")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"Completed.")

    log(f"Scaling the data using StandardScaler.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log(f"Data scaling completed.")

    log(f"Creating the XGBoost model.")

    if classification:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_estimators=2000,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
        )
    elif regression:
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=2000,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
        )
    log(f"Model created.")

    log(
        f"Training the model.\n"
        f"Shape of X_train: {X_train_scaled.shape}\n"
        f"Shape of y_train: {y_train.shape}"
    )
    model.fit(X_train_scaled, y_train)
    log(f"Training completed.")

    log(f"Evaluating the model on test data.")
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log(f"Mean Squared Error: {mse}")
    log(f"R-squared: {r2}")

    model_filename = (
        data_filename.replace("df_", "model_")
        .replace("_calculated", "_xgboost")
        .replace("csv", "model")
    )
    model.save_model(model_filename)
    log(f"Model saved as {model_filename}")

    end_time = time()

    log(
        f"XGBoost Model training completed.\n"
        f"{'Regression' if regression else 'Classification'}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
    )

    visualise_model_performance(
        y_test, y_pred, result_marker, regression, classification
    )


if __name__ == "__main__":
    train_xgboost_model()
