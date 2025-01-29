"""
MariolaCryptoTradingBot - Training Random Forest Model

This script is responsible for training a Random Forest model (regression or classification) 
using prepared data. The process includes loading CSV files with data and configuration, 
preparing features, training the model, evaluating results, performing feature selection, 
and saving the trained model as a .joblib file.

Functions:
    train_rf_model() - Main function for training the Random Forest model.

Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - app_utils: Contains utility functions for extracting settings and handling app-related tasks.
    - sklearn: Provides functionality for training Random Forest model.

Usage:
    Run the script from the command line:
        python3 train_rf_model.py <settings_filename.json> <data_filename.csv>

    Example:
        python3 train_rf_model.py settings.json calculated_df.csv
        
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.plot_utils import visualise_model_performance
from utils.app_utils import (
    extract_settings_data,
    load_data_from_csv,
)

def train_rf_model():
    """
    Trains a Random Forest model using data from a CSV file and settings from a JSON file.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and data filename.
        2. Loads configuration settings from the JSON file.
        3. Loads data from the CSV file.
        4. Trains a Random Forest model using the prepared data.
        5. Evaluates the model's performance and logs the results.
        6. Selects the most important features based on the trained model.
        7. Saves the trained model to a .joblib file.

    Arguments:
        None (Relies on command-line arguments.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the CSV data file.

    Returns:
        None

    Example:
        python3 train_rf_model.py settings.json calculated_df.csv
    """

    start_time = time()

    settings_filename, data_filename = get_parsed_arguments(
        first_arg_string="Settings filename.json",
        second_arg_string="Calculated and prepared data filename.csv"
    )

    initialize_logger(settings_filename)
    log(f"Received arguments: "
        f"settings_filename={settings_filename}, "
        f"data_filename={data_filename}")

    settings_data = extract_settings_data(settings_filename)
    regression = settings_data['settings']['regression']
    classification = settings_data['settings']['classification']
    result_marker = settings_data['settings']['result_marker']
    test_size = settings_data['settings']['test_size']
    random_state = settings_data['settings']['random_state']

    log("Loading data from CSV file.")
    loaded_df = load_data_from_csv(data_filename)
    log("Data loading completed.")
    
    if result_marker not in loaded_df.columns:
        log(f"Error: result_marker '{result_marker}' not found in DataFrame.")
        raise ValueError(f"result_marker '{result_marker}' not found in DataFrame.")

    log(f"Splitting data into training and testing sets "
        f"(test_size={test_size}).")
    X = loaded_df.drop(columns=[result_marker])
    y = loaded_df[result_marker]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    log("Data splitting and scaling completed.")

    log("Creating the Random Forest model.")
    if classification:
        model = RandomForestClassifier(random_state=random_state)
    elif regression:
        model = RandomForestRegressor(random_state=random_state, n_estimators=100)
    else:
        raise ValueError("Invalid model type. Specify either 'regression' or "
                         "'classification' in settings.")
    log("Model creation completed.")

    log(f"Training the model with data.\n"
        f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    model.fit(X_train, y_train)
    log("Model training completed.")

    log("Evaluating model performance.")
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    log(f"Mean Absolute Error (MAE) = {mae:.4f}")
    log(f"Mean Squared Error (MSE) = {mse:.4f}")

    log("Calculating feature importances.")
    importances = model.feature_importances_
    features = X.columns
    sorted_indices = np.argsort(importances)[::-1]
    log("Top 10 important features:")
    for idx in sorted_indices[:10]:
        log(f"{features[idx]}: {importances[idx]:.4f}")

    log("Performing feature selection.")
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    model_filename = data_filename.replace('df_', 'model_').replace('_calculated', '_rf')\
                                   .replace('.csv', '.joblib')
    joblib.dump(model, model_filename)
    log(f"Model saved as {model_filename}")
    
    end_time = time()

    log(f"Random Forest Model training completed.\n"
        f"{'Regression' if regression else 'Classification'}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
        )

    visualise_model_performance(
        y_test, 
        y_pred, 
        result_marker, 
        regression, 
        classification
        )
    
if __name__ == "__main__":
    train_rf_model()