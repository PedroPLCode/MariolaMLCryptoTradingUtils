"""
MariolaMLCryptoTradingUtils - Training LSTM Model

This script is responsible for training an LSTM model (regression or classification)
using prepared data. The process includes loading CSV files with data and configuration,
normalizing the data, performing PCA (Principal Component Analysis), creating sequences,
splitting the data into training and testing sets, training the LSTM model, and saving the
trained model as a .keras file.

Functions:
    train_lstm_model() - Main function for training the LSTM model.

Requirements:
    - parser_utils: Provides functionality for parsing command-line arguments.
    - logger_utils: Handles initialization and logging of messages.
    - app_utils: Contains utility functions for extracting settings, loading data,
      and handling app-related tasks.
    - ml_utils: Provides functions for normalizing data, handling PCA, and creating sequences.
    - tensorflow.keras: Provides functionality for preparing and training LSTM model.

Usage:
    Run the script from the command line:
        python3 train_lstm_model.py <settings_filename.json> <data_filename.csv>

    Example:
        python3 train_lstm_model.py settings.json calculated_df.csv

Author:
    PedroMolina

Last Update:
    2025-01-25
"""

import sys
from time import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.plot_utils import visualise_model_performance
from utils.app_utils import extract_settings_data, load_data_from_csv, save_df_info
from utils.ml_utils import normalize_df, handle_pca, create_sequences


def train_lstm_model():
    """
    Trains an LSTM model using data from a CSV file and settings from a JSON file.

    The function performs the following steps:
        1. Parses command-line arguments to retrieve the settings filename and data filename.
        2. Initializes logging and loads the settings from the JSON file.
        3. Loads the data from the CSV file and normalizes it.
        4. Performs Principal Component Analysis (PCA) on the normalized data.
        5. Creates sequences of the data for training.
        6. Splits the data into training and testing sets.
        7. Defines the LSTM model architecture, including input, dropout, and output layers.
        8. Compiles the model with appropriate loss function and optimizer.
        9. Trains the model on the prepared data, using early stopping to avoid overfitting.
        10. Evaluates the model on the test data and logs the results.
        11. Saves the trained model as a .keras file.

    Arguments:
        None (Relies on command-line arguments.)

    Command-line Arguments:
        - First argument: The path to the JSON settings file.
        - Second argument: The path to the CSV data file.

    Returns:
        None

    Example:
        python3 train_lstm_model.py settings.json calculated_df.csv
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
    window_size = settings_data["settings"]["window_size"]
    lookback = settings_data["settings"]["window_lookback"]
    test_size = settings_data["settings"]["test_size"]
    random_state = settings_data["settings"]["random_state"]

    log(
        f"Load data from csv file.\n"
        f"starting load_data_from_csv.\n"
        f"filename: {data_filename}"
    )
    loaded_df = load_data_from_csv(data_filename)
    log(f"load_data_from_csv completed.")

    log(f"Normalize data." f"starting normalize_df.")
    df_normalized = normalize_df(df=loaded_df, result_marker=result_marker)
    csv_filename = data_filename.replace("_calculated", "_normalized")
    info_filename = csv_filename.replace("csv", "info")
    save_df_info(df_normalized, info_filename)
    log(f"normalize_df completed.")

    log(
        f"Principal Component Analysis.\n"
        f"starting handle_pca.\n"
        f"result_marker: {result_marker}"
    )
    df_reduced = handle_pca(
        df_normalized=df_normalized, loaded_df=loaded_df, result_marker=result_marker
    )
    csv_filename = csv_filename.replace("_normalized", "_pca_analyzed")
    info_filename = csv_filename.replace("csv", "info")
    save_df_info(df_reduced, info_filename)
    log(f"handle_pca completed.")

    log(
        f"Create sequences.\n"
        f"starting normalize_df.\n"
        f"window_size: {window_size}\n"
        f"lookback: {lookback}"
    )
    X, y = create_sequences(
        df_reduced=df_reduced,
        lookback=lookback,
        window_size=window_size,
        result_marker=result_marker,
        training_mode=True,
    )
    csv_filename = csv_filename.replace("_pca_analyzed", "_sequenced")
    info_filename = csv_filename.replace("csv", "info")
    log(f"create_sequences completed.")

    log(
        f"Splitting the data into training and testing sets.\n"
        f"starting train_test_split.\n"
        f"test_size: {test_size}\n"
        f"random_state: {random_state}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log(f"train_test_split completed.")

    log(f"Creating the LSTM model.")
    model = Sequential()
    log(f"Creating completed.")

    log(f"Input layer (only the last value of the sequence is returned).")
    model.add(
        LSTM(
            units=64,
            return_sequences=False,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    log(f"Layers completed.")

    log(f"Dropout to avoid overfitting.")
    model.add(Dropout(0.2))
    log(f"Dropout completed.")

    log(f"Output layer (binary classification - predicting one label).")
    if classification:
        model.add(Dense(units=1, activation="sigmoid"))
    elif regression:
        model.add(Dense(1))
    log(f"Layers completed.")

    log(f"Compiling the model.")
    if classification:
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    elif regression:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    log(f"Compiling completed.")

    log(
        f"Training the model.\n"
        f"X_train shape: {X_train.shape}\n"
        f"y_train shape: {y_train.shape}"
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
    )
    log(f"Training completed.")

    model.summary()

    log("Evaluating the model on test data.")
    y_pred = model.predict(X_test)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model_filename = (
        csv_filename.replace("df_", "model_")
        .replace("_sequenced", "_lstm")
        .replace("csv", "keras")
    )
    model.save(model_filename)
    log(f"Model saved as {model_filename}")

    end_time = time()

    log(
        f"LSTM Model training completed.\n"
        f"{'regression' if regression else 'classification'}\n"
        f"Time taken: {end_time - start_time:.2f} seconds"
    )

    visualise_model_performance(
        y_test, y_pred, result_marker, regression, classification
    )


if __name__ == "__main__":
    train_lstm_model()
