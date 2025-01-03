import sys
from time import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.parser_utils import get_parsed_arguments
from utils.logger_utils import initialize_logger, log
from utils.app_utils import (
    extract_settings_data, 
    load_data_from_csv,
    save_pandas_df_info
)
from mariola_utils import (
    normalize_df, 
    handle_pca, 
    create_sequences
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

def mariola_train():
    
    start_time = time()

    settings_filename, data_filename = get_parsed_arguments(
        first_arg_str='Settings filename.json',
        second_arg_string='Calculated and prepared data filename.csv'
    )

    initialize_logger(settings_filename)
    log(f"MariolaCryptoTradingBot. Training process starting.\n"
        f"Received filename arguments: {settings_filename} {data_filename}"
        )

    settings_data = extract_settings_data(settings_filename)

    result_marker=settings_data['settings']['result_marker']
    window_size=settings_data['settings']['window_size']
    lookback=settings_data['settings']['window_lookback']
    test_size=settings_data['settings']['test_size']
    random_state=settings_data['settings']['random_state']

    log(f"MariolaCryptoTradingBot. Load data from csv file.\n"
        f"starting load_data_from_csv.\n"
        f"filename: {data_filename}"
        )
    result_df = load_data_from_csv(data_filename)
    log(f"MariolaCryptoTradingBot. load_data_from_csv completed.")

    log(f"MariolaCryptoTradingBot. Normalize data."
        f"starting normalize_df."
        )
    df_normalized = normalize_df(
        result_df=result_df
        )
    csv_filename = data_filename.replace('_calculated', '_normalized')
    info_filename = csv_filename.replace('csv', 'info')
    save_pandas_df_info(df_normalized, info_filename)
    log(f"MariolaCryptoTradingBot. normalize_df completed.")

    log(f"MariolaCryptoTradingBot. Principal Component Analysis.\n"
        f"starting handle_pca.\n"
        f"result_marker: {result_marker}"
        )
    df_reduced = handle_pca(
        df_normalized=df_normalized, 
        result_df=result_df, 
        result_marker=result_marker
        )
    csv_filename = csv_filename.replace('_normalized', '_pca_analyzed')
    info_filename = csv_filename.replace('csv', 'info')
    save_pandas_df_info(df_normalized, info_filename)
    log(f"MariolaCryptoTradingBot. handle_pca completed.")

    log(f"MariolaCryptoTradingBot. Create sequences.\n"
        f"starting normalize_df.\n"
        f"window_size: {window_size}\n"
        f"lookback: {lookback}"
        )
    X, y = create_sequences(
        df_reduced=df_reduced, 
        lookback=lookback, 
        window_size=window_size,
        result_marker=result_marker,
        training_mode=True
        )
    csv_filename = csv_filename.replace('_pca_analyzed', '_sequenced')
    info_filename = csv_filename.replace('csv', 'info')
    save_pandas_df_info(df_normalized, info_filename)
    log(f"MariolaCryptoTradingBot. create_sequences completed.")

    log(f"MariolaCryptoTradingBot. Splitting the data into training and testing sets.\n"
        f"starting train_test_split.\n"
        f"test_size: {test_size}\n"
        f"random_state: {random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
        )
    log(f"MariolaCryptoTradingBot. train_test_split completed.")

    log(f"MariolaCryptoTradingBot. Creating the LSTM model.")
    model = Sequential()
    log(f"MariolaCryptoTradingBot. Creating completed.")

    log(f"MariolaCryptoTradingBot. Input layer (only the last value of the sequence is returned.")
    model.add(
        LSTM(
            units=64, 
            return_sequences=False, 
            input_shape=(
                X_train.shape[1], 
                X_train.shape[2]
                )
            )
        )
    log(f"MariolaCryptoTradingBot. Layers completed.")

    log(f"MariolaCryptoTradingBot. Dropout to avoid overfitting.")
    model.add(
        Dropout(
            0.2
            )
        )
    log(f"MariolaCryptoTradingBot. Dropout completed.")

    log(f"MariolaCryptoTradingBot. Output layer (binary classification - predicting one label).")
    model.add(
        Dense(
            units=1, 
            activation='sigmoid'
            )
        )
    log(f"MariolaCryptoTradingBot. Layers completed.")

    log(f"MariolaCryptoTradingBot. Compiling the model.")
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
        )
    log(f"MariolaCryptoTradingBot. Compiling completed.")

    log(f"MariolaCryptoTradingBot. Training the model.")
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    model.fit(
        X_train, 
        y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(
            X_test, 
            y_test
            ),
        callbacks=[early_stopping]
        )
    log(f"MariolaCryptoTradingBot. Training completed.")

    model.summary()
    
    log("MariolaCryptoTradingBot. Evaluating the model on test data.")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    model_filename = csv_filename.replace('df_', 'model_').replace('_sequenced', '_lstm').replace('csv', 'keras')
    model.save(model_filename)
    log(f"MariolaCryptoTradingBot. Model saved as {model_filename}")

    end_time = time()

    log(f"MariolaCryptoTradingBot. LSTM Model training completed"
        f"Time taken: {end_time - start_time:.2f} seconds"
        )
    
if __name__ == "__main__":
    mariola_train()