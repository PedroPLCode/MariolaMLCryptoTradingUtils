import argparse
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.layers import LSTM
from utils.logger_utils import initialize_logger, log
from utils.app_utils import load_data_from_csv
from utils.calc_utils import prepare_df
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
                    help="A required argument. Klines full historical data filename.csv"
                    )

args = parser.parse_args()
settings_filename = args.first_argument
data_filename = args.second_argument
base_filename = data_filename.split('.')[0]

initialize_logger(settings_filename)
log(f"MariolaCryptoTradingBot. Training process starting.\n"
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
regresion=settings_data['settings']['regresion']
clasification=settings_data['settings']['clasification']
result_marker=settings_data['settings']['result_marker']
window_size=settings_data['settings']['window_size']
lookback=settings_data['settings']['window_lookback']
test_size=settings_data['settings']['test_size']
random_state=settings_data['settings']['random_state']


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
    settings=settings
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
    result_marker=result_marker
    )
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
    result_marker=result_marker
    )
log(f"MariolaCryptoTradingBot. handle_pca completed.")


log(f"MariolaCryptoTradingBot. Splitting the data into training and testing sets.\n"
      f"starting train_test_split.\n"
      f"test_size: {test_size}\n"
      f"random_state: {random_state}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
log(f"MariolaCryptoTradingBot. train_test_split completed.")


log(f"MariolaCryptoTradingBot. Creating the LSTM model.")
model = Sequential()
log(f"MariolaCryptoTradingBot. completed.")


log(f"MariolaCryptoTradingBot. Input layer (only the last value of the sequence is returned.")
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
log(f"MariolaCryptoTradingBot. completed.")


log(f"MariolaCryptoTradingBot. Dropout to avoid overfitting.")
model.add(Dropout(0.2))
log(f"MariolaCryptoTradingBot. completed.")


log(f"MariolaCryptoTradingBot. Output layer (binary classification - predicting one label).")
model.add(Dense(units=1, activation='sigmoid'))
log(f"MariolaCryptoTradingBot. completed.")

log(f"MariolaCryptoTradingBot. Compiling the model.")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
log(f"MariolaCryptoTradingBot. completed.")


log(f"MariolaCryptoTradingBot. Training the model.")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
log(f"MariolaCryptoTradingBot. completed.")


model.summary()


keras_filename = f'{base_filename}.keras'
model.save(keras_filename)
log(f"MariolaCryptoTradingBot. Model saved as {keras_filename}")