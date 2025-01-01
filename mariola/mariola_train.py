from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.layers import LSTM
from app_utils import load_data_from_csv
from api_utils import fetch_data
from calc_utils import prepare_df
from mariola_utils import (
    normalize_df, 
    handle_pca, 
    create_sequences
)

# Load data
data_df = load_data_from_csv()

# Prepare df
result_df = prepare_df(
    df=data_df, 
    regresion=True,
    clasification=True
    )

# Normalize data
df_normalized = normalize_df(
    result_df=result_df
    )

# Principal Component Analysis
result_marker = 'marker_close_trade_success_in_next_14_periods'
df_reduced = handle_pca(
    df_normalized=df_normalized, 
    result_df=result_df, 
    result_marker=result_marker
    )

# Create sequences
window_size = 30
lookback = 14
X, y = create_sequences(
    df_reduced=df_reduced, 
    lookback=lookback, 
    window_size=window_size,
    result_marker=result_marker
    )

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating the LSTM model
model = Sequential()
# Input layer (only the last value of the sequence is returned)
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
# Dropout to avoid overfitting
model.add(Dropout(0.2))
# Output layer (binary classification - predicting one label)
model.add(Dense(units=1, activation='sigmoid'))
# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Model summary
model.summary()
# Save the trained model to a file
model.save('model.keras')