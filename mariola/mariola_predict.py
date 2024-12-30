from keras.models import load_model
from api_utils import fetch_data
from calc_utils import prepare_df
from mariola_utils import (
    normalize_df, 
    handle_pca, 
    create_sequences
)

# Fetch data
symbol='BTCUSDC'
interval='1h'
lookback='100d'
data_df = fetch_data(
    symbol=symbol, 
    interval=interval, 
    lookback=lookback
    )
print(data_df)

# Prepare df
result_df = prepare_df(
    df=data_df, 
    regresion=False,
    clasification=False
    )

# Normalize data
df_normalized = normalize_df(
    result_df=result_df
    )

# Principal Component Analysis
df_reduced = handle_pca(
    df_normalized=df_normalized, 
    result_df=result_df, 
    result_marker=None
    )

# Create sequences
window_size = 30
lookback = 14
result_marker = 'marker_close_trade_success_in_next_14_periods'
X, _ = create_sequences(
    df_reduced=df_reduced, 
    lookback=lookback, 
    window_size=window_size,
    result_marker=result_marker
    )

# Load the saved model
loaded_model = load_model('model.keras')
# Prediction on new data
y_pred = loaded_model.predict(X)
# Converting the predictions to binary values (0 or 1)
y_pred = (y_pred > 0.5)

print("Predictions:\n", y_pred[:10])