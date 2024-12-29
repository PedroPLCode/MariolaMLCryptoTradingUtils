from calc_utils import calculate_technical_indicators
from api_utils import (
    create_binance_client,
    fetch_data
)

binance_client = create_binance_client(None)
symbol='BTCUSDC'
interval='1h'
lookback='1000d'
df = fetch_data(binance_client, symbol, interval, lookback)
result = calculate_technical_indicators(df)