from sklearn.preprocessing import MinMaxScaler
from calc_utils import prepare_df_for_ml
from api_utils import fetch_data
from app_utils import save_df_info

symbol='BTCUSDC'
interval='1h'
lookback='10d'

data_df = fetch_data(
    symbol=symbol, 
    interval=interval, 
    lookback=lookback
    )

result_df = prepare_df_for_ml(
    df=data_df, 
    regresion=True,
    clasification=True
    )

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(result_df.dropna().values)

save_df_info(result_df, 'output.txt')