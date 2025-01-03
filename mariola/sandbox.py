from datetime import datetime
import sys
from pathlib import Path
from utils.logger_utils import initialize_logger
from utils.app_utils import (
    save_pandas_df_info, 
    load_data_from_csv
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

start_date = datetime(2020, 12, 13)
end_date = datetime(2025, 1, 1)
time_difference = end_date - start_date
num_candles = time_difference.total_seconds() / 3600
print(f"num_candles: {num_candles}")

initialize_logger('settings.json')
df = load_data_from_csv('data/btc_1h_calculated.csv')
save_pandas_df_info(df, 'data/btc_1h_calculated.info')