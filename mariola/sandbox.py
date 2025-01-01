from datetime import datetime, timedelta

start_date = datetime(2020, 12, 13)
end_date = datetime(2025, 1, 1)

time_difference = end_date - start_date

num_candles = time_difference.total_seconds() / 3600
print(f"num_candles: {num_candles}")