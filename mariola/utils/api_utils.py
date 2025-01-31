from dotenv import load_dotenv
from datetime import datetime as dt, timedelta
import time
import pandas as pd
from binance.client import Client
import os
from utils.logger_utils import log
from utils.exception_handler import exception_handler

load_dotenv()

@exception_handler()
def create_binance_client():
    """
    Creates and returns a Binance client using API keys from environment variables.
    
    Returns:
        Client: A Binance client object, or None if an error occurs during client creation.
        
    Notes:
        This function retrieves the Binance API key and secret from environment variables 
        (`BINANCE_GENERAL_API_KEY` and `BINANCE_GENERAL_API_SECRET`), and initializes a 
        Binance client using the `Client` class from the `binance` library.
    """
    api_key = os.environ.get('BINANCE_GENERAL_API_KEY')
    api_secret = os.environ.get('BINANCE_GENERAL_API_SECRET')
    binance_client = Client(api_key, api_secret)
    
    return binance_client


@exception_handler()
def get_klines(
    symbol='BTCUSDC', 
    interval='1h', 
    lookback='2d', 
    start_str=None, 
    end_str=None
    ):
    """
    Fetch historical market data (klines) from Binance for a given symbol and time range.
    
    Args:
        symbol (str): The trading pair symbol (default is 'BTCUSDC').
        interval (str): The time interval for each kline (default is '1h').
        lookback (str): The lookback period for fetching data (e.g., '1000d' for 1000 days, '10h' for 10 hours).
        start_str (str, optional): The start time for fetching data in 'YYYY-MM-DD HH:MM:SS' format. 
                                   If not provided, the lookback period will be used.
        end_str (str, optional): The end time for fetching data in 'YYYY-MM-DD HH:MM:SS' format.
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical kline data with the following columns:
            ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
             'taker_buy_quote_asset_volume', 'ignore'].
        None: If an error occurs during data retrieval (e.g., API exceptions, connection issues).
        
    Raises:
        ValueError: If the lookback period format is unsupported.
        BinanceAPIException: If the Binance API returns an error.
        ConnectionError: If there is a connection issue.
        TimeoutError: If the request times out.
        
    Notes:
        - If no start and end time are provided, the function will use the lookback period to calculate 
          the start time and fetch data from there.
        - The function supports lookback periods in hours ('h'), days ('d'), and minutes ('m').
    """
    binance_client = create_binance_client()
    klines = None
    if not start_str and not end_str:
        if lookback[-1] == 'h':
            hours = int(lookback[:-1])
            start_time = dt.utcnow() - timedelta(hours=hours)
        elif lookback[-1] == 'd':
            days = int(lookback[:-1])
            start_time = dt.utcnow() - timedelta(days=days)
        elif lookback[-1] == 'm':
            minutes = int(lookback[:-1])
            start_time = dt.utcnow() - timedelta(minutes=minutes)
        else:
            raise ValueError("Unsupported lookback period format.")
        
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        klines = binance_client.get_historical_klines(
            symbol=symbol, 
            interval=interval, 
            start_str=start_str
        )
    else:
        klines = binance_client.get_historical_klines(
            symbol=symbol, 
            interval=interval, 
            start_str=str(start_str), 
            end_str=str(end_str)
        )
    
    df = pd.DataFrame(
        klines, 
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
            'ignore'
        ]
    )
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    return df

    
@exception_handler()
def get_full_historical_klines(
    symbol='BTCUSDC', 
    interval='1h', 
    start_str=None, 
):
    """
    Fetches complete historical data for a given symbol from the Binance API.
    
    The function retrieves data in intervals of up to 1000 candles per request
    and iterates until it collects all the data from the specified start time 
    (start_str) up to the present time.

    Parameters:
        symbol (str): The trading pair symbol, e.g., 'BTCUSDC' (default is 'BTCUSDC').
        interval (str): The time interval for the candles, e.g., '1m', '5m', '1h', '1d' (default is '1h').
        start_str (str): The start time in the format '1 Jan, 2020' or timestamp in milliseconds.
    
    Returns:
        pd.DataFrame: A DataFrame containing the historical data with the following columns:
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'.
        If an error occurs, the function returns None.
    
    Exceptions:
        - BinanceAPIException: Error during communication with the Binance API.
        - ConnectionError: Connection error.
        - TimeoutError: Timeout error.
        - ValueError: Value error.
        - Other exceptions: General exception.
    """
    all_klines = []
    binance_client = create_binance_client()
    
    if isinstance(start_str, int):
        start_str = str(start_str)
    if isinstance(start_str, str):
        start_date = dt.strptime(start_str, "%d %b, %Y")
        start_str = str(int(start_date.timestamp() * 1000))
    
    timestamp = int(start_str) / 1000
    start_date = dt.utcfromtimestamp(timestamp)
    log(f"Ready to fetch all {symbol} {interval} klines data. start_str: {start_date}")

    klines_count = binance_client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        limit=1
    )

    if klines_count:
        first_kline_timestamp = klines_count[0][0]
        first_kline_date = dt.utcfromtimestamp(first_kline_timestamp / 1000)
        log(f"First available kline timestamp: {first_kline_timestamp}, "
            f"corresponding to {first_kline_date}")
    else:
        log("No data available.")
        return None

    now = dt.now()
    delta = now - start_date
    
    total_candles = None
    if interval == '1h':
        total_candles = delta.total_seconds() / 3600
    elif interval == '30m':
        total_candles = delta.total_seconds() / 1800
    elif interval == '15m':
        total_candles = delta.total_seconds() / 900
    else:
        log(f"Error. Wrong start date: {start_date} or delta: {delta}")
        return
    
    log(f"Total candles from start date: {total_candles}")

    candles_per_iteration = 1000
    total_iterations = (total_candles // candles_per_iteration) + \
        (1 if total_candles % candles_per_iteration else 0)
    log(f"Total iterations required: {total_iterations}")

    iteration = 0
    while iteration < total_iterations:
        iteration += 1
        log(f"Fetching iteration {iteration}/{total_iterations} - {symbol} {interval} klines data.")
        klines = binance_client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            limit=candles_per_iteration
        )
        
        if not klines:
            break
        
        all_klines.extend(klines)

        start_str = klines[-1][0]
        start_str_date = dt.utcfromtimestamp(start_str / 1000)
        log(f"Fetch loop completed. Next start_str: {start_str_date}")

        log(f"Progress: {iteration}/{total_iterations} iterations completed")
        time.sleep(0.1)
    
    df = pd.DataFrame(
        all_klines, 
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
            'ignore'
        ]
    )
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    log(f"All {symbol} {interval} klines data fetched successfully.")
    return df