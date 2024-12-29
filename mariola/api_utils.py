from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os

load_dotenv()

def get_binance_api_credentials(bot_id=None, testnet=False):
    if testnet:
        api_key = os.environ.get('BINANCE_TESTNET_API_KEY')
        api_secret = os.environ.get('BINANCE_TESTNET_API_SECRET')
    else:
        if bot_id:
            api_key = os.environ.get(f'BINANCE_BOT{bot_id}_API_KEY')
            api_secret = os.environ.get(f'BINANCE_BOT{bot_id}_API_SECRET')
        else:
            api_key = os.environ.get('BINANCE_GENERAL_API_KEY')
            api_secret = os.environ.get('BINANCE_GENERAL_API_SECRET')
    
    return api_key, api_secret


def create_binance_client(bot_id=None, testnet=False):
    try:
        api_key, api_secret = get_binance_api_credentials(bot_id, testnet)
        return Client(api_key, api_secret, testnet=testnet)
    except Exception as e:
        return False


def fetch_data(binance_client, symbol, interval='1h', lookback='2d', start_str=None, end_str=None):
    try: 
        klines = None
        if not start_str and not end_str:
            if lookback[-1] == 'h':
                hours = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(hours=hours)
            elif lookback[-1] == 'd':
                days = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(days=days)
            elif lookback[-1] == 'm':
                minutes = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(minutes=minutes)
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
    
    except BinanceAPIException as e:
        return None
    except ConnectionError as e:
        return None
    except TimeoutError as e:
        return None
    except ValueError as e:
        return None
    except Exception as e:
        return None