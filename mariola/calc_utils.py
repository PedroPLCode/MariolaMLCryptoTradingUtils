import pandas as pd
import numpy as np
import talib

def is_hammer(df):
    """
    Identifies hammer candlestick patterns in a DataFrame containing stock or cryptocurrency data.
    
    A hammer is a candlestick pattern where the real body is at the top of the candlestick, and 
    the lower shadow is at least twice the size of the real body.
    
    Args:
        df (pd.DataFrame): A DataFrame containing columns ['open', 'high', 'low', 'close'] representing 
                            the open, high, low, and close prices of the candlesticks.
                            
    Returns:
        pd.DataFrame: The original DataFrame with a new column 'hammer' indicating whether each row represents 
                      a hammer candlestick pattern (True or False).
                      
    Notes:
        The 'hammer' column is calculated using the following conditions:
        - The difference between high and close is greater than twice the difference between open and low.
        - The ratio of close to the range (high - low) is greater than 0.6.
        - The ratio of open to the range (high - low) is greater than 0.6.
    """
    
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")
    
    df['hammer'] = ((df['high'] - df['close']) > 2 * (df['open'] - df['low'])) & \
                   ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.6) & \
                   ((df['open'] - df['low']) / (df['high'] - df['low']) > 0.6)
    return df


def is_morning_star(df):
    """
    Identifies morning star candlestick patterns in a DataFrame containing stock or cryptocurrency data.
    
    A morning star is a three-candlestick pattern where:
    - The first candle is a long bearish candle.
    - The second candle is a small-bodied candle (either bullish or bearish) that gaps down.
    - The third candle is a long bullish candle that closes above the midpoint of the first candle.
    
    Args:
        df (pd.DataFrame): A DataFrame containing columns ['open', 'high', 'low', 'close'] representing 
                            the open, high, low, and close prices of the candlesticks.
                            
    Returns:
        pd.DataFrame: The original DataFrame with a new column 'morning_star' indicating whether each row represents 
                      a morning star candlestick pattern (True or False).
                      
    Notes:
        The 'morning_star' column is calculated using the following conditions:
        - The close of the second candlestick is less than the open of the second candlestick (first bearish candle).
        - The open of the third candlestick is less than the close of the second candlestick.
        - The close of the third candlestick is greater than the open of the third candlestick.
    """
    
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")
    
    df['morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                          (df['open'].shift(1) < df['close'].shift(1)) &
                          (df['close'] > df['open']))
    return df


def is_bullish_engulfing(df):
    """
    Identifies bullish engulfing candlestick patterns in a DataFrame containing stock or cryptocurrency data.

    A bullish engulfing pattern occurs when a small bearish candlestick is followed by a large bullish candlestick
    that completely engulfs the previous bearish candle.

    Args:
        df (pd.DataFrame): A DataFrame containing columns ['open', 'high', 'low', 'close'] representing 
                            the open, high, low, and close prices of the candlesticks.

    Returns:
        pd.DataFrame: The original DataFrame with a new column 'bullish_engulfing' indicating whether each row represents 
                      a bullish engulfing candlestick pattern (True or False).

    Notes:
        The 'bullish_engulfing' column is calculated using the following conditions:
        - The previous candlestick is bearish (its open is greater than its close).
        - The current candlestick is bullish (its open is less than its close).
        - The current candlestick's open is less than the previous candlestick's close.
        - The current candlestick's close is greater than the previous candlestick's open.
    """
    
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")
    
    df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & \
                              (df['open'] < df['close']) & \
                              (df['open'] < df['close'].shift(1)) & \
                              (df['close'] > df['open'].shift(1))
    return df


def prepare_df(df=None, regresion=False, clasification=False):
    """
    Prepares the dataframe by calculating technical indicators such as 
    moving averages, RSI, MACD, volume trends, etc., and returns the modified dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing market data. 
    - regresion (bool): Flag to indicate if the dataframe is being prepared for regression (not used in the code right now).
    - clasification (bool): Flag to indicate if the dataframe is being prepared for classification (not used in the code right now).

    Returns:
    - pd.DataFrame: The modified dataframe with technical indicators added as new columns.
    """
    
    min_df_len = 200
    general_timeperiods = [3, 5, 7, 14, 20, 28]
    averages_timeperiods = [3, 5, 7, 14, 20, 28]
    macd_timeperiods = [[12, 9], [6, 9]]
    ema_fast_timeperiod = 9
    ema_slow_timeperiod = 21
    boilinger_timeperiod = [20, 2]
    stochastic_timeperiods = [14, 3]
    stochastic_rsi_timeperiods = [14, 3]
    psar_acceleration = 0.02
    psar_maximum = 0.2
    
    markers_periods = [14, 28]
        
    rsi_buy_value = 30
    rsi_sell_value = 70
    cci_buy_value = -100
    cci_sell_value = 100
    mfi_buy_value = 30
    mfi_sell_value = 70
    stoch_buy_value = 20
    stoch_sell_value = 80
        
    adx_strong_trend = 25
    adx_weak_trend = 20
    adx_no_trend = 5
    
    success_threshold = 5
    drop_threshold = -2
    
    if df is None or df.empty or len(df) < min_df_len:
        raise ValueError("df must be provided and cannot be None or empty.")
    
    result = df.copy()
    
    result['open'] = pd.to_numeric(result['close'], errors='coerce')
    
    result['close'] = pd.to_numeric(result['close'], errors='coerce')
    result[f'is_close_rising'] = result[f'close'].diff() > 0
    result[f'is_close_dropping'] = result[f'close'].diff() < 0
    result['close_change'] = result['close'].diff()
    result['close_pct_change'] = result['close'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'close_ma_{avg_period}'] = result['close'].rolling(window=avg_period).mean()
        result[f'is_close_rising_in_avg_period_{avg_period}'] = result['close'] > result[f'close_ma_{avg_period}']
        result[f'is_close_dropping_in_avg_period_{avg_period}'] = result['close'] < result[f'close_ma_{avg_period}']
        result[f'close_change_vs_ma_{avg_period}'] = result['close'] - result[f'close_ma_{avg_period}']
        
        result[f'close_pct_change_vs_ma_{avg_period}'] = np.where(
            result[f'close_ma_{avg_period}'] != 0,
            (result['close'] - result[f'close_ma_{avg_period}']) / result[f'close_ma_{avg_period}'] * 100,
            0
        )
        
        result[f'max_close_in_{avg_period}'] = result[f'close_ma_{avg_period}'].rolling(window=avg_period).max()
        result[f'min_close_in_{avg_period}'] = result[f'close_ma_{avg_period}'].rolling(window=avg_period).min()
        

    result['high'] = pd.to_numeric(result['high'], errors='coerce')
    result[f'is_high_rising'] = result[f'high'].diff() > 0
    result[f'is_high_dropping'] = result[f'high'].diff() < 0
    result['high_change'] = result['high'].diff()
    result['high_pct_change'] = result['high'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'high_ma_{avg_period}'] = result['high'].rolling(window=avg_period).mean()
        result[f'is_high_rising_in_avg_period_{avg_period}'] = result['high'] > result[f'high_ma_{avg_period}']
        result[f'is_high_dropping_in_avg_period_{avg_period}'] = result['high'] < result[f'high_ma_{avg_period}']
        result[f'high_change_vs_ma_{avg_period}'] = result['high'] - result[f'high_ma_{avg_period}']
        
        result[f'high_pct_change_vs_ma_{avg_period}'] = np.where(
            result[f'high_ma_{avg_period}'] != 0,
            (result['high'] - result[f'high_ma_{avg_period}']) / result[f'high_ma_{avg_period}'] * 100,
            0
        )
        
        result[f'max_high_in_{avg_period}'] = result[f'high_ma_{avg_period}'].rolling(window=avg_period).max()
        result[f'min_high_in_{avg_period}'] = result[f'high_ma_{avg_period}'].rolling(window=avg_period).min()
        
        
    result['low'] = pd.to_numeric(result['low'], errors='coerce')
    result[f'is_low_rising'] = result[f'low'].diff() > 0
    result[f'is_low_dropping'] = result[f'low'].diff() < 0
    result['low_change'] = result['low'].diff()
    result['low_pct_change'] = result['low'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'low_ma_{avg_period}'] = result['low'].rolling(window=avg_period).mean()
        result[f'is_low_rising_in_avg_period_{avg_period}'] = result['low'] > result[f'low_ma_{avg_period}']
        result[f'is_low_dropping_in_avg_period_{avg_period}'] = result['low'] < result[f'low_ma_{avg_period}']
        result[f'low_change_vs_ma_{avg_period}'] = result['low'] - result[f'low_ma_{avg_period}']
        
        result[f'low_pct_change_vs_ma_{avg_period}'] = np.where(
            result[f'low_ma_{avg_period}'] != 0,
            (result['low'] - result[f'low_ma_{avg_period}']) / result[f'low_ma_{avg_period}'] * 100,
            0
        )
        
        result[f'max_low_in_{avg_period}'] = result[f'low_ma_{avg_period}'].rolling(window=avg_period).max()
        result[f'min_low_in_{avg_period}'] = result[f'low_ma_{avg_period}'].rolling(window=avg_period).min()

        
    result['volume'] = pd.to_numeric(result['volume'], errors='coerce')
    result[f'is_volume_rising'] = result[f'volume'].diff() > 0
    result[f'is_volume_dropping'] = result[f'volume'].diff() < 0
    result['volume_change'] = result['volume'].diff()
    result['volume_pct_change'] = result['volume'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'volume_ma_{avg_period}'] = result['volume'].rolling(window=avg_period).mean()
        result[f'is_volume_rising_in_avg_period_{avg_period}'] = result['volume'] > result[f'volume_ma_{avg_period}']
        result[f'is_volume_dropping_in_avg_period_{avg_period}'] = result['volume'] < result[f'volume_ma_{avg_period}']
        result[f'volume_change_vs_ma_{avg_period}'] = result['volume'] - result[f'volume_ma_{avg_period}']
        
        result[f'volume_pct_change_vs_ma_{avg_period}'] = np.where(
            result[f'volume_ma_{avg_period}'] != 0,
            (result['volume'] - result[f'volume_ma_{avg_period}']) / result[f'volume_ma_{avg_period}'] * 100,
            0
        )
        

    result['open_time'] = pd.to_datetime(result['open_time'], unit='ms')
    
    result['open_time_hour'] = result['open_time'].dt.hour
    result['open_time_hour_sin'] = np.sin(2 * np.pi * result['open_time_hour'] / 24)
    result['open_time_hour_cos'] = np.cos(2 * np.pi * result['open_time_hour'] / 24)
    
    result['open_time_weekday'] = result['open_time'].dt.weekday
    result['open_time_weekday_sin'] = np.sin(2 * np.pi * result['open_time_weekday'] / 24)
    result['open_time_weekday_cos'] = np.cos(2 * np.pi * result['open_time_weekday'] / 24)
    
    result['open_time_month'] = result['open_time'].dt.month
    result['open_time_month_sin'] = np.sin(2 * np.pi * result['open_time_month'] / 24)
    result['open_time_month_cos'] = np.cos(2 * np.pi * result['open_time_month'] / 24)

    result['is_open_time_weekend'] = result['open_time_weekday'].isin([5, 6])
    
    result['close_time'] = pd.to_datetime(result['close_time'], unit='ms')
    
    result['close_time_hour'] = result['close_time'].dt.hour
    result['close_time_hour_sin'] = np.sin(2 * np.pi * result['close_time_hour'] / 24)
    result['close_time_hour_cos'] = np.cos(2 * np.pi * result['close_time_hour'] / 24)
    
    result['close_time_weekday'] = result['close_time'].dt.weekday
    result['close_time_weekday_sin'] = np.sin(2 * np.pi * result['close_time_weekday'] / 24)
    result['close_time_weekday_cos'] = np.cos(2 * np.pi * result['close_time_weekday'] / 24)
    
    result['close_time_month'] = result['close_time'].dt.month
    result['close_time_month_sin'] = np.sin(2 * np.pi * result['close_time_month'] / 24)
    result['close_time_month_cos'] = np.cos(2 * np.pi * result['close_time_month'] / 24)
    
    result['is_close_time_weekend'] = result['close_time_weekday'].isin([5, 6])

    is_hammer(result)
    is_morning_star(result)
    is_bullish_engulfing(result)
    
    result['is_hammer_morning'] = (result['hammer'] & (result['open_time_hour'] >= 9) & (result['open_time_hour'] <= 12))
    result['is_morning_star_morning'] = (result['morning_star'] & (result['open_time_hour'] >= 9) & (result['open_time_hour'] <= 12))
    result['is_bullish_engulfing_morning'] = (result['bullish_engulfing'] & (result['open_time_hour'] >= 9) & (result['open_time_hour'] <= 12))

    result['is_hammer_weekend'] = (result['hammer'] & result['open_time_weekday'].isin([5, 6]))
    result['is_morning_star_weekend'] = (result['morning_star'] & result['open_time_weekday'].isin([5, 6]))
    result['is_bullish_engulfing_weekend'] = (result['bullish_engulfing'] & result['open_time_weekday'].isin([5, 6]))


    for period in general_timeperiods:
        
        result[f'rsi_{period}'] = talib.RSI(
            result['close'], 
            timeperiod=period
            )
        result[f'is_rsi_{period}_rising'] = result[f'rsi_{period}'].diff() > 0
        result[f'is_rsi_{period}_dropping'] = result[f'rsi_{period}'].diff() < 0
        result[f'rsi_{period}_change'] = result[f'rsi_{period}'].diff()
        result[f'rsi_{period}_pct_change'] = result[f'rsi_{period}'].pct_change() * 100
        
        result[f'rsi_{period}_buy_signal'] = result[f'rsi_{period}'] < rsi_buy_value
        result[f'rsi_{period}_sell_signal'] = result[f'rsi_{period}'] > rsi_sell_value
        
        result[f'rsi_{period}_bullish_divergence_signal'] = (
            (result[f'is_rsi_{period}_rising'] == True) & 
            (result[f'is_close_dropping'] == True)
        )
        result[f'rsi_{period}_bearish_divergence_signal'] = (
            (result[f'is_rsi_{period}_dropping'] == True) & 
            (result[f'is_close_rising'] == True)
        )

        for avg_period in averages_timeperiods:
            result[f'rsi_{period}_ma_{avg_period}'] = result[f'rsi_{period}'].rolling(window=avg_period).mean()
            result[f'is_rsi_{period}_rising_in_avg_period_{avg_period}'] = result[f'rsi_{period}'] > result[f'rsi_{period}_ma_{avg_period}']
            result[f'is_rsi_{period}_dropping_in_avg_period_{avg_period}'] = result[f'rsi_{period}'] < result[f'rsi_{period}_ma_{avg_period}']
            result[f'rsi_{period}_change_vs_ma_{avg_period}'] = result[f'rsi_{period}'] - result[f'rsi_{period}_ma_{avg_period}']
            
            result[f'rsi_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'rsi_{period}_ma_{avg_period}'] != 0,
                (result[f'rsi_{period}'] - result[f'rsi_{period}_ma_{avg_period}']) / result[f'rsi_{period}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'rsi_{period}_ma_{avg_period}_buy_signal'] = result[f'rsi_{period}_ma_{avg_period}'] < rsi_buy_value
            result[f'rsi_{period}_ma_{avg_period}_sell_signal'] = result[f'rsi_{period}_ma_{avg_period}'] > rsi_sell_value
            
            result[f'rsi_{period}_ma_{avg_period}_bullish_divergence_signal'] = (
                (result[f'is_rsi_{period}_rising_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_dropping_in_avg_period_{avg_period}'] == True)
            )
            result[f'rsi_{period}_ma_{avg_period}_bearish_divergence_signal'] = (
                (result[f'is_rsi_{period}_dropping_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_rising_in_avg_period_{avg_period}'] == True)
            )


        result[f'cci_{period}'] = talib.CCI(
            result['high'],
            result['low'],
            result['close'],
            timeperiod=period
            )
        result[f'is_cci_{period}_rising'] = result[f'cci_{period}'].diff() > 0
        result[f'is_cci_{period}_dropping'] = result[f'cci_{period}'].diff() < 0
        result[f'cci_{period}_change'] = result[f'cci_{period}'].diff()
        result[f'cci_{period}_pct_change'] = result[f'cci_{period}'].pct_change() * 100
        
        result[f'cci_{period}_buy_signal'] = result[f'cci_{period}'] < cci_buy_value
        result[f'cci_{period}_sell_signal'] = result[f'cci_{period}'] > cci_sell_value
        
        result[f'cci_{period}_bullish_divergence_signal'] = (
            (result[f'is_cci_{period}_rising'] == True) & 
            (result[f'is_close_dropping'] == True)
        )
        result[f'cci_{period}_bearish_divergence_signal'] = (
            (result[f'is_cci_{period}_dropping'] == True) & 
            (result[f'is_close_rising'] == True)
        )

        for avg_period in averages_timeperiods:
            result[f'cci_{period}_ma_{avg_period}'] = result[f'cci_{period}'].rolling(window=avg_period).mean()
            result[f'is_cci_{period}_rising_in_avg_period_{avg_period}'] = result[f'cci_{period}'] > result[f'cci_{period}_ma_{avg_period}']
            result[f'is_cci_{period}_dropping_in_avg_period_{avg_period}'] = result[f'cci_{period}'] < result[f'cci_{period}_ma_{avg_period}']
            result[f'cci_{period}_change_vs_ma_{avg_period}'] = result[f'cci_{period}'] - result[f'cci_{period}_ma_{avg_period}']
            
            result[f'cci_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'cci_{period}_ma_{avg_period}'] != 0,
                (result[f'cci_{period}'] - result[f'cci_{period}_ma_{avg_period}']) / result[f'cci_{period}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'cci_{period}_ma_{avg_period}_buy_signal'] = result[f'cci_{period}_ma_{avg_period}'] < cci_buy_value
            result[f'cci_{period}_ma_{avg_period}_sell_signal'] = result[f'cci_{period}_ma_{avg_period}'] > cci_sell_value
            
            result[f'cci_{period}_ma_{avg_period}_bullish_divergence_signal'] = (
                (result[f'is_cci_{period}_rising_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_dropping_in_avg_period_{avg_period}'] == True)
            )
            result[f'cci_{period}_ma_{avg_period}_bearish_divergence_signal'] = (
                (result[f'is_cci_{period}_dropping_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_rising_in_avg_period_{avg_period}'] == True)
            )
            
            
        result[f'mfi_{period}'] = talib.MFI(
            result['high'],
            result['low'],
            result['close'],
            result['volume'],
            timeperiod=period
        )
        result[f'is_mfi_{period}_rising'] = result[f'mfi_{period}'].diff() > 0
        result[f'is_mfi_{period}_dropping'] = result[f'mfi_{period}'].diff() < 0
        result[f'mfi_{period}_change'] = result[f'mfi_{period}'].diff()
        result[f'mfi_{period}_pct_change'] = result[f'mfi_{period}'].pct_change() * 100
        
        result[f'mfi_{period}_buy_signal'] = result[f'mfi_{period}'] < mfi_buy_value
        result[f'mfi_{period}_sell_signal'] = result[f'mfi_{period}'] > mfi_sell_value
        
        result[f'mfi_{period}_bullish_divergence_signal'] = (
            (result[f'is_mfi_{period}_rising'] == True) & 
            (result[f'is_close_dropping'] == True)
        )
        result[f'mfi_{period}_bearish_divergence_signal'] = (
            (result[f'is_mfi_{period}_dropping'] == True) & 
            (result[f'is_close_rising'] == True)
        )

        for avg_period in averages_timeperiods:
            result[f'mfi_{period}_ma_{avg_period}'] = result[f'mfi_{period}'].rolling(window=avg_period).mean()
            result[f'is_mfi_{period}_rising_in_avg_period_{avg_period}'] = result[f'mfi_{period}'] > result[f'mfi_{period}_ma_{avg_period}']
            result[f'is_mfi_{period}_dropping_in_avg_period_{avg_period}'] = result[f'mfi_{period}'] < result[f'mfi_{period}_ma_{avg_period}']
            result[f'mfi_{period}_change_vs_ma_{avg_period}'] = result[f'mfi_{period}'] - result[f'mfi_{period}_ma_{avg_period}']
            
            result[f'mfi_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'mfi_{period}_ma_{avg_period}'] != 0,
                (result[f'mfi_{period}'] - result[f'mfi_{period}_ma_{avg_period}']) / result[f'mfi_{period}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'mfi_{period}_ma_{avg_period}_buy_signal'] = result[f'mfi_{period}_ma_{avg_period}'] < mfi_buy_value
            result[f'mfi_{period}_ma_{avg_period}_sell_signal'] = result[f'mfi_{period}_ma_{avg_period}'] > mfi_sell_value
            
            result[f'mfi_{period}_ma_{avg_period}_bullish_divergence_signal'] = (
                (result[f'is_mfi_{period}_rising_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_dropping_in_avg_period_{avg_period}'] == True)
            )
            result[f'mfi_{period}_ma_{avg_period}_bearish_divergence_signal'] = (
                (result[f'is_mfi_{period}_dropping_in_avg_period_{avg_period}'] == True) & 
                (result[f'is_close_rising_in_avg_period_{avg_period}'] == True)
            )


        result[f'atr_{period}'] = talib.ATR(
            result['high'], 
            result['low'], 
            result['close'], 
            timeperiod=period
            )
        result[f'is_atr_{period}_rising'] = result[f'atr_{period}'].diff() > 0
        result[f'is_atr_{period}_dropping'] = result[f'atr_{period}'].diff() < 0
        result[f'atr_{period}_change'] = result[f'atr_{period}'].diff()
        result[f'atr_{period}_pct_change'] = result[f'atr_{period}'].pct_change() * 100
        for avg_period in averages_timeperiods:
            result[f'atr_{period}_ma_{avg_period}'] = result[f'atr_{period}'].rolling(window=avg_period).mean()
            result[f'is_atr_{period}_rising_in_avg_period_{avg_period}'] = result[f'atr_{period}'] > result[f'atr_{period}_ma_{avg_period}']
            result[f'is_atr_{period}_dropping_in_avg_period_{avg_period}'] = result[f'atr_{period}'] < result[f'atr_{period}_ma_{avg_period}']
            result[f'atr_{period}_change_vs_ma_{avg_period}'] = result[f'atr_{period}'] - result[f'atr_{period}_ma_{avg_period}']
            
            result[f'atr_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'atr_{period}_ma_{avg_period}'] != 0,
                (result[f'atr_{period}'] - result[f'atr_{period}_ma_{avg_period}']) / result[f'atr_{period}_ma_{avg_period}'] * 100,
                0
            )


        result[f'adx_{period}'] = talib.ADX(
            result['high'],
            result['low'],
            result['close'],
            timeperiod=period
            )
        result[f'adx_{period}_strong_trend'] = result[f'adx_{period}'] > adx_strong_trend
        result[f'adx_{period}_weak_trend'] = (result[f'adx_{period}'] > adx_weak_trend) & (result[f'adx_{period}'] < adx_strong_trend)
        result[f'adx_{period}_no_trend'] = result[f'adx_{period}'] < adx_no_trend
        result[f'is_adx_{period}_rising'] = result[f'adx_{period}'].diff() > 0
        result[f'is_adx_{period}_dropping'] = result[f'adx_{period}'].diff() < 0
        result[f'adx_{period}_change'] = result[f'adx_{period}'].diff()
        result[f'adx_{period}_pct_change'] = result[f'adx_{period}'].pct_change() * 100
        
        for avg_period in averages_timeperiods:
            result[f'adx_{period}_ma_{avg_period}'] = result[f'adx_{period}'].rolling(window=avg_period).mean()
            result[f'is_adx_{period}_rising_in_avg_period_{avg_period}'] = result[f'adx_{period}'] > result[f'adx_{period}_ma_{avg_period}']
            result[f'is_adx_{period}_dropping_in_avg_period_{avg_period}'] = result[f'adx_{period}'] < result[f'adx_{period}_ma_{avg_period}']
            result[f'adx_{period}_change_vs_ma_{avg_period}'] = result[f'adx_{period}'] - result[f'adx_{period}_ma_{avg_period}']
            
            result[f'adx_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'adx_{period}_ma_{avg_period}'] != 0,
                (result[f'adx_{period}'] - result[f'adx_{period}_ma_{avg_period}']) / result[f'adx_{period}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'adx_{period}_ma_{avg_period}_strong_trend'] = result[f'adx_{period}_ma_{avg_period}'] > adx_strong_trend
            result[f'adx_{period}_ma_{avg_period}_weak_trend'] = (result[f'adx_{period}_ma_{avg_period}'] > adx_weak_trend) & (result[f'adx_{period}_ma_{avg_period}'] < adx_strong_trend)
            result[f'adx_{period}_ma_{avg_period}_no_trend'] = result[f'adx_{period}_ma_{avg_period}'] < adx_no_trend
            
            
        result[f'plus_di_{period}'] = talib.PLUS_DI(
            result['high'],
            result['low'],
            result['close'],
            timeperiod=period
        )
        result[f'is_plus_di_{period}_rising'] = result[f'plus_di_{period}'].diff() > 0
        result[f'is_plus_di_{period}_dropping'] = result[f'plus_di_{period}'].diff() < 0
        result[f'plus_di_{period}_change'] = result[f'plus_di_{period}'].diff()
        result[f'plus_di_{period}_pct_change'] = result[f'plus_di_{period}'].pct_change() * 100
        
        for avg_period in averages_timeperiods:
            result[f'plus_di_{period}_ma_{avg_period}'] = result[f'plus_di_{period}'].rolling(window=avg_period).mean()
            result[f'is_plus_di_{period}_rising_in_avg_period_{avg_period}'] = result[f'plus_di_{period}'] > result[f'plus_di_{period}_ma_{avg_period}']
            result[f'is_plus_di_{period}_dropping_in_avg_period_{avg_period}'] = result[f'plus_di_{period}'] < result[f'plus_di_{period}_ma_{avg_period}']
            result[f'plus_di_{period}_change_vs_ma_{avg_period}'] = result[f'plus_di_{period}'] - result[f'plus_di_{period}_ma_{avg_period}']
            
            result[f'plus_di_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'plus_di_{period}_ma_{avg_period}'] != 0,
                (result[f'plus_di_{period}'] - result[f'plus_di_{period}_ma_{avg_period}']) / result[f'plus_di_{period}_ma_{avg_period}'] * 100,
                0
            )


        result[f'minus_di_{period}'] = talib.MINUS_DI(
            result['high'],
            result['low'],
            result['close'],
            timeperiod=period
        )
        result[f'is_minus_di_{period}_rising'] = result[f'minus_di_{period}'].diff() > 0
        result[f'is_minus_di_{period}_dropping'] = result[f'minus_di_{period}'].diff() < 0
        result[f'minus_di_{period}_change'] = result[f'minus_di_{period}'].diff()
        result[f'minus_di_{period}_pct_change'] = result[f'minus_di_{period}'].pct_change() * 100
        
        for avg_period in averages_timeperiods:
            result[f'minus_di_{period}_ma_{avg_period}'] = result[f'minus_di_{period}'].rolling(window=avg_period).mean()
            result[f'is_minus_di_{period}_rising_in_avg_period_{avg_period}'] = result[f'minus_di_{period}'] > result[f'minus_di_{period}_ma_{avg_period}']
            result[f'is_minus_di_{period}_dropping_in_avg_period_{avg_period}'] = result[f'minus_di_{period}'] < result[f'minus_di_{period}_ma_{avg_period}']
            result[f'minus_di_{period}_change_vs_ma_{avg_period}'] = result[f'minus_di_{period}'] - result[f'minus_di_{period}_ma_{avg_period}']
            
            result[f'minus_di_{period}_pct_change_vs_ma_{avg_period}'] = np.where(
                result[f'minus_di_{period}_ma_{avg_period}'] != 0,
                (result[f'minus_di_{period}'] - result[f'minus_di_{period}_ma_{avg_period}']) / result[f'minus_di_{period}_ma_{avg_period}'] * 100,
                0
            )


        result[f'bullish_trend_{period}_signal'] = (result[f'adx_{period}'] > adx_strong_trend) & \
                                                (result[f'plus_di_{period}'] > result[f'minus_di_{period}'])
        result[f'bearish_trend_{period}_signal'] = (result[f'adx_{period}'] > adx_strong_trend) & \
                                           (result[f'plus_di_{period}'] < result[f'minus_di_{period}'])


    result[f'ema_{ema_fast_timeperiod}'] = talib.EMA(
        result['close'], 
        timeperiod=ema_fast_timeperiod
        )
    result[f'ema_{ema_fast_timeperiod}_rising'] = result[f'ema_{ema_fast_timeperiod}'].diff() > 0
    result[f'is_ema_{ema_fast_timeperiod}_dropping'] = result[f'ema_{ema_fast_timeperiod}'].diff() < 0
    result[f'is_ema_{ema_fast_timeperiod}_change'] = result[f'ema_{ema_fast_timeperiod}'].diff()
    result[f'ema_{ema_fast_timeperiod}_pct_change'] = result[f'ema_{ema_fast_timeperiod}'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}'] = result[f'ema_{ema_fast_timeperiod}'].rolling(window=avg_period).mean()
        result[f'is_ema_{ema_fast_timeperiod}_rising_in_avg_period_{avg_period}'] = result[f'ema_{ema_fast_timeperiod}'] > result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}']
        result[f'is_ema_{ema_fast_timeperiod}_dropping_in_avg_period_{avg_period}'] = result[f'ema_{ema_fast_timeperiod}'] < result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}']
        result[f'ema_{ema_fast_timeperiod}_change_vs_ma_{avg_period}'] = result[f'ema_{ema_fast_timeperiod}'] - result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}']

        result[f'ema_{ema_fast_timeperiod}_pct_change_vs_ma_{avg_period}'] = np.where(  
            result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}'] != 0,
            (result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}'] - result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}']) / result[f'ema_{ema_fast_timeperiod}_ma_{avg_period}'] * 100,
            0
        )

    result[f'ema_{ema_slow_timeperiod}'] = talib.EMA(
        result['close'], 
        timeperiod=ema_slow_timeperiod
        )
    result[f'is_ema_{ema_slow_timeperiod}_rising'] = result[f'ema_{ema_slow_timeperiod}'].diff() > 0
    result[f'is_ema_{ema_slow_timeperiod}_dropping'] = result[f'ema_{ema_slow_timeperiod}'].diff() < 0
    result[f'ema_{ema_slow_timeperiod}_change'] = result[f'ema_{ema_slow_timeperiod}'].diff()
    result[f'ema_{ema_slow_timeperiod}_pct_change'] = result[f'ema_{ema_slow_timeperiod}'].pct_change() * 100
    
    for avg_period in averages_timeperiods:
        result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}'] = result[f'ema_{ema_slow_timeperiod}'].rolling(window=avg_period).mean()
        result[f'is_ema_{ema_slow_timeperiod}_rising_in_avg_period_{avg_period}'] = result[f'ema_{ema_slow_timeperiod}'] > result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}']
        result[f'is_ema_{ema_slow_timeperiod}_dropping_in_avg_period_{avg_period}'] = result[f'ema_{ema_slow_timeperiod}'] < result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}']
        result[f'ema_{ema_slow_timeperiod}_change_vs_ma_{avg_period}'] = result[f'ema_{ema_slow_timeperiod}'] - result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}']
        
        result[f'ema_{ema_slow_timeperiod}_pct_change_vs_ma_{avg_period}'] = np.where(  
            result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}'] != 0,
            (result[f'ema_{ema_slow_timeperiod}'] - result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}']) / result[f'ema_{ema_slow_timeperiod}_ma_{avg_period}'] * 100,
            0
        )

    result[f'ema_{ema_fast_timeperiod}_prev'] = result[f'ema_{ema_fast_timeperiod}'].shift(1)
    result[f'ema_{ema_slow_timeperiod}_prev'] = result[f'ema_{ema_slow_timeperiod}'].shift(1)
    result[f'ema_{ema_fast_timeperiod}_{ema_slow_timeperiod}_cross_up_signal'] = (result[f'ema_{ema_fast_timeperiod}_prev'] < result[f'ema_{ema_slow_timeperiod}_prev']) & (result[f'ema_{ema_fast_timeperiod}'] > result[f'ema_{ema_slow_timeperiod}'])
    result[f'ema_{ema_fast_timeperiod}_{ema_slow_timeperiod}_cross_down_signal'] = (result[f'ema_{ema_fast_timeperiod}_prev'] > result[f'ema_{ema_slow_timeperiod}_prev']) & (result[f'ema_{ema_fast_timeperiod}'] < result[f'ema_{ema_slow_timeperiod}'])
        
        
    for period in macd_timeperiods:
        macd, macd_signal, _ = talib.MACD(result['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        result[f'macd_{period[0]}'] = macd
        result[f'macd_signal_{period[1]}'] = macd_signal
        result[f'macd_histogram_{period[0]}'] = result[f'macd_{period[0]}'] - result[f'macd_signal_{period[1]}']
        
        result[f'is_macd_{period[0]}_rising'] = result[f'macd_{period[0]}'].diff() > 0
        result[f'is_macd_{period[0]}_dropping'] = result[f'macd_{period[0]}'].diff() < 0
        result[f'macd_{period[0]}_change'] = result[f'macd_{period[0]}'].diff()
        result[f'macd_{period[0]}_pct_change'] = result[f'macd_{period[0]}'].pct_change() * 100
        
        result[f'is_macd_signal_{period[1]}_rising'] = result[f'macd_signal_{period[1]}'].diff() > 0
        result[f'is_macd_signal_{period[1]}_dropping'] = result[f'macd_signal_{period[1]}'].diff() < 0
        result[f'macd_signal_{period[1]}_change'] = result[f'macd_signal_{period[1]}'].diff()
        result[f'macd_signal_{period[1]}_pct_change'] = result[f'macd_signal_{period[1]}'].pct_change() * 100
        
        result[f'is_macd_histogram_{period[0]}_rising'] = result[f'macd_histogram_{period[0]}'].diff() > 0
        result[f'is_macd_histogram_{period[0]}_dropping'] = result[f'macd_histogram_{period[0]}'].diff() < 0
        result[f'macd_histogram_{period[0]}_change'] = result[f'macd_histogram_{period[0]}'].diff()
        result[f'macd_histogram_{period[0]}_pct_change'] = result[f'macd_histogram_{period[0]}'].pct_change() * 100
        
        for avg_period in averages_timeperiods:
            result[f'macd_{period[0]}_ma_{avg_period}'] = result[f'macd_{period[0]}'].rolling(window=avg_period).mean()
            result[f'is_macd_{period[0]}_rising_in_avg_period_{avg_period}'] = result[f'macd_{period[0]}'] > result[f'macd_{period[0]}_ma_{avg_period}']
            result[f'is_macd_{period[0]}_dropping_in_avg_period_{avg_period}'] = result[f'macd_{period[0]}'] < result[f'macd_{period[0]}_ma_{avg_period}']
            result[f'macd_{period[0]}_change_vs_ma_{avg_period}'] = result[f'macd_{period[0]}'] - result[f'macd_{period[0]}_ma_{avg_period}']

            result[f'macd_{period[0]}_pct_change_vs_ma_{avg_period}'] = np.where(  
                result[f'macd_{period[0]}_ma_{avg_period}'] != 0,
                (result[f'macd_{period[0]}'] - result[f'macd_{period[0]}_ma_{avg_period}']) / result[f'macd_{period[0]}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'macd_signal_{period[1]}_ma_{avg_period}'] = result[f'macd_signal_{period[1]}'].rolling(window=avg_period).mean()
            result[f'is_macd_signal_{period[1]}_rising_in_avg_period_{avg_period}'] = result[f'macd_signal_{period[1]}'] > result[f'macd_signal_{period[1]}_ma_{avg_period}']
            result[f'is_macd_signal_{period[1]}_dropping_in_avg_period_{avg_period}'] = result[f'macd_signal_{period[1]}'] < result[f'macd_signal_{period[1]}_ma_{avg_period}']
            result[f'macd_signal_{period[1]}_change_vs_ma_{avg_period}'] = result[f'macd_signal_{period[1]}'] - result[f'macd_signal_{period[1]}_ma_{avg_period}']
            
            result[f'macd_signal_{period[1]}_pct_change_vs_ma_{avg_period}'] = np.where(  
                result[f'macd_signal_{period[1]}_ma_{avg_period}'] != 0,
                (result[f'macd_signal_{period[1]}'] - result[f'macd_signal_{period[1]}_ma_{avg_period}']) / result[f'macd_signal_{period[1]}_ma_{avg_period}'] * 100,
                0
            )
            
            result[f'macd_histogram_{period[0]}_ma_{avg_period}'] = result[f'macd_histogram_{period[0]}'].rolling(window=avg_period).mean()
            result[f'is_macd_histogram_{period[0]}_rising_in_avg_period_{avg_period}'] = result[f'macd_histogram_{period[0]}'] > result[f'macd_histogram_{period[0]}_ma_{avg_period}']
            result[f'is_macd_histogram_{period[0]}_dropping_in_avg_period_{avg_period}'] = result[f'macd_histogram_{period[0]}'] < result[f'macd_histogram_{period[0]}_ma_{avg_period}']
            result[f'macd_histogram_{period[0]}_change_vs_ma_{avg_period}'] = result[f'macd_histogram_{period[0]}'] - result[f'macd_histogram_{period[0]}_ma_{avg_period}']
            
            result[f'macd_histogram_{period[0]}_pct_change_vs_ma_{avg_period}'] = np.where(  
                result[f'macd_histogram_{period[0]}_ma_{avg_period}'] != 0,
                (result[f'macd_histogram_{period[0]}'] - result[f'macd_histogram_{period[0]}_ma_{avg_period}']) / result[f'macd_histogram_{period[0]}_ma_{avg_period}'] * 100,
                0
            )

        result[f'macd_{period[0]}_prev'] = result[f'macd_{period[0]}'].shift(1)
        result[f'macd_signal_{period[1]}_prev'] = result[f'macd_signal_{period[1]}'].shift(1)
        result[f'macd_{period[0]}_cross_up_signal'] = (result[f'macd_{period[0]}_prev'] < result[f'macd_signal_{period[1]}_prev']) & (result[f'macd_{period[0]}'] > result[f'macd_signal_{period[1]}'])
        result[f'macd_{period[0]}_cross_down_signal'] = (result[f'macd_{period[0]}_prev'] > result[f'macd_signal_{period[1]}_prev']) & (result[f'macd_{period[0]}'] < result[f'macd_signal_{period[1]}'])
        result[f'macd_histogram_{period[0]}_buy_signal'] = (result[f'macd_histogram_{period[0]}'].shift(1) < 0) & (result[f'macd_histogram_{period[0]}'] > 0)
        result[f'macd_histogram_{period[0]}_sell_signal'] = (result[f'macd_histogram_{period[0]}'].shift(1) > 0) & (result[f'macd_histogram_{period[0]}'] < 0)

        
    result['upper_band'], result['middle_band'], result['lower_band'] = talib.BBANDS(
            result['close'],
            timeperiod=boilinger_timeperiod[0],
            nbdevup=boilinger_timeperiod[1],
            nbdevdn=boilinger_timeperiod[1],
            matype=0
        )
    result[f'is_upper_band_rising'] = result[f'upper_band'].diff() > 0
    result[f'is_upper_band_dropping'] = result[f'upper_band'].diff() < 0
    result[f'upper_band_change'] = result['upper_band'].diff()
    result[f'upper_band_pct_change'] = result['upper_band'].pct_change() * 100
    
    result[f'is_middle_band_rising'] = result[f'middle_band'].diff() > 0
    result[f'is_middle_band_dropping'] = result[f'middle_band'].diff() < 0
    result[f'middle_band_change'] = result['middle_band'].diff()
    result[f'middle_band_pct_change'] = result['middle_band'].pct_change() * 100
    
    result[f'is_lower_band_rising'] = result[f'lower_band'].diff() > 0
    result[f'is_lower_band_dropping'] = result[f'lower_band'].diff() < 0
    result[f'lower_band_change'] = result['lower_band'].diff()
    result[f'lower_band_pct_change'] = result['lower_band'].pct_change() * 100
    
    result['lower_band_buy_signal'] = (result['close'] < result['lower_band'])
    result['upper_band_sell_signal'] = (result['close'] > result['upper_band'])
    
    
    result['stoch_k'], result['stoch_d'] = talib.STOCH(
        result['high'],
        result['low'],
        result['close'],
        fastk_period=stochastic_timeperiods[0],
        slowk_period=stochastic_timeperiods[1],
        slowk_matype=0,
        slowd_period=stochastic_timeperiods[1],
        slowd_matype=0
    )
    
    result[f'is_stoch_k_rising'] = result[f'stoch_k'].diff() > 0
    result[f'is_stoch_k_dropping'] = result[f'stoch_k'].diff() < 0
    result[f'stoch_k_change'] = result['stoch_k'].diff()
    result[f'stoch_k_pct_change'] = result['stoch_k'].pct_change() * 100
    
    result[f'is_stoch_d_rising'] = result[f'stoch_d'].diff() > 0
    result[f'is_stoch_d_dropping'] = result[f'stoch_d'].diff() < 0
    result[f'stoch_d_change'] = result['stoch_d'].diff()
    result[f'stoch_d_pct_change'] = result['stoch_d'].pct_change() * 100
    
    result['stoch_buy_signal'] = (result['stoch_k'] > result['stoch_d']) & (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1))
    result['stoch_buy_signal_2'] = (result['stoch_k'] > stoch_buy_value) & (result['stoch_k'].shift(1) <= stoch_buy_value)
    result['stoch_buy_signal_combined'] = result['stoch_buy_signal'] | result['stoch_buy_signal_2']
    
    
    result['stoch_rsi_k'], result['stoch_rsi_d'] = talib.STOCHRSI(
        result['close'],
        timeperiod=stochastic_rsi_timeperiods[0],
        fastk_period=stochastic_rsi_timeperiods[1],
        fastd_period=stochastic_rsi_timeperiods[1],
        fastd_matype=0
    )
        
    result[f'is_stoch_rsi_k_rising'] = result[f'stoch_rsi_k'].diff() > 0
    result[f'is_stoch_rsi_k_dropping'] = result[f'stoch_rsi_k'].diff() < 0
    result[f'stoch_rsi_k_change'] = result['stoch_rsi_k'].diff()
    result[f'stoch_rsi_k_pct_change'] = result['stoch_rsi_k'].pct_change() * 100
    
    result[f'is_stoch_rsi_d_rising'] = result[f'stoch_rsi_d'].diff() > 0
    result[f'is_stoch_rsi_d_dropping'] = result[f'stoch_rsi_d'].diff() < 0
    result[f'stoch_rsi_d_change'] = result['stoch_rsi_d'].diff()
    result[f'stoch_rsi_d_pct_change'] = result['stoch_rsi_d'].pct_change() * 100
    
    result['stoch_rsi_buy_signal'] = (result['stoch_rsi_k'] > result['stoch_rsi_d']) & (result['stoch_rsi_k'].shift(1) < result['stoch_rsi_d'].shift(1))
    result['stoch_rsi_sell_signal'] = (result['stoch_rsi_k'] < result['stoch_rsi_d']) & (result['stoch_rsi_k'].shift(1) > result['stoch_rsi_d'].shift(1))
        
        
    result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
    result['vwap'] = (result['typical_price'] * result['volume']).cumsum() / result['volume'].cumsum()
    result[f'is_vwap_rising'] = result[f'vwap'].diff() > 0
    result[f'is_vwap_dropping'] = result[f'vwap'].diff() < 0
    result[f'vwap_change'] = result['vwap'].diff()
    result[f'vwap_pct_change'] = result['vwap'].pct_change() * 100
    
    result['vwap_buy_signal'] = (result['close'] > result['vwap'])
    result['vwap_sell_signal'] = (result['close'] < result['vwap'])
    result['vwap_cross_up_signal'] = ((result['close'] > result['vwap']) & (result['close'].shift(1) <= result['vwap'].shift(1)))
    result['vwap_cross_down_signal'] = ((result['close'] < result['vwap']) & (result['close'].shift(1) >= result['vwap'].shift(1)))
    
    
    result['psar'] = talib.SAR(
        result['high'],
        result['low'],
        acceleration=psar_acceleration,
        maximum=psar_maximum
    )
    result[f'is_psar_rising'] = result[f'psar'].diff() > 0
    result[f'is_psar_dropping'] = result[f'psar'].diff() < 0
    result[f'psar_change'] = result['psar'].diff()
    result[f'psar_pct_change'] = result['psar'].pct_change() * 100
    
    result['psar_buy_signal'] = (result['psar'] < result['close'])
    result['psar_sell_signal'] = (result['psar'] > result['close'])

    result['psar_cross_up_signal'] = (result['psar'] < result['close']) & (result['psar'].shift(1) > result['close'].shift(1))  # Cross from down to up (buy signal)
    result['psar_cross_down_signal'] = (result['psar'] > result['close']) & (result['psar'].shift(1) < result['close'].shift(1))  # Cross from up to down (sell signal)

        
    result['ma_200'] = result['close'].rolling(window=200).mean()
    result[f'is_ma_200_rising'] = result[f'ma_200'].diff() > 0
    result[f'is_ma_200_dropping'] = result[f'ma_200'].diff() < 0
    result[f'ma_200_change'] = result['ma_200'].diff()
    result[f'ma_200_pct_change'] = result['ma_200'].pct_change() * 100
    
    result['ma_200_buy_signal'] = (result['close'] > result['ma_200'])
    result['ma_200_sell_signal'] = (result['close'] < result['ma_200'])
  
    result['ma_50'] = result['close'].rolling(window=50).mean()
    result[f'is_ma_50_rising'] = result[f'ma_50'].diff() > 0
    result[f'is_ma_50_dropping'] = result[f'ma_50'].diff() < 0
    result[f'ma_50_change'] = result['ma_50'].diff()
    result[f'ma_50_pct_change'] = result['ma_50'].pct_change() * 100
    
    result['ma_50_buy_signal'] = (result['close'] > result['ma_50'])
    result['ma_50_sell_signal'] = (result['close'] < result['ma_50'])

    result['ma_50_200_cross_up_signal'] = (result['ma_50'] > result['ma_200']) & (result['ma_50'].shift(1) <= result['ma_200'].shift(1))
    result['ma_50_200_cross_down_signal'] = (result['ma_50'] < result['ma_200']) & (result['ma_50'].shift(1) >= result['ma_200'].shift(1))
  
  
    for marker_period in markers_periods:
        
        result[f'marker_close_pct_change_in_next_{marker_period}_periods'] = ((result['close'].shift(-marker_period) - result['close']) / result['close'] * 100) if regresion else 0
            
        result[f'marker_close_trade_success_in_next_{marker_period}_periods'] = (
            ((result[f'max_close_in_{marker_period}'] - result['close']) / result['close'] * 100 >= success_threshold) & 
            ((result[f'min_close_in_{marker_period}'] - result['close']) / result['close'] * 100 > drop_threshold)
            ) if clasification else 0
            

    result.drop(columns=['open_time', 'close_time'])
    result.fillna(0, inplace=True)
    result[result.select_dtypes(include=['bool']).columns] = result.select_dtypes(include=['bool']).astype(int)

        
    return result