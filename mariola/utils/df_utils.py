import pandas as pd
import numpy as np
from typing import Union, Optional
import talib
from utils.exception_handler import exception_handler


@exception_handler()
def find_ml_hammer_patterns(df: pd.DataFrame) -> Union[pd.DataFrame, Optional[int]]:
    """
    Identifies the Hammer candlestick pattern and adds related features to the DataFrame.

    A Hammer candlestick pattern is identified when the following conditions are met:
    - The distance between the high and close is more than twice the distance between the open and low.
    - The lower shadow (close to low) is more than 60% of the entire candle's range (high to low).
    - The upper shadow (open to high) is more than 60% of the entire candle's range (high to low).

    This function adds the following features to the DataFrame:
    - 'hammer': Boolean indicating the presence of a Hammer pattern.
    - 'is_hammer_morning': Boolean indicating if the Hammer pattern occurred between 9 AM and 12 PM.
    - 'is_hammer_weekend': Boolean indicating if the Hammer pattern occurred on a weekend (Saturday or Sunday).

    Args:
        df (pandas.DataFrame): DataFrame containing candlestick data (open, high, low, close, close_time).

    Returns:
        pandas.DataFrame or None: DataFrame with the 'hammer', 'is_hammer_morning',
                                   and 'is_hammer_weekend' columns added. Returns None if an error occurs.

    Raises:
        ValueError: If the DataFrame is None or empty.
        Exception: If any error occurs during the pattern identification process.
    """
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")

    df["hammer"] = (
        ((df["high"] - df["close"]) > 2 * (df["open"] - df["low"]))
        & ((df["close"] - df["low"]) / (df["high"] - df["low"]) > 0.6)
        & ((df["open"] - df["low"]) / (df["high"] - df["low"]) > 0.6)
    )

    df["is_hammer_morning"] = (
        df["hammer"] & (df["close_time_hour"] >= 9) & (df["close_time_hour"] <= 12)
    )

    df["is_hammer_weekend"] = df["hammer"] & df["close_time_weekday"].isin([5, 6])

    return df


@exception_handler()
def find_ml_morning_star_patterns(
    df: pd.DataFrame,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Identifies the Morning Star candlestick pattern and adds related features to the DataFrame.

    A Morning Star candlestick pattern is identified when the following conditions are met:
    - The first candle is a bearish candle (close < open).
    - The second candle is a small-bodied candle that gaps down (open < close).
    - The third candle is a bullish candle (close > open).

    This function adds the following features to the DataFrame:
    - 'morning_star': Boolean indicating the presence of a Morning Star pattern.
    - 'is_morning_star_morning': Boolean indicating if the Morning Star pattern occurred between 9 AM and 12 PM.
    - 'is_morning_star_weekend': Boolean indicating if the Morning Star pattern occurred on a weekend (Saturday or Sunday).

    Args:
        df (pandas.DataFrame): DataFrame containing candlestick data (open, close, close_time).

    Returns:
        pandas.DataFrame or None: DataFrame with the 'morning_star', 'is_morning_star_morning',
                                   and 'is_morning_star_weekend' columns added. Returns None if an error occurs.

    Raises:
        ValueError: If the DataFrame is None or empty.
        Exception: If any error occurs during the pattern identification process.
    """
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")

    df["morning_star"] = (
        (df["close"].shift(2) < df["open"].shift(2))
        & (df["open"].shift(1) < df["close"].shift(1))
        & (df["close"] > df["open"])
    )

    df["is_morning_star_morning"] = (
        df["morning_star"]
        & (df["close_time_hour"] >= 9)
        & (df["close_time_hour"] <= 12)
    )

    df["is_morning_star_weekend"] = df["morning_star"] & df["close_time_weekday"].isin(
        [5, 6]
    )

    return df


@exception_handler()
def find_ml_bullish_engulfing_patterns(
    df: pd.DataFrame,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Identifies the Bullish Engulfing candlestick pattern and adds related features to the DataFrame.

    A Bullish Engulfing candlestick pattern is identified when the following conditions are met:
    - The previous candle is a bearish candle (open > close).
    - The current candle is a bullish candle (open < close).
    - The current candle engulfs the previous candle's body (current open < previous close, current close > previous open).

    This function adds the following features to the DataFrame:
    - 'bullish_engulfing': Boolean indicating the presence of a Bullish Engulfing pattern.
    - 'is_bullish_engulfing_morning': Boolean indicating if the Bullish Engulfing pattern occurred between 9 AM and 12 PM.
    - 'is_bullish_engulfing_weekend': Boolean indicating if the Bullish Engulfing pattern occurred on a weekend (Saturday or Sunday).

    Args:
        df (pandas.DataFrame): DataFrame containing candlestick data (open, close, close_time).

    Returns:
        pandas.DataFrame or None: DataFrame with the 'bullish_engulfing', 'is_bullish_engulfing_morning',
                                   and 'is_bullish_engulfing_weekend' columns added. Returns None if an error occurs.

    Raises:
        ValueError: If the DataFrame is None or empty.
        Exception: If any error occurs during the pattern identification process.
    """
    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")

    df["bullish_engulfing"] = (
        (df["open"].shift(1) > df["close"].shift(1))
        & (df["open"] < df["close"])
        & (df["open"] < df["close"].shift(1))
        & (df["close"] > df["open"].shift(1))
    )

    df["is_bullish_engulfing_morning"] = (
        df["bullish_engulfing"]
        & (df["close_time_hour"] >= 9)
        & (df["close_time_hour"] <= 12)
    )

    df["is_bullish_engulfing_weekend"] = df["bullish_engulfing"] & df[
        "close_time_weekday"
    ].isin([5, 6])

    return df


@exception_handler()
def calculate_ml_pct_change_and_lags(
    df: pd.DataFrame, column_names_list: list, lag_period: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Adds percentage change and lag features to multiple columns in the DataFrame.

    This function iterates over a list of column names and performs the following operations:
    - Calculates the percentage change for each column and creates a new column for it.
    - Generates lagged versions of each column for a range of lag values (from 'lag_min' to 'lag_max'
      as specified).

    Args:
        result (pandas.DataFrame): The DataFrame containing the data.
        column_names_list (list of str): A list of column names to which the percentage change and lag features
                                         will be added.
        lag_min (int): The minimum lag value to compute.
        lag_max (int): The maximum lag value to compute.

    Returns:
        pandas.DataFrame or None: The DataFrame with the added columns for percentage change and lags.
                                   Returns None if an error occurs during the process.

    Raises:
        Exception: If an error occurs during the calculation of percentage change or lag features for any column.
    """
    for column_name in column_names_list:

        df[f"{column_name}_pct_change"] = df[f"{column_name}"].pct_change() * 100
        df[f"{column_name}_lag_{lag_period}"] = df[f"{column_name}"].shift(lag_period)

    return df


@exception_handler()
def calculate_ml_momentum_signals(
    df: pd.DataFrame, general_timeperiod: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Adds momentum-related signals to the DataFrame based on the close price.

    This function calculates whether a price is in support or resistance based on a rolling window,
    and adds boolean columns indicating the direction of momentum (positive or negative).
    It also adds a signal for trend reversal, defined when the sign of the close price change reverses.

    Args:
        df (pandas.DataFrame): The DataFrame containing the candlestick data.
        general_timeperiod (int): The rolling window used to calculate support and resistance.

    Returns:
        pandas.DataFrame or None: The DataFrame with the added momentum-related signals.
                                   Returns None if an error occurs during the calculation.
    """
    df["is_support"] = (
        df["close"] == df["close"].rolling(window=general_timeperiod).min()
    )
    df["is_resistance"] = (
        df["close"] == df["close"].rolling(window=general_timeperiod).max()
    )

    df["momentum_positive"] = df["close_pct_change"] > 0
    df["momentum_negative"] = df["close_pct_change"] < 0

    df["trend_reversal_signal"] = (
        df["close_pct_change"].shift(1) * df["close_pct_change"] < 0
    )

    return df


@exception_handler()
def calculate_ml_rsi(
    df: pd.DataFrame, general_timeperiod: int, rsi_buy_value: int, rsi_sell_value: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Calculates the Relative Strength Index (RSI) and generates buy/sell signals.

    This function calculates the RSI based on the closing prices and the given time period.
    It adds columns for the RSI and generates buy/sell signals based on predefined thresholds.

    Args:
        df (pandas.DataFrame): The DataFrame containing the candlestick data.
        general_timeperiod (int): The time period for calculating the RSI.
        rsi_buy_value (float): The RSI value below which a buy signal is generated.
        rsi_sell_value (float): The RSI value above which a sell signal is generated.

    Returns:
        pandas.DataFrame or None: The DataFrame with the calculated RSI and buy/sell signals.
                                   Returns None if an error occurs during the calculation.
    """
    df[f"rsi_{general_timeperiod}"] = talib.RSI(
        df["close"], timeperiod=general_timeperiod
    )

    df[f"rsi_{general_timeperiod}_buy_signal"] = (
        df[f"rsi_{general_timeperiod}"] < rsi_buy_value
    )
    df[f"rsi_{general_timeperiod}_sell_signal"] = (
        df[f"rsi_{general_timeperiod}"] > rsi_sell_value
    )

    return df


@exception_handler()
def calculate_ml_ema(
    df: pd.DataFrame, ema_fast_timeperiod: int, ema_slow_timeperiod: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Calculates the Exponential Moving Averages (EMA) and generates buy/sell signals.

    This function calculates the fast and slow EMAs based on the closing prices and adds buy/sell signals
    based on the crossovers between the fast and slow EMAs.

    Args:
        df (pandas.DataFrame): The DataFrame containing the candlestick data.
        ema_fast_timeperiod (int): The time period for the fast EMA.
        ema_slow_timeperiod (int): The time period for the slow EMA.

    Returns:
        pandas.DataFrame or None: The DataFrame with the calculated EMAs and buy/sell signals.
                                   Returns None if an error occurs during the calculation.
    """
    df[f"ema_{ema_fast_timeperiod}"] = talib.EMA(
        df["close"], timeperiod=ema_fast_timeperiod
    )

    df[f"ema_{ema_slow_timeperiod}"] = talib.EMA(
        df["close"], timeperiod=ema_slow_timeperiod
    )

    df["ema_buy_signal"] = (
        df[f"ema_{ema_fast_timeperiod}"] > df[f"ema_{ema_slow_timeperiod}"]
    )
    df["ema_sell_signal"] = (
        df[f"ema_{ema_fast_timeperiod}"] < df[f"ema_{ema_slow_timeperiod}"]
    )

    return df


@exception_handler()
def calculate_ml_macd(
    df: pd.DataFrame, macd_timeperiod: int, macd_signalperiod: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Calculates the Moving Average Convergence Divergence (MACD) and generates buy/sell signals.

    This function calculates the MACD and its signal line, and generates buy/sell signals based on
    the MACD line crossing above or below the signal line. It also calculates the MACD histogram.

    Args:
        df (pandas.DataFrame): The DataFrame containing the candlestick data.
        macd_timeperiod (int): The time period for calculating the MACD.
        macd_signalperiod (int): The time period for the MACD signal line.

    Returns:
        pandas.DataFrame or None: The DataFrame with the calculated MACD and buy/sell signals.
                                   Returns None if an error occurs during the calculation.
    """
    df[f"macd_{macd_timeperiod}"], df[f"macd_signal_{macd_signalperiod}"], _ = (
        talib.MACD(
            df["close"],
            fastperiod=macd_timeperiod,
            slowperiod=macd_timeperiod * 2,
            signalperiod=macd_signalperiod,
        )
    )

    df[f"macd_histogram_{macd_timeperiod}"] = (
        df[f"macd_{macd_timeperiod}"] - df[f"macd_signal_{macd_signalperiod}"]
    )

    df["macd_buy_signal"] = (
        df[f"macd_{macd_timeperiod}"] > df[f"macd_signal_{macd_signalperiod}"]
    )
    df["macd_sell_signal"] = (
        df[f"macd_{macd_timeperiod}"] < df[f"macd_signal_{macd_signalperiod}"]
    )

    return df


@exception_handler()
def calculate_ml_bollinger_bands(
    df: pd.DataFrame, bollinger_timeperiod: int, bollinger_nbdev: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Calculates Bollinger Bands and generates buy/sell signals.

    This function calculates the upper and lower Bollinger Bands based on the closing prices,
    and generates buy/sell signals when the price crosses the bands.

    Args:
        df (pandas.DataFrame): The DataFrame containing the candlestick data.
        bollinger_timeperiod (int): The time period for calculating the Bollinger Bands.
        bollinger_nbdev (int): The number of standard deviations for the bands.

    Returns:
        pandas.DataFrame or None: The DataFrame with the calculated Bollinger Bands and buy/sell signals.
                                   Returns None if an error occurs during the calculation.
    """
    df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(
        df["close"],
        timeperiod=bollinger_timeperiod,
        nbdevup=bollinger_nbdev,
        nbdevdn=bollinger_nbdev,
        matype=0,
    )

    df["bollinger_buy_signal"] = df["close"] < df["lower_band"]
    df["bollinger_sell_signal"] = df["close"] > df["upper_band"]

    return df


@exception_handler()
def calculate_ml_time_patterns(df: pd.DataFrame) -> Union[pd.DataFrame, Optional[int]]:
    """
    Adds time-based features to the DataFrame based on the 'close_time' column.

    This function creates new features based on the 'close_time' column:
    - Extracts the hour, weekday, and month from 'close_time'.
    - Computes cyclic (sinusoidal and cosinusoidal) transformations for hour, weekday, and month
      to capture cyclical patterns in time.
    - Adds a boolean feature indicating whether the 'close_time' corresponds to a weekend (Saturday or Sunday).

    Args:
        df (pandas.DataFrame): The DataFrame containing the 'close_time' column.

    Returns:
        pandas.DataFrame or None: The DataFrame with additional time-based features.
                                   Returns None if an error occurs during the calculation.
    """
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    df["close_time_hour"] = df["close_time"].dt.hour
    df["close_time_weekday"] = df["close_time"].dt.weekday
    df["close_time_month"] = df["close_time"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["close_time_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["close_time_hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["close_time_weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["close_time_weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["close_time_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["close_time_month"] / 12)

    df["is_weekend"] = df["close_time_weekday"].isin([5, 6])

    return df


@exception_handler()
def calculate_ml_rsi_macd_ratio_and_diff(
    df: pd.DataFrame,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Preprocesses the input DataFrame for use in a Random Forest model.

    The function performs the following steps:
        1. Calculates the 'rsi_macd_ratio' by dividing the RSI (Relative Strength Index) by the MACD histogram, with a small epsilon to prevent division by zero.
        2. Computes the 'macd_signal_diff' as the difference between the MACD signal line and the MACD histogram.

    Arguments:
        df (pandas.DataFrame): The input DataFrame containing columns 'rsi_14', 'macd_histogram_12', and 'macd_signal_9' that will be used for feature calculation.

    Returns:
        pandas.DataFrame: The processed DataFrame with new columns 'rsi_macd_ratio' and 'macd_signal_diff', or None if an error occurs.

    Example:
        df = preprocess_df_for_random_forest(df)
    """
    epsilon = 1e-10
    df["rsi_macd_ratio"] = df["rsi_14"] / (df["macd_histogram_12"] + epsilon)
    df["macd_signal_diff"] = df["macd_signal_9"] - df["macd_histogram_12"]

    return df


@exception_handler()
def handle_initial_ml_df_preparaition(
    df: pd.DataFrame,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Prepares the initial DataFrame by converting specific columns to numeric types.

    This function ensures that the columns 'open', 'low', 'high', 'close', and 'volume'
    in the given DataFrame are converted to numeric types. Non-numeric values are coerced
    to NaN. If an exception occurs during processing, it is logged, and the function
    returns None.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing market data with columns
        'open', 'low', 'high', 'close', and 'volume'.

    Returns:
        pandas.DataFrame: The modified DataFrame with specified columns converted to numeric types.
        Returns None if an exception occurs.

    Raises:
        None: All exceptions are handled and logged internally.
    """
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df


@exception_handler()
def add_ml_regression_etiquete(
    df: pd.DataFrame, marker_period: int
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Adds a regression label to the DataFrame.

    This function calculates the percentage change in the 'close' price over the next
    specified number of periods and adds it as a new column to the DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing a 'close' column with price data.
        marker_period (int): The number of periods in the future to calculate the percentage change.

    Returns:
        pandas.DataFrame: The modified DataFrame with a new column indicating the percentage change
        in the 'close' price for the next 'marker_period' periods.
        Returns None if an exception occurs.

    Raises:
        None: All exceptions are handled and logged internally.
    """
    df[f"marker_close_pct_change_in_next_{marker_period}_periods"] = (
        (df["close"].shift(-marker_period) - df["close"]) / df["close"] * 100
    )

    return df


@exception_handler()
def add_ml_classification_etiquete(
    df: pd.DataFrame,
    marker_period: int,
    success_threshold: float,
    drop_threshold: float,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Adds a classification label to the DataFrame.

    This function determines whether a trade is considered successful or not within
    the next specified number of periods based on the given success and drop thresholds.
    It adds a new boolean column to the DataFrame indicating trade success.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing columns for 'close',
        'max_close_in_{marker_period}', and 'min_close_in_{marker_period}'.
        marker_period (int): The number of periods to evaluate for trade success.
        success_threshold (float): The minimum percentage increase in the 'close' price
        for a trade to be considered successful.
        drop_threshold (float): The maximum percentage decrease in the 'close' price
        to consider the trade successful.

    Returns:
        pandas.DataFrame: The modified DataFrame with a new boolean column indicating
        trade success within the given period.
        Returns None if an exception occurs.

    Raises:
        None: All exceptions are handled and logged internally.
    """
    df[f"marker_close_trade_success_in_next_{marker_period}_periods"] = (
        (df[f"max_close_in_{marker_period}"] - df["close"]) / df["close"] * 100
        >= success_threshold
    ) & (
        (df[f"min_close_in_{marker_period}"] - df["close"]) / df["close"] * 100
        > drop_threshold
    )

    return df


@exception_handler()
def handle_final_ml_df_cleaninig(
    df: pd.DataFrame, columns_to_drop: list
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Performs final cleaning on the DataFrame.

    This function removes specified columns, fills missing values with 0,
    and converts boolean columns to integer type for further processing.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be cleaned.
        columns_to_drop (list): A list of column names to be dropped from the DataFrame.

    Returns:
        pandas.DataFrame: The cleaned DataFrame with specified columns dropped,
        missing values filled with 0, and boolean columns converted to integers.
        Returns None if an exception occurs.

    Raises:
        None: All exceptions are handled and logged internally.
    """
    df.drop(columns=columns_to_drop, inplace=True)
    df.fillna(0, inplace=True)
    df[df.select_dtypes(include=["bool"]).columns] = df.select_dtypes(
        include=["bool"]
    ).astype(int)

    return df


@exception_handler()
def prepare_ml_df(
    df: pd.DataFrame = None,
    regression: bool = False,
    classification: bool = False,
    settings: object = None,
    training_mode: bool = False,
) -> Union[pd.DataFrame, Optional[int]]:
    """
    Prepares the dataframe by calculating technical indicators such as
    moving averages, RSI, MACD, volume trends, etc., and returns the modified dataframe.

    Parameters:
        - df (pd.DataFrame): The input dataframe containing market data.
        - regression (bool): Flag to indicate if the dataframe is being prepared for regression (not used in the code right now).
        - classification (bool): Flag to indicate if the dataframe is being prepared for classification (not used in the code right now).

    Returns:
        - pd.DataFrame: The modified dataframe with technical indicators added as new columns.
    """

    if settings is None:
        raise ValueError("settings must be provided and cannot be None or empty.")

    if df is None or df.empty:
        raise ValueError("df must be provided and cannot be None or empty.")

    result = df.copy()

    handle_initial_ml_df_preparaition(result)

    general_timeperiod = settings["general_timeperiod"]
    bollinger_timeperiod = settings["bollinger_timeperiod"]
    bollinger_nbdev = settings["bollinger_nbdev"]
    macd_timeperiod = settings["macd_timeperiod"]
    macd_signalperiod = settings["macd_signalperiod"]
    ema_fast_timeperiod = settings["ema_fast_timeperiod"]
    ema_slow_timeperiod = settings["ema_slow_timeperiod"]
    rsi_buy_value = settings["rsi_buy_value"]
    rsi_sell_value = settings["rsi_sell_value"]
    calculate_ml_rsi(result, general_timeperiod, rsi_buy_value, rsi_sell_value)
    calculate_ml_macd(result, macd_timeperiod, macd_signalperiod)
    calculate_ml_ema(result, ema_fast_timeperiod, ema_slow_timeperiod)
    calculate_ml_bollinger_bands(result, bollinger_timeperiod, bollinger_nbdev)
    calculate_ml_rsi_macd_ratio_and_diff(result)

    calculate_ml_time_patterns(result)
    find_ml_hammer_patterns(result)
    find_ml_morning_star_patterns(result)
    find_ml_bullish_engulfing_patterns(result)

    lag_period = settings["lag_period"]
    columns_to_calc = ["close", "volume", f"rsi_{general_timeperiod}"]
    calculate_ml_pct_change_and_lags(result, columns_to_calc, lag_period)

    calculate_ml_momentum_signals(result, general_timeperiod)

    if training_mode:
        marker_period = settings["marker_periods"]

        if regression:
            add_ml_regression_etiquete(result, marker_period)

        if classification:
            success_threshold = settings["success_threshold"]
            drop_threshold = settings["drop_threshold"]
            add_ml_classification_etiquete(
                result, marker_period, success_threshold, drop_threshold
            )

    columns_to_drop = [
        "open_time",
        "close_time",
        "ignore",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    handle_final_ml_df_cleaninig(result, columns_to_drop)

    return result
