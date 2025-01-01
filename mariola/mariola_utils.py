import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils.logger_utils import log

def normalize_df(result_df=None):
    """
    Normalize the numeric columns in a DataFrame using MinMaxScaler, while replacing infinite values and clipping extreme values.
    
    Args:
        result_df (pd.DataFrame): The input DataFrame containing both numeric and non-numeric columns.
    
    Returns:
        pd.DataFrame: A DataFrame with normalized numeric columns. 
   
    Notes:
        This function replaces `np.inf` and `-np.inf` values with 0, and clips numeric values to the range [-1.8e308, 1.8e308].
        Non-numeric columns are excluded from the normalization process.
    """
    try:
        
        if result_df is None or result_df.empty:
            raise ValueError("result_df must be provided and cannot be None.")
        
        non_numeric_features = result_df.select_dtypes(include=['bool', 'datetime', 'string']).columns.tolist()
        numeric_features = result_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        result_df = result_df.replace([np.inf, -np.inf], 0)
        result_df[numeric_features] = result_df[numeric_features].clip(lower=-1.8e308, upper=1.8e308)
        
        scaler = MinMaxScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(scaler.fit_transform(result_df[numeric_features]), columns=numeric_features)
        
        return df_normalized
    
    except Exception as e:
        log(e)
        return None


def handle_pca(df_normalized=None, result_df=None, result_marker=None):
    """
    Perform Principal Component Analysis (PCA) on the normalized DataFrame and add the target marker to the resulting DataFrame.
    
    Args:
        df_normalized (pd.DataFrame): The normalized DataFrame containing only numeric features.
        result_df (pd.DataFrame): The original DataFrame containing the target marker.
        result_marker (str): The name of the column in `result_df` representing the target marker.
    
    Returns:
        pd.DataFrame: A DataFrame with the reduced features (after PCA) and the target marker.
    
    Notes:
        The PCA transformation reduces the features to the specified number of components (`n_components=50`).
    """
    try:
        
        if result_df is None or result_df.empty:
            raise ValueError("df_normalized must be provided and cannot be None.")
        
        pca = PCA(n_components=50)
        df_reduced = pca.fit_transform(df_normalized)
        df_reduced = pd.DataFrame(df_reduced)
        
        if result_df is not None and result_marker:
            df_reduced[result_marker] = result_df[result_marker]
            
        return df_reduced
    
    except Exception as e:
        log(e)
        return None


def create_sequences(df_reduced=None, lookback=None, window_size=None, result_marker=None):
    """
    Create sequences of features and corresponding target labels from the reduced DataFrame for time series prediction.
    
    Args:
        df_reduced (pd.DataFrame): The reduced DataFrame containing PCA-transformed features and the target marker.
        lookback (int): The number of periods ahead to predict (used to index the target column).
        window_size (int): The number of previous periods used as features in the time series.
        result_marker (str): Column name to be predicted.
    
    Returns:
        tuple: A tuple containing two numpy arrays:
            - X (numpy.ndarray): Array of feature sequences of shape (num_samples, window_size, num_features).
            - y (numpy.ndarray): Array of target labels corresponding to each sequence.
    
    Notes:
        The function extracts sequences of length `window_size` from `df_reduced` and uses the `lookback` value to get the target for each sequence.
    """
    try:
        
        if df_reduced is None or lookback is None or window_size is None or result_marker is None:
            raise ValueError("All arguments must be provided and cannot be None.")
        
        X, y = [], []
        
        for i in range(window_size, len(df_reduced) - lookback):
            
            X.append(df_reduced.iloc[i-window_size:i].values)
            
            if result_marker in df_reduced.columns:
                y.append(df_reduced.iloc[i+lookback][result_marker])
            
        X = np.array(X)
        y = np.array(y)
        
        return np.array(X), np.array(y)
    
    except Exception as e:
        log(e)
        return None