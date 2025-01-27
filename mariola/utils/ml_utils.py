import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils.logger_utils import log

def normalize_df(df=None, 
                 training_mode=True, 
                 result_marker=None):
    """
    Normalize the numeric columns in a DataFrame using MinMaxScaler, while replacing infinite values 
    and clipping extreme values. Non-numeric columns are excluded from normalization, and the 
    `result_marker` column (if specified) is retained without modification.

    Args:
        df (pd.DataFrame): The input DataFrame containing both numeric and non-numeric columns.
        training_mode (bool): If True, it indicates that the function is being used for training. 
                               In this mode, the `result_marker` column is excluded from normalization 
                               and included in the final output.
        result_marker (str): The name of the column to retain in the final DataFrame without modification. 
                             If None, no column is retained.

    Returns:
        pd.DataFrame: A DataFrame with normalized numeric columns and the `result_marker` column added 
                      at the end, if specified.

    Raises:
        ValueError: If `df` is `None`, empty, or if `result_marker` is specified but not found 
                    in the DataFrame columns.

    Notes:
        - The function replaces `np.inf` and `-np.inf` values in the DataFrame with 0.
        - Numeric values are clipped to the range [-1.8e308, 1.8e308] to avoid extreme outliers.
        - Non-numeric columns such as booleans, datetimes, and strings are excluded from the normalization process.
        - In training mode, the `result_marker` column is excluded from normalization but included in the final DataFrame.

    Example:
        normalized_df = normalize_df(df, training_mode=True, result_marker='target')
    """
    try:
        
        if df is None or df.empty:
            raise ValueError("df must be provided and cannot be None.")
        
        if training_mode and (result_marker and result_marker not in df.columns):
            raise ValueError(f"result_marker '{result_marker}' must be a valid column in df.")
        
        non_numeric_features = df.select_dtypes(
            include=['bool', 'datetime', 'string']
        ).columns.tolist()
        
        numeric_features = df.select_dtypes(
            include=['float64', 'int64']
        ).columns.tolist()
        
        if training_mode and result_marker in numeric_features:
            numeric_features.remove(result_marker)
        
        df = df.replace([np.inf, -np.inf], 0)
        df[numeric_features] = df[numeric_features].clip(
            lower=-1.8e308, upper=1.8e308
        )
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_features]),
            columns=numeric_features
        )
        
        if training_mode and result_marker:
            df_normalized[result_marker] = df[result_marker].values
        
        return df_normalized
    
    except Exception as e:
        log(e)
        return None


def handle_pca(df_normalized=None, 
               loaded_df=None, 
               result_marker=None):
    """
    Perform Principal Component Analysis (PCA) on the normalized DataFrame and add the target marker to the resulting DataFrame.
    
    Args:
        df_normalized (pd.DataFrame): The normalized DataFrame containing only numeric features.
        loaded_df (pd.DataFrame): The original DataFrame containing the target marker.
        result_marker (str): The name of the column in `loaded_df` representing the target marker.
    
    Returns:
        pd.DataFrame: A DataFrame with the reduced features (after PCA) and the target marker.
    
    Notes:
        The PCA transformation reduces the features to the specified number of components (`n_components=50`).
    """
    try:
        
        if df_normalized is None or df_normalized.empty:
            raise ValueError("df_normalized must be provided and cannot be None.")
        
        pca = PCA(n_components=50)
        df_reduced = pca.fit_transform(df_normalized)
        df_reduced = pd.DataFrame(df_reduced)
        
        if result_marker and result_marker in loaded_df.columns:
            df_reduced[result_marker] = loaded_df[result_marker]
        
        return df_reduced
    
    except Exception as e:
        log(e)
        return None


def create_sequences(df_reduced=None, 
                     lookback=None, 
                     window_size=None, 
                     result_marker=None, 
                     training_mode=False):
    """
    Create sequences of features and corresponding target labels from the reduced DataFrame for time series prediction.
    
    Args:
        df_reduced (pd.DataFrame): The DataFrame containing PCA-transformed features and the target marker column.
        lookback (int): The number of periods ahead to predict.
        window_size (int): The number of previous periods used as features in each sequence.
        result_marker (str): The column name in `df_reduced` to be predicted.
        training_mode (bool): If True, generates both feature sequences (X) and corresponding target labels (y). 
                               If False, only generates feature sequences (X) without labels.

    Returns:
        tuple: 
            - X (numpy.ndarray): Array of feature sequences of shape (num_samples, window_size, num_features).
            - y (numpy.ndarray, optional): Array of target labels corresponding to each feature sequence. 
              Only returned if `training_mode=True`.

    Notes:
        - The function extracts sequences of length `window_size` from `df_reduced` for the features. 
          For each sequence, it uses the `lookback` value to determine the target label.
        - If `training_mode` is set to `False`, the function returns only the feature sequences (X) and does not generate target labels (y).
        - If any of the arguments are missing or `None`, the function raises a `ValueError`.

    Example:
        X, y = create_sequences(df_reduced, lookback=14, window_size=30, result_marker='marker_column', training_mode=True)
        X = create_sequences(df_reduced, lookback=14, window_size=30, result_marker='marker_column', training_mode=False)
    """
    try:
        
        if (
            df_reduced is None or
            lookback is None or
            window_size is None or
            result_marker is None
        ):
            raise ValueError("All arguments must be provided and cannot be None.")
        
        X = []
        y = [] if training_mode else None
        
        df_features = df_reduced
        
        if training_mode:
            df_features = df_reduced.drop(columns=[result_marker])

        for i in range(window_size, len(df_reduced) - lookback):
            X.append(df_features.iloc[i-window_size:i].values)
            
            if training_mode:
                y.append(df_reduced.iloc[i+lookback][result_marker])
        
        return np.array(X), np.array(y) if training_mode else np.array(X)
    
    except Exception as e:
        log(e)
        return None, None