import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from mariola.mariola_utils import normalize_df, handle_pca, create_sequences


@patch("mariola.log")
def test_normalize_df(mock_log):
    data = {
        "feature1": [1, 2, 3, np.inf],
        "feature2": [-1, -2, -3, -np.inf],
        "marker": [1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    result = normalize_df(result_df=df, training_mode=True, result_marker="marker")

    assert result is not None
    assert "marker" in result.columns
    assert result.shape == (4, 3)

    result = normalize_df(result_df=None, training_mode=True, result_marker="marker")
    assert result is None
    mock_log.assert_called()

    with pytest.raises(ValueError):
        normalize_df(result_df=df, training_mode=True, result_marker="missing_marker")

    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        normalize_df(result_df=empty_df, training_mode=True, result_marker="marker")


@patch("mariola.log")
def test_handle_pca(mock_log):
    data = {
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.9, 0.8, 0.7, 0.6],
        "marker": [1, 0, 1, 0],
    }
    df_normalized = pd.DataFrame(data)

    result = handle_pca(
        df_normalized=df_normalized.drop(columns=["marker"]),
        result_df=df_normalized,
        result_marker="marker",
    )

    assert result is not None
    assert result.shape[1] == 51

    result = handle_pca(df_normalized=None, result_df=None, result_marker="marker")
    assert result is None
    mock_log.assert_called()


@patch("mariola.log")
def test_create_sequences(mock_log):
    data = {
        "pca1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "pca2": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "marker": [1, 0, 1, 0, 1, 0],
    }
    df_reduced = pd.DataFrame(data)

    X, y = create_sequences(
        df_reduced=df_reduced,
        lookback=1,
        window_size=2,
        result_marker="marker",
        training_mode=True,
    )

    assert X.shape == (3, 2, 2)
    assert y.shape == (3,)

    X = create_sequences(
        df_reduced=df_reduced,
        lookback=1,
        window_size=2,
        result_marker="marker",
        training_mode=False,
    )

    assert X.shape == (3, 2, 2)

    X, y = create_sequences(
        df_reduced=None,
        lookback=1,
        window_size=2,
        result_marker="marker",
        training_mode=True,
    )
    assert X is None
    assert y is None
    mock_log.assert_called()

    with pytest.raises(KeyError):
        create_sequences(
            df_reduced=df_reduced.drop(columns=["marker"]),
            lookback=1,
            window_size=2,
            result_marker="marker",
            training_mode=True,
        )
