import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from utils.api_utils import create_binance_client, get_klines, get_full_historical_klines

@patch("os.environ.get")
def test_create_binance_client(mock_get):
    mock_get.side_effect = lambda key: "mocked_key" if key == "BINANCE_GENERAL_API_KEY" else "mocked_secret"
    client = create_binance_client()
    assert isinstance(client, Client)

    mock_get.side_effect = Exception("Error")
    client = create_binance_client()
    assert client is None


@patch("utils.api_utils.create_binance_client")
def test_get_klines(mock_create_client):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_client.get_historical_klines.return_value = [
        [1640995200000, "47000", "48000", "46000", "47500", "1000", 1640998800000, "47500000", 100, "500", "47500", "0"]
    ]

    df = get_klines(symbol="BTCUSDC", interval="1h", lookback="1d")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "close" in df.columns
    assert df["close"].iloc[0] == 47500.0

    with pytest.raises(ValueError):
        get_klines(symbol="BTCUSDC", interval="1h", lookback="1x")

    mock_client.get_historical_klines.side_effect = BinanceAPIException(None, "Error")
    df = get_klines(symbol="BTCUSDC", interval="1h", lookback="1d")
    assert df is None


@patch("utils.api_utils.create_binance_client")
@patch("time.sleep", return_value=None)
def test_get_full_historical_klines(mock_sleep, mock_create_client):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_client.get_historical_klines.side_effect = [
        [[1640995200000, "47000", "48000", "46000", "47500", "1000", 1640998800000, "47500000", 100, "500", "47500", "0"]],
        []
    ]

    df = get_full_historical_klines(symbol="BTCUSDC", interval="1h", start_str="1 Jan, 2022")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "close" in df.columns
    assert df["close"].iloc[0] == 47500.0

    mock_client.get_historical_klines.side_effect = BinanceAPIException(None, "Error")
    df = get_full_historical_klines(symbol="BTCUSDC", interval="1h", start_str="1 Jan, 2022")
    assert df is None

    with pytest.raises(ValueError):
        get_full_historical_klines(symbol="BTCUSDC", interval="1h", start_str="Invalid date")