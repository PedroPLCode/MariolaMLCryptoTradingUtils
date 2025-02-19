import pytest
from unittest.mock import patch


@pytest.fixture
def mock_settings_data():
    return {
        "settings": {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "lookback": 100,
            "regresion": False,
            "clasification": True,
            "result_marker": "close",
            "window_size": 10,
            "window_lookback": 5,
        }
    }


@pytest.fixture
def mock_data_df():
    return [
        {"time": 1234567890, "open": 100, "high": 110, "low": 95, "close": 105},
        {"time": 1234567891, "open": 106, "high": 115, "low": 100, "close": 110},
    ]


@pytest.fixture
def mock_model():
    with patch("tensorflow.keras.models.load_model") as mock_load_model:
        mock_model = mock_load_model.return_value
        mock_model.predict.return_value = [[0.8], [0.4]]
        yield mock_model
