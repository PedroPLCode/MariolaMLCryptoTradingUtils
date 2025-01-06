import pytest
import pandas as pd
from utils.calc_utils import is_hammer, is_morning_star, is_bullish_engulfing

@pytest.fixture
def sample_data():
    data = {
        'open': [100, 105, 90, 95, 110],
        'high': [110, 110, 95, 100, 120],
        'low': [95, 100, 85, 90, 105],
        'close': [105, 90, 95, 110, 115]
    }
    return pd.DataFrame(data)


def test_is_hammer(sample_data):
    df = sample_data.copy()
    result = is_hammer(df)
    
    assert 'hammer' in result.columns

    expected = [False, False, False, False, False]
    assert result['hammer'].tolist() == expected


def test_is_morning_star(sample_data):
    df = sample_data.copy()
    result = is_morning_star(df)

    assert 'morning_star' in result.columns

    expected = [False, False, False, False, False]
    assert result['morning_star'].tolist() == expected


def test_is_bullish_engulfing(sample_data):
    df = sample_data.copy()
    result = is_bullish_engulfing(df)

    assert 'bullish_engulfing' in result.columns

    expected = [False, False, False, True, False]
    assert result['bullish_engulfing'].tolist() == expected


def test_empty_dataframe():
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

    with pytest.raises(ValueError):
        is_hammer(empty_df)

    with pytest.raises(ValueError):
        is_morning_star(empty_df)

    with pytest.raises(ValueError):
        is_bullish_engulfing(empty_df)


def test_invalid_dataframe():
    invalid_df = pd.DataFrame({})

    with pytest.raises(KeyError):
        is_hammer(invalid_df)

    with pytest.raises(KeyError):
        is_morning_star(invalid_df)

    with pytest.raises(KeyError):
        is_bullish_engulfing(invalid_df)