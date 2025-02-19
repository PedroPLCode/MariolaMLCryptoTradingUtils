import pytest
import pandas as pd
from mariola.utils.df_utils import (
    find_hammer_patterns,
    find_morning_star_patterns,
    find_bullish_engulfing_patterns,
)


@pytest.fixture
def sample_data():
    data = {
        "open": [100, 105, 90, 95, 110],
        "high": [110, 110, 95, 100, 120],
        "low": [95, 100, 85, 90, 105],
        "close": [105, 90, 95, 110, 115],
    }
    return pd.DataFrame(data)


def test_find_hammer_patterns(sample_data):
    df = sample_data.copy()
    result = find_hammer_patterns(df)

    assert "hammer" in result.columns

    expected = [False, False, False, False, False]
    assert result["hammer"].tolist() == expected


def test_find_morning_star_patterns(sample_data):
    df = sample_data.copy()
    result = find_morning_star_patterns(df)

    assert "morning_star" in result.columns

    expected = [False, False, False, False, False]
    assert result["morning_star"].tolist() == expected


def test_find_bullish_engulfing_patterns(sample_data):
    df = sample_data.copy()
    result = find_bullish_engulfing_patterns(df)

    assert "bullish_engulfing" in result.columns

    expected = [False, False, False, True, False]
    assert result["bullish_engulfing"].tolist() == expected


def test_empty_dataframe():
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close"])

    with pytest.raises(ValueError):
        find_hammer_patterns(empty_df)

    with pytest.raises(ValueError):
        find_morning_star_patterns(empty_df)

    with pytest.raises(ValueError):
        find_bullish_engulfing_patterns(empty_df)


def test_invalid_dataframe():
    invalid_df = pd.DataFrame({})

    with pytest.raises(KeyError):
        find_hammer_patterns(invalid_df)

    with pytest.raises(KeyError):
        find_morning_star_patterns(invalid_df)

    with pytest.raises(KeyError):
        find_bullish_engulfing_patterns(invalid_df)
