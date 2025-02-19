import os
import json
import pandas as pd
import pytest
from utils.app_utils import (
    save_data_to_csv,
    load_data_from_csv,
    save_df_info,
    extract_settings_data,
    save_dataframe_with_info,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


@pytest.fixture
def temp_csv_file(tmp_path):
    return tmp_path / "temp.csv"


@pytest.fixture
def temp_info_file(tmp_path):
    return tmp_path / "temp.info"


@pytest.fixture
def temp_json_file(tmp_path):
    settings = {"key": "value", "number": 42}
    path = tmp_path / "settings.json"
    with open(path, "w") as f:
        json.dump(settings, f)
    return path


def test_save_data_to_csv(sample_dataframe, temp_csv_file):
    save_data_to_csv(sample_dataframe, temp_csv_file)
    assert os.path.exists(temp_csv_file)
    loaded_df = pd.read_csv(temp_csv_file)
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)


def test_save_data_to_csv_none():
    assert save_data_to_csv(None, "test.csv") is None


def test_load_data_from_csv(sample_dataframe, temp_csv_file):
    sample_dataframe.to_csv(temp_csv_file, index=False)
    loaded_df = load_data_from_csv(temp_csv_file)
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)


def test_load_data_from_csv_invalid():
    assert load_data_from_csv("nonexistent.csv") is None


def test_save_df_info(sample_dataframe, temp_info_file):
    save_df_info(sample_dataframe, temp_info_file)
    assert os.path.exists(temp_info_file)
    with open(temp_info_file, "r") as f:
        content = f.read()
        assert "Pandas DataFrame" in content
        assert "Number of Columns: 3" in content
        assert "Number of Rows: 3" in content


def test_save_df_info_invalid():
    assert save_df_info(None, "test.info") is None


def test_extract_settings_data(temp_json_file):
    settings = extract_settings_data(temp_json_file)
    assert settings == {"key": "value", "number": 42}


def test_extract_settings_data_invalid():
    with pytest.raises(SystemExit):
        extract_settings_data("nonexistent.json")


def test_save_dataframe_with_info(sample_dataframe, temp_csv_file):
    base_filename = str(temp_csv_file).replace(".csv", "_calculated.csv")
    stage_name = "processed"
    save_dataframe_with_info(sample_dataframe, base_filename, stage_name)

    expected_csv_file = base_filename.replace("_calculated", f"_{stage_name}")
    expected_info_file = expected_csv_file.replace("csv", "info")

    assert os.path.exists(expected_csv_file)
    assert os.path.exists(expected_info_file)

    saved_df = pd.read_csv(expected_csv_file)
    pd.testing.assert_frame_equal(sample_dataframe, saved_df)

    with open(expected_info_file, "r") as f:
        content = f.read()
        assert "Pandas DataFrame" in content
        assert "Number of Columns: 3" in content
        assert "Number of Rows: 3" in content
