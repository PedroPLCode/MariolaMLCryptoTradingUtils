import json
import pytest
from utils.logger_utils import initialize_logger, log, log_filename


@pytest.fixture
def valid_settings_file(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {"settings": {"log_filename": str(tmp_path / "test_log.log")}}
    with open(settings_path, "w") as f:
        json.dump(settings_data, f)
    return settings_path


@pytest.fixture
def invalid_settings_file(tmp_path):
    settings_path = tmp_path / "invalid_settings.json"
    with open(settings_path, "w") as f:
        f.write("{invalid json}")
    return settings_path


@pytest.fixture
def missing_key_settings_file(tmp_path):
    settings_path = tmp_path / "missing_key_settings.json"
    settings_data = {"settings": {}}
    with open(settings_path, "w") as f:
        json.dump(settings_data, f)
    return settings_path


def test_initialize_logger_with_valid_file(valid_settings_file):
    initialize_logger(valid_settings_file)
    assert log_filename == str(valid_settings_file.parent / "test_log.log")


def test_initialize_logger_file_not_found():
    with pytest.raises(SystemExit):
        initialize_logger("nonexistent_file.json")


def test_initialize_logger_invalid_json(invalid_settings_file):
    with pytest.raises(SystemExit):
        initialize_logger(invalid_settings_file)


def test_initialize_logger_missing_key(missing_key_settings_file):
    with pytest.raises(SystemExit):
        initialize_logger(missing_key_settings_file)


def test_log_with_initialized_logger(valid_settings_file, capsys):
    initialize_logger(valid_settings_file)
    log("Test message")

    log_file_path = valid_settings_file.parent / "test_log.log"
    assert log_file_path.exists()

    with open(log_file_path, "r") as f:
        log_contents = f.read()

    assert "Test message" in log_contents

    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_log_without_initialization():
    with pytest.raises(RuntimeError, match="Logger has not been initialized"):
        log("Test message")
