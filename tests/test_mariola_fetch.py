from unittest.mock import patch
from mariola.mariola_fetch import mariola_fetch

@patch("mariola.log")
@patch("utils.logger_utils.initialize_logger")
@patch("utils.app_utils.extract_settings_data")
@patch("utils.parser_utils.get_parsed_arguments")
@patch("utils.api_utils.get_full_historical_klines")
@patch("utils.app_utils.save_data_to_csv")
@patch("utils.app_utils.save_df_info")
def test_mariola_fetch_full_flow(
    mock_save_df_info,
    mock_save_data_to_csv,
    mock_get_full_historical_klines,
    mock_get_parsed_arguments,
    mock_extract_settings_data,
    mock_initialize_logger,
    mock_log,
):
    mock_get_parsed_arguments.return_value = ("test_settings.json", "no")

    mock_extract_settings_data.return_value = {
        "fetch_sequence": {
            "step_1": {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "start_str": "1 day ago UTC",
            }
        }
    }

    mock_get_full_historical_klines.return_value = [["fake", "data"]]

    mariola_fetch()

    mock_get_parsed_arguments.assert_called_once_with(
        first_arg_str="Settings filename.json",
        second_arg_string="Dry run mode (yes/no)",
    )
    mock_initialize_logger.assert_called_once_with("test_settings.json")
    mock_extract_settings_data.assert_called_once_with("test_settings.json")
    mock_log.assert_any_call("MariolaCryptoTradingBot. Fetch starting.\nReceived filename argument: test_settings.json")
    mock_get_full_historical_klines.assert_called_once_with(
        symbol="BTCUSDT", interval="1m", start_str="1 day ago UTC"
    )
    mock_save_data_to_csv.assert_called_once_with([["fake", "data"]], "data/df_step_1_fetched.csv")
    mock_save_df_info.assert_called_once_with(
        [["fake", "data"]], "data/df_step_1_fetched.info"
    )
    mock_log.assert_any_call("MariolaCryptoTradingBot Fetching historical data completed.\nTotal steps: 1\nTime taken: ")


@patch("mariola.log")
def test_mariola_fetch_dry_run(mock_log):
    with patch("utils.parser_utils.get_parsed_arguments", return_value=("test_settings.json", "yes")), \
         patch("utils.logger_utils.initialize_logger"), \
         patch("utils.app_utils.extract_settings_data", return_value={"fetch_sequence": {}}):

        mariola_fetch()

    mock_log.assert_any_call("Dry run mode enabled. No data will be fetched or saved.")
    mock_log.assert_any_call("MariolaCryptoTradingBot. Fetch starting.\nReceived filename argument: test_settings.json")


@patch("mariola.log")
@patch("utils.app_utils.extract_settings_data")
def test_mariola_fetch_missing_settings(mock_extract_settings_data, mock_log):
    mock_extract_settings_data.side_effect = ValueError("Invalid settings file.")

    with patch("utils.parser_utils.get_parsed_arguments", return_value=("invalid.json", "no")):
        mariola_fetch()

    mock_log.assert_any_call("Error during data fetching for step None: Invalid settings file.")


@patch("mariola.log")
@patch("utils.api_utils.get_full_historical_klines")
def test_mariola_fetch_api_error(mock_get_full_historical_klines, mock_log):
    mock_get_full_historical_klines.side_effect = Exception("API error")

    with patch(
        "utils.parser_utils.get_parsed_arguments", return_value=("test_settings.json", "no")
    ), patch(
        "mariola_fetch.extract_settings_data",
        return_value={
            "fetch_sequence": {
                "step_1": {
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "start_str": "1 day ago UTC",
                }
            }
        },
    ):
        mariola_fetch()

    mock_log.assert_any_call("Error during data fetching for step step_1: API error")