import pytest
from unittest.mock import patch, MagicMock
from mariola.mariola_calc import mariola_calc

@pytest.fixture
def mock_dependencies():
    with patch('utils.parser_utils.get_parsed_arguments') as mock_get_parsed_arguments, \
         patch('utils.logger_utils.initialize_logger') as mock_initialize_logger, \
         patch('utils.logger_utils.log') as mock_log, \
         patch('utils.app_utils.extract_settings_data') as mock_extract_settings_data, \
         patch('utils.app_utils.load_data_from_csv') as mock_load_data_from_csv, \
         patch('utils.app_utils.save_data_to_csv') as mock_save_data_to_csv, \
         patch('utils.app_utils.save_df_info') as mock_save_df_info, \
         patch('utils.calc_utils.prepare_df') as mock_prepare_df:

        mock_get_parsed_arguments.return_value = ('settings_filename.json', 'data_filename.csv')
        mock_initialize_logger.return_value = None
        mock_log.return_value = None
        mock_extract_settings_data.return_value = {'settings': {'regresion': True, 'clasification': False}}
        mock_load_data_from_csv.return_value = MagicMock()
        mock_save_data_to_csv.return_value = None
        mock_save_df_info.return_value = None
        mock_prepare_df.return_value = MagicMock()

        yield {
            'mock_get_parsed_arguments': mock_get_parsed_arguments,
            'mock_initialize_logger': mock_initialize_logger,
            'mock_log': mock_log,
            'mock_extract_settings_data': mock_extract_settings_data,
            'mock_load_data_from_csv': mock_load_data_from_csv,
            'mock_save_data_to_csv': mock_save_data_to_csv,
            'mock_save_df_info': mock_save_df_info,
            'mock_prepare_df': mock_prepare_df
        }
        

def test_mariola_calc(mock_dependencies):

    mariola_calc()

    mock_dependencies['mock_get_parsed_arguments'].assert_called_with(
        first_arg_str='Settings filename.json',
        second_arg_string='Klines full historical data filename.csv'
    )

    mock_dependencies['mock_initialize_logger'].assert_called_with('settings_filename.json')

    mock_dependencies['mock_extract_settings_data'].assert_called_with('settings_filename.json')

    mock_dependencies['mock_load_data_from_csv'].assert_called_with('data_filename.csv')

    mock_dependencies['mock_prepare_df'].assert_called_with(
        df=mock_dependencies['mock_load_data_from_csv'](),
        regresion=True,
        clasification=False,
        settings=mock_dependencies['mock_extract_settings_data']()['settings'],
        training_mode=True
    )

    mock_dependencies['mock_save_data_to_csv'].assert_called_with(mock_dependencies['mock_prepare_df'](), 'data_filename_calculated')
    mock_dependencies['mock_save_df_info'].assert_called_with(mock_dependencies['mock_prepare_df'](), 'data_filename_calculated.info')

    mock_dependencies['mock_log'].assert_any_call('MariolaCryptoTradingBot. Calculating DataFrame process starting.')
    mock_dependencies['mock_log'].assert_any_call('MariolaCryptoTradingBot. Load data from csv file.')

    mock_dependencies['mock_log'].assert_any_call("MariolaCryptoTradingBot. Calculating Technical Analysis parameters completed.")