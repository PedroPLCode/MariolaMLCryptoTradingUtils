import pytest
from unittest import mock
from mariola.mariola_train import mariola_train

@pytest.fixture
def mock_dependencies():
    with mock.patch('mariola_train.get_parsed_arguments') as mock_get_args, \
         mock.patch('mariola_train.initialize_logger') as mock_initialize_logger, \
         mock.patch('mariola_train.log') as mock_log, \
         mock.patch('mariola_train.extract_settings_data') as mock_extract_settings, \
         mock.patch('mariola_train.load_data_from_csv') as mock_load_data, \
         mock.patch('mariola_train.normalize_df') as mock_normalize, \
         mock.patch('mariola_train.handle_pca') as mock_handle_pca, \
         mock.patch('mariola_train.create_sequences') as mock_create_sequences, \
         mock.patch('mariola_train.train_test_split') as mock_train_test_split, \
         mock.patch('mariola_train.Sequential') as mock_sequential, \
         mock.patch('mariola_train.model.fit') as mock_model_fit, \
         mock.patch('mariola_train.model.evaluate') as mock_model_evaluate, \
         mock.patch('mariola_train.save_df_info') as mock_save_info:

        mock_get_args.return_value = ("settings.json", "data.csv")
        mock_extract_settings.return_value = {
            'settings': {
                'regresion': True,
                'clasification': False,
                'result_marker': 'close',
                'window_size': 10,
                'window_lookback': 5,
                'test_size': 0.2,
                'random_state': 42
            }
        }
        mock_load_data.return_value = mock.MagicMock()
        mock_normalize.return_value = mock.MagicMock()
        mock_handle_pca.return_value = mock.MagicMock()
        mock_create_sequences.return_value = (mock.MagicMock(), mock.MagicMock()) 
        mock_train_test_split.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        mock_model = mock.MagicMock()
        mock_sequential.return_value = mock_model
        mock_model_fit.return_value = None 
        mock_model_evaluate.return_value = (0.01, 0.99) 
        mock_save_info.return_value = None

        yield mock_get_args, mock_initialize_logger, mock_log, mock_extract_settings, mock_load_data, \
              mock_normalize, mock_handle_pca, mock_create_sequences, mock_train_test_split, \
              mock_sequential, mock_model_fit, mock_model_evaluate, mock_save_info


def test_mariola_train(mock_dependencies):
    mock_get_args, mock_initialize_logger, mock_log, mock_extract_settings, mock_load_data, \
    mock_normalize, mock_handle_pca, mock_create_sequences, mock_train_test_split, \
    mock_sequential, mock_model_fit, mock_model_evaluate, mock_save_info = mock_dependencies

    mariola_train.mariola_train()

    mock_get_args.assert_called_once_with(first_arg_str='Settings filename.json', second_arg_string='Calculated and prepared data filename.csv')
    mock_initialize_logger.assert_called_once_with("settings.json")
    mock_log.assert_any_call("MariolaCryptoTradingBot. Training process starting.")
    mock_load_data.assert_called_once_with("data.csv")
    mock_normalize.assert_called_once()
    mock_handle_pca.assert_called_once()
    mock_create_sequences.assert_called_once()
    mock_train_test_split.assert_called_once()
    
    mock_sequential.assert_called_once()
    mock_model_fit.assert_called_once()
    
    mock_model_evaluate.assert_called_once()
    assert mock_model_evaluate.return_value == (0.01, 0.99)
    
    mock_model.save.assert_called_once()
    
    mock_log.assert_any_call("MariolaCryptoTradingBot. LSTM Model training completed.")


if __name__ == "__main__":
    pytest.main()