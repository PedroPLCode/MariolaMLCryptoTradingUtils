import pytest
from unittest.mock import patch
from mariola.mariola_predict import mariola_predict


@pytest.fixture
def mock_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script_name", "settings.json", "model.keras"])


def test_main_flow(mock_args, mock_settings_data, mock_data_df, mock_model):
    with patch('mariola_prediction.get_parsed_arguments', return_value=("settings.json", "model.keras")), \
         patch('mariola_prediction.get_klines', return_value=mock_data_df), \
         patch('mariola_prediction.prepare_df', return_value=mock_data_df), \
         patch('mariola_prediction.normalize_df', return_value=mock_data_df), \
         patch('mariola_prediction.handle_pca', return_value=mock_data_df), \
         patch('mariola_prediction.create_sequences', return_value=[[1, 2], [3, 4]]), \
         patch('mariola_prediction.load_model', return_value=mock_model):
        
        mariola_predict()
        mock_model.predict.assert_called_once_with([[1, 2], [3, 4]])

        with patch("mariola_prediction.log") as mock_log:
            mock_log.assert_any_call("MariolaCryptoTradingBot. Prediction on new data.")
            mock_log.assert_any_call("MariolaCryptoTradingBot. Prediction completed.")

            mock_log.assert_any_call("MariolaCryptoTradingBot. Converting the predictions to binary values (0 or 1).")
            mock_log.assert_any_call("Predictions (clasification):")
            mock_log.assert_any_call(f"Index 1: [[False]]")