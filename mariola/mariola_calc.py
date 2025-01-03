import sys
from time import time
from pathlib import Path
from utils.logger_utils import initialize_logger, log
from utils.parser_utils import get_parsed_arguments
from utils.calc_utils import prepare_df
from utils.app_utils import (
    extract_settings_data, 
    load_data_from_csv,
    save_data_to_csv, 
    save_pandas_df_info
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

def mariola_calc():
    
    start_time = time()

    settings_filename, data_filename = get_parsed_arguments(
        first_arg_str='Settings filename.json',
        second_arg_string='Klines full historical data filename.csv'
    )

    initialize_logger(settings_filename)
    log(f"MariolaCryptoTradingBot. Calculating DataFrame process starting.\n"
        f"Received filename arguments: {settings_filename} {data_filename}"
        )

    settings_data = extract_settings_data(settings_filename)

    settings=settings_data['settings']
    regresion=settings_data['settings']['regresion']
    clasification=settings_data['settings']['clasification']

    log(f"MariolaCryptoTradingBot. Load data from csv file.\n"
        f"starting load_data_from_csv.\n"
        f"filename: {data_filename}"
        )
    data_df = load_data_from_csv(data_filename)
    log(f"MariolaCryptoTradingBot. load_data_from_csv completed.")

    log(f"MariolaCryptoTradingBot. Prepare DataFrame.\n"
        f"starting prepare_df.\n"
        f"regresion: {regresion}\n"
        f"clasification: {clasification}"
        )
    result_df = prepare_df(
        df=data_df, 
        regresion=regresion,
        clasification=clasification,
        settings=settings,
        training_mode=True
        )
    csv_filename = data_filename.replace('_fetched', '_calculated')
    info_filename = csv_filename.replace('csv', 'info')
    save_data_to_csv(result_df, csv_filename)
    save_pandas_df_info(result_df, info_filename)

    end_time = time()

    log(f"MariolaCryptoTradingBot. Calculating Technical Analysis parameters completed"
        f"prepare_df completed and result_df saved to {csv_filename}."
        f"Time taken: {end_time - start_time:.2f} seconds"
        )
    
if __name__ == "__main__":
    mariola_calc()