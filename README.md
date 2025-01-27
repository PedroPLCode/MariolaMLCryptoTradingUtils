# MariolaCryptoTradingBot

MariolaCryptoTradingBot is a collection of powerful scripts designed for training and optimizing trading models to be later used in the main trading bot, [StefanCryptoTradingBot](https://github.com/PedroPLCode/StefanCryptoTradingBot). Born from the rib of StefanCryptoTradingBot, this project focuses on developing machine learning models that predict cryptocurrency market trends, which can be integrated into the trading system for real-time autonomous decision-making.
The models can be used in real-time applications, such as trading bots. This repository includes scripts for fetching historical data, training models (LSTM, Random Forest, XGBoost), and making predictions based on these trained models.

## Features
- Fetch historical cryptocurrency data from Binance API.
- Train and evaluate machine learning models (LSTM, Random Forest, XGBoost).
- Make predictions using pre-trained models.
- Normalize, apply PCA, and create sequences for model training.
- Logging and settings management.

## Technologies Used
- **Python**: The primary language used for development.
- **Binance API**: For fetching market data and executing trades on Binance.
- **NumPy, Pandas, and TA-Lib**: Libraries used for implementing trading algorithms, data processing, and technical analysis.
- **Scikit-Learn, XGBoost and Keras**: A machine learning frameworks used for building the ML models for predictions and forecasting market trends.

## Current Status
**Ongoing Development**

## Key Features:
- **Modular Design**: It is designed as a modular component, which can be used within StefanCryptoTradingBot or any other system.
- **Core Scripts**: The bot includes essential scripts:
  - `fetch_data.py`: Fetch historical market data.
  - `calculate_df.py`: Process and calculate technical indicators for the data.
  - `train_lstm_model.py`: Train an LSTM model with the given data.
  - `train_rf_model.py`: Train a Random Forest model.
  - `train_xgboost_model.py`: Train an XGBoost model.
  - `predict_lstm_model.py`: Make predictions using the LSTM model.
  - `predict_rf_model.py`: Make predictions using the Random Forest model.
  - `predict_xgboost_model.py`: Make predictions using the XGBoost model.
  
- **Settings Configuration**: The bot requires a properly configured `settings.json` file to function correctly. This configuration file contains parameters and settings needed to run the bot smoothly.

## How to use

To install and set up the bot locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MariolaCryptoTradingBot.git
cd MariolaCryptoTradingBot
```

2. Set up a Python virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate or . venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your environment variables by creating a .env file with your Binance API credentials.
```bash
BINANCE_GENERAL_API_KEY='your_binance_general_api_key'
BINANCE_GENERAL_API_SECRET='your_binance_general_api_secret'
```

5. Run tests:
```bash
pytest
```

6. Run scripts:
```bash
python script.py settings.json data_filename.csv or model_filename.keras
```

## Important!
Familiarize yourself thoroughly with the source code. Understand its operation. Only then will you be able to customize and adjust scripts to your own needs, preferences, and requirements. Only then will you be able to use it correctly and avoid potential issues. Knowledge of the underlying code is essential for making informed decisions and ensuring the successful implementation of the bot for your specific use case. Make sure to review all components and dependencies before running the scripts.

Code created by me, with no small contribution from Dr. Google and Mr. ChatGPT.
Any comments welcome.

MariolaCryptoTradingBot Project is under GNU General Public License Version 3, 29 June 2007