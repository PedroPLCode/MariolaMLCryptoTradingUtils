# MariolaCryptoTradingBot

MariolaCryptoTradingBot is an advanced evolution of [StefanCryptoTradingBot](https://github.com/PedroPLCode/StefanCryptoTradingBot). Born from the rib of StefanCryptoTradingBot, she represents a significant upgrade, offering powerful features and intelligent capabilities. In the future, MariolaCryptoTradingBot will be integrated as a module into StefanCryptoTradingBot, expanding its functionality and skills.

## Features

### She will be:
- **Smarter than Ever**: MariolaCryptoTradingBot is designed to be significantly smarter, utilizing advanced algorithms and learning capabilities.
- **Self-Learning**: The bot continuously learns and adapts to market conditions, making it more effective over time.
- **Autonomous Trading Decisions**: MariolaCryptoTradingBot can make independent trading decisions, analyzing data, trends, and signals to execute trades without human intervention.

## Technologies Used
- **Python**: The primary language used for development.
- **Binance API**: For fetching market data and executing trades on Binance.
- **NumPy, Pandas, and TA-Lib**: Libraries used for implementing trading algorithms, data processing, and technical analysis.
- **Keras**: A deep learning framework used for building the LSTM (Long Short-Term Memory) model for predictions and forecasting market trends.

## Current Status
- **In Planning Phase**: MariolaCryptoTradingBot is still in the deep planning phase, and its final features and capabilities are being developed.
- **Ongoing Development**: Once the planning phase concludes, the bot will enter active development, where the core features will be implemented and tested.

## Key Features:
- **LSTM Model with Keras**: MariolaCryptoTradingBot utilizes a Long Short-Term Memory (LSTM) model built with Keras for time series predictions and market forecasting.
- **Modular Design**: It is designed as a modular component, which can be used within StefanCryptoTradingBot or any other system.
- **Core Scripts**: The bot includes essential scripts:
  - `mariola_fetch.py`: Fetches market data and other required inputs.
  - `mariola_calc.py`: Performs necessary calculations, including indicators and signals.
  - `mariola_train.py`: Handles model training and learning processes.
  - `mariola_predict.py`: Makes predictions based on the trained model and executes trades accordingly.
  
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
python mariola_fetch.py settings.json
python mariola_calc.py settings.json df_filename.csv
python mariola_train.py settings.json df_filename.csv
python mariola_predict.py settings.json model_filename.keras
```

## Important!
Familiarize yourself thoroughly with the source code. Understand its operation. Only then will you be able to customize and adjust scripts to your own needs, preferences, and requirements.

Code created by me, with no small contribution from Dr. Google and Mr. ChatGPT.
Any comments welcome.

MariolaCryptoTradingBot Project is under GNU General Public License Version 3, 29 June 2007