from api_utils import get_full_historical_klines


# Pobranie danych
symbol = "BTCUSDC"
interval = Client.KLINE_INTERVAL_1HOUR
start_str = "1 Jan, 2020"

data = get_full_historical_klines(symbol, interval, start_str)

# Przetworzenie do DataFrame
columns = ["Open time", "Open", "High", "Low", "Close", "Volume",
           "Close time", "Quote asset volume", "Number of trades",
           "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
df = pd.DataFrame(data, columns=columns)

# Konwersja czasu
df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")

# Zapisanie do pliku CSV
df.to_csv("BTCUSDC_1h_data.csv", index=False)
print("Dane zapisane do BTCUSDC_1h_data.csv")