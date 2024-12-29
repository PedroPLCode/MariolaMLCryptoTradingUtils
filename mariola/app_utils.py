def is_hammer(df):
    df['hammer'] = ((df['high'] - df['close']) > 2 * (df['open'] - df['low'])) & \
                   ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.6) & \
                   ((df['open'] - df['low']) / (df['high'] - df['low']) > 0.6)
    return df

def is_morning_star(df):
    df['morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &  # pierwsza świeca spadkowa
                          (df['open'].shift(1) < df['close'].shift(1)) &  # druga świeca jest doji
                          (df['close'] > df['open']))  # trzecia świeca wzrostowa
    return df

def is_bullish_engulfing(df):
    df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & \
                              (df['open'] < df['close']) & \
                              (df['open'] < df['close'].shift(1)) & \
                              (df['close'] > df['open'].shift(1))
    return df

def calculate_indicator_changes(result, column, avg_periods):
    for avg_period in avg_periods:
        result[f'{column}_ma_{avg_period}'] = result[column].rolling(window=avg_period).mean()
        result[f'{column}_rising_in_avg_period_{avg_period}'] = result[column] > result[f'{column}_ma_{avg_period}']
        result[f'{column}_dropping_in_avg_period_{avg_period}'] = result[column] < result[f'{column}_ma_{avg_period}']
        result[f'{column}_change_vs_ma_{avg_period}'] = result[column] - result[f'{column}_ma_{avg_period}']
        result[f'{column}_pct_change_vs_ma_{avg_period}'] = (result[f'{column}_change_vs_ma_{avg_period}'] / result[f'{column}_ma_{avg_period}']) * 100

def determine_trend(row):
    if row['adx'] > 25:
        if row['plus_di'] > row['minus_di']:
            return 'Bullish'
        elif row['plus_di'] < row['minus_di']:
            return 'Bearish'
    return 'No trend'