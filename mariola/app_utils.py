def is_hammer(df):
    df['hammer'] = ((df['high'] - df['close']) > 2 * (df['open'] - df['low'])) & \
                   ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.6) & \
                   ((df['open'] - df['low']) / (df['high'] - df['low']) > 0.6)
    return df


def is_morning_star(df):
    df['morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                          (df['open'].shift(1) < df['close'].shift(1)) &
                          (df['close'] > df['open']))
    return df


def is_bullish_engulfing(df):
    df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & \
                              (df['open'] < df['close']) & \
                              (df['open'] < df['close'].shift(1)) & \
                              (df['close'] > df['open'].shift(1))
    return df


def save_df_info(df, filename):
    with open(filename, 'w') as f:
        f.write("Headers DataFrame:\n")
        f.write(str(df.columns) + "\n\n")
        f.write("Len Columns:\n")
        f.write(str(len(df.columns)) + "\n\n")
        f.write("Last 3 rows:\n")
        f.write(df.tail(3).to_string(index=False))