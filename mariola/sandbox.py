import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequence_length = 30  # np. 30 okresów
features = result.drop(columns=['non_features_column']).values  # Usuń kolumny, które nie są cechami
sequences = create_sequences(features, sequence_length)


train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]


target = result['close'].shift(-1).values  # Przesunięcie w przód o jeden krok
y_sequences = create_sequences(target, sequence_length)  # Dopasowanie do sekwencji


model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, features.shape[1])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(train_sequences, train_targets, epochs=50, batch_size=32, validation_split=0.2)
