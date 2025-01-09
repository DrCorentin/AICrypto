# src/models/utility_model.py

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 60

def preprocess_data_utility(df, on_chain_data):
    """
    Preprocess price and on-chain data for utility token analysis.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Add on-chain data as an additional feature
    scaled_on_chain_data = scaler.fit_transform(on_chain_data.reshape(-1, 1))

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_prices) - 1):
        price_window = scaled_prices[i - LOOKBACK:i, 0]
        on_chain_window = scaled_on_chain_data[i - LOOKBACK:i, 0]
        combined_features = np.column_stack((price_window, on_chain_window))
        X.append(combined_features)
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 2))  # Two features: price and on-chain data
    return X, y, scaler

def build_utility_model(input_shape):
    """
    Build a utility token model combining price and on-chain metrics.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
