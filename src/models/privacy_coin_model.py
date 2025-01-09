# src/models/privacy_coin_model.py

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 60

def preprocess_data_privacy(df, sentiment_data):
    """
    Preprocess price and sentiment data for privacy coin analysis.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    sentiment_scores = np.array(sentiment_data).reshape(-1, 1)
    scaled_sentiment = scaler.fit_transform(sentiment_scores)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_prices) - 1):
        price_window = scaled_prices[i - LOOKBACK:i, 0]
        sentiment_window = scaled_sentiment[i - LOOKBACK:i, 0]
        combined_features = np.column_stack((price_window, sentiment_window))
        X.append(combined_features)
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 2))  # Two features: price and sentiment
    return X, y, scaler

def build_privacy_model(input_shape):
    """
    Build a privacy coin model integrating sentiment and price data.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
