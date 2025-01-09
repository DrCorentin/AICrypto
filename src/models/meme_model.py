# src/models/meme_model.py

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from src.agents.sentiment_analysis import fetch_social_data, get_sentiment_score

LOOKBACK = 60

def preprocess_data_meme(df):
    """
    Preprocess price and sentiment data for the meme coin model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Get sentiment data
    social_data = fetch_social_data(keyword="meme coin", limit=100)
    sentiment_score = get_sentiment_score(social_data)

    # Create input data with sentiment as a feature
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_prices) - 1):
        feature_window = scaled_prices[i - LOOKBACK:i, 0]
        sentiment_window = np.full((LOOKBACK,), sentiment_score)  # Repeat the sentiment score
        combined_features = np.column_stack((feature_window, sentiment_window))
        X.append(combined_features)
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 2))  # Two features: price and sentiment
    return X, y, scaler

def build_meme_model(input_shape):
    """
    Build a meme coin prediction model that includes sentiment analysis.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
