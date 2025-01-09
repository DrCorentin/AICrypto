# src/models/stablecoin_model.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 60

def preprocess_data_stablecoin(df):
    """
    Preprocess price data to predict deviations from the stablecoin peg.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data) - 1):
        X.append(scaled_data[i - LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_stablecoin_model(input_shape):
    """
    Build a simple regression model for stablecoin peg deviation analysis.
    """
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=input_shape[1]))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
