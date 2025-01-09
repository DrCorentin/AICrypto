# src/models/btc_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

LOOKBACK = 60  # Number of past minutes to consider for prediction
PREDICTION_INTERVAL = 1  # Predict 1 minute into the future

def preprocess_data(df):
    """
    Preprocess the data for LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data) - PREDICTION_INTERVAL):
        X.append(scaled_data[i - LOOKBACK:i, 0])
        y.append(scaled_data[i + PREDICTION_INTERVAL, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the LSTM model and log the status.
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        history = model.fit(X_train, y_train, batch_size=batch_size, verbose=0)
        print(f"Loss: {history.history['loss'][-1]:.5f}")
    return model

def predict_price(model, data, scaler, log=True):
    """
    Pre
