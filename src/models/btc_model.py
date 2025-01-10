# src/models/btc_model.py

from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

LOOKBACK = 60  # Number of past time steps to consider for prediction

def preprocess_data_btc(df):
    """
    Preprocess the data for the BTC model.
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

def build_btc_model(input_shape):
    """
    Build and compile the LSTM model for BTC price prediction.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_until_convergence(model, X_train, y_train, batch_size=32, patience=3, min_delta=1e-4):
    """
    Train the model until convergence using Early Stopping.

    Args:
        model (Sequential): The LSTM model.
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        batch_size (int): Batch size for training.
        patience (int): Number of epochs with no improvement before stopping.
        min_delta (float): Minimum change in loss to be considered as an improvement.

    Returns:
        model (Sequential): The trained model.
    """
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True
    )
    print(f"Training the model with Early Stopping (patience={patience}, min_delta={min_delta})...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, verbose=1, callbacks=[early_stopping])
    return model

def predict_price(model, X_test, scaler):
    """
    Predict the next price using the trained BTC model.
    """
    predicted_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    return predicted_price
