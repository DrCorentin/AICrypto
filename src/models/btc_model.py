# src/models/btc_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import os

LOOKBACK = 60  # Number of past time steps to consider for prediction
MODEL_PATH = 'models/btc_model.h5'  # Path to save/load the model

class PredictionLogger(Callback):
    """
    Custom Keras callback to log predicted vs actual prices during training.
    """
    def __init__(self, X_val, y_val, scaler):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_val)
        predicted_prices = self.scaler.inverse_transform(predictions)
        actual_prices = self.scaler.inverse_transform(self.y_val.reshape(-1, 1))
        print(f"\nEpoch {epoch + 1}:")
        print(f"Predicted Price: {predicted_prices[0][0]:.2f}, Actual Price: {actual_prices[0][0]:.2f}")

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

def load_or_build_model(input_shape):
    """
    Load the model if it exists; otherwise, build a new one.
    """
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
    else:
        print("Building a new model...")
        model = build_btc_model(input_shape)
    return model

def train_model(model, X_train, y_train, scaler, X_val=None, y_val=None, epochs=100, batch_size=32):
    """
    Train the model with checkpointing and optionally log predicted vs actual prices.

    Args:
        model (Sequential): The LSTM model.
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        scaler (MinMaxScaler): Scaler for data normalization.
        X_val (ndarray, optional): Validation features.
        y_val (ndarray, optional): Validation labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        model (Sequential): Trained model.
    """
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
    callbacks_list = [checkpoint, early_stopping]

    if X_val is not None and y_val is not None:
        # Add custom callback to log predictions
        prediction_logger = PredictionLogger(X_val, y_val, scaler)
        callbacks_list.append(prediction_logger)
        print(f"Training the model for {epochs} epochs with batch size {batch_size}...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, 
                  validation_data=(X_val, y_val), callbacks=callbacks_list)
    else:
        print(f"Training the model for {epochs} epochs with batch size {batch_size} (no validation)...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks_list)
    
    return model

def predict_price(model, X_test, scaler):
    """
    Predict the next price using the trained BTC model.
    """
    predicted_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    return predicted_price

def calculate_confidence_interval(predictions):
    """
    Calculate the confidence interval of the predictions.
    """
    mean_prediction = np.mean(predictions)
    std_dev = np.std(predictions)
    confidence_interval = (mean_prediction - 1.96 * std_dev, mean_prediction + 1.96 * std_dev)
    return confidence_interval
