# src/data_preprocessing.py

import pandas as pd
import numpy as np
from src.models.btc_fundamental_model import BTCFundamentalModel
import os

def prepare_features(data):
    """
    Prepare features for the composite model using fundamental indicators.

    Args:
        data (DataFrame): Historical Bitcoin data with necessary columns.

    Returns:
        X (ndarray): Features array for the model.
        y (ndarray): Target array of actual Bitcoin prices.
    """
    model = BTCFundamentalModel()

    # Calculate individual features
    stock_to_flow = model.stock_to_flow(data['circulating_supply'], data['annual_mining_rate'])
    nvt = model.nvt_ratio(data['market_cap'], data['tx_volume'])
    mining_cost = model.mining_cost(data['hash_rate'], data['energy_cost'], data['block_reward'])
    metcalfe_value = model.metcalfe_value(data['wallet_count'])

    # Combine features into a single array
    X = pd.DataFrame({
        "Stock-to-Flow": stock_to_flow,
        "NVT": nvt,
        "Mining Cost": mining_cost,
        "Metcalfe Value": metcalfe_value,
    }).fillna(0).to_numpy()

    y = data['price'].to_numpy()  # Actual prices as the target

    return X, y

def save_preprocessed_data(X, y, scaler, filepath="data/preprocessed_data.csv"):
    """
    Save preprocessed features, targets, and scaler to a file.
    
    Args:
        X (ndarray): Features array (3D) for the model.
        y (ndarray): Target array.
        scaler (MinMaxScaler): Scaler used for normalization (optional for loading).
        filepath (str): Path to save the preprocessed data.
    """
    # Flatten X from 3D (samples, lookback, features) to 2D (samples, lookback * features)
    X_flat = X.reshape(X.shape[0], -1)
    
    # Create a DataFrame for saving
    data = pd.DataFrame(X_flat, columns=["Feature_" + str(i) for i in range(X_flat.shape[1])])
    data["Target"] = y  # Add the target column
    data.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}.")

def load_preprocessed_data(filepath="data/preprocessed_data.csv", lookback=60, num_features=1):
    """
    Load preprocessed features and targets from a CSV file.
    
    Args:
        filepath (str): Path to the preprocessed data file.
        lookback (int): The number of time steps in the lookback window.
        num_features (int): Number of features per time step.

    Returns:
        X (ndarray): Features array (3D) reshaped for the model.
        y (ndarray): Target array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No preprocessed data found at {filepath}. Please preprocess the data first.")
    
    data = pd.read_csv(filepath)
    X_flat = data.drop(columns=["Target"]).to_numpy()
    y = data["Target"].to_numpy()

    # Reshape X back to 3D
    X = X_flat.reshape(X_flat.shape[0], lookback, num_features)
    print(f"Preprocessed data loaded from {filepath}.")
    return X, y


