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
    # Flatten X to 2D for saving
    X_flat = X.reshape(X.shape[0], -1)
    data = pd.DataFrame(X_flat, columns=["Feature_" + str(i) for i in range(X_flat.shape[1])])
    data["Target"] = y
    data.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}.")

def load_preprocessed_data(filepath="data/preprocessed_data.csv"):
    """
    Load preprocessed features and targets from a CSV file.
    Args:
        filepath (str): Path to the preprocessed data file.
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
    num_features = X_flat.shape[1] // 60  # 60 is the LOOKBACK
    X = X_flat.reshape(X_flat.shape[0], 60, num_features)
    print(f"Preprocessed data loaded from {filepath}.")
    return X, y


