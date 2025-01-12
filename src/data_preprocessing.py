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

def save_preprocessed_data(X, y, filepath="data/preprocessed_data.csv"):
    """
    Save preprocessed features and targets to a CSV file.

    Args:
        X (ndarray): Features array.
        y (ndarray): Target array.
        filepath (str): Path to save the preprocessed data.
    """
    data = pd.DataFrame(X, columns=["Stock-to-Flow", "NVT", "Mining Cost", "Metcalfe Value"])
    data['Price'] = y
    data.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}.")

def load_preprocessed_data(filepath="data/preprocessed_data.csv"):
    """
    Load preprocessed features and targets from a CSV file.

    Args:
        filepath (str): Path to the preprocessed data file.

    Returns:
        X (ndarray): Features array.
        y (ndarray): Target array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No preprocessed data found at {filepath}. Please preprocess the data first.")
    
    data = pd.read_csv(filepath)
    X = data[["Stock-to-Flow", "NVT", "Mining Cost", "Metcalfe Value"]].to_numpy()
    y = data["Price"].to_numpy()
    print(f"Preprocessed data loaded from {filepath}.")
    return X, y
