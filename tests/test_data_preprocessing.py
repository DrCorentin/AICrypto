# tests/test_data_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import prepare_features, save_preprocessed_data, load_preprocessed_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "circulating_supply": [19000000, 19010000],
        "annual_mining_rate": [328500, 328500],
        "market_cap": [900000000000, 905000000000],
        "tx_volume": [50000000000, 50500000000],
        "hash_rate": [100000000, 105000000],
        "energy_cost": [0.05, 0.05],
        "block_reward": [6.25, 6.25],
        "wallet_count": [100000000, 101000000],
        "price": [45000, 46000]
    })

def test_prepare_features(sample_data):
    X, y = prepare_features(sample_data)
    assert X.shape[0] == len(sample_data)
    assert len(y) == len(sample_data)

def test_save_and_load_preprocessed_data(tmp_path, sample_data):
    X, y = prepare_features(sample_data)
    filepath = tmp_path / "preprocessed_data.csv"
    save_preprocessed_data(X, y, filepath)
    X_loaded, y_loaded = load_preprocessed_data(filepath)

    np.testing.assert_array_almost_equal(X, X_loaded)
    np.testing.assert_array_almost_equal(y, y_loaded)
