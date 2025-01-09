# src/tools/api_binance.py

import requests
import pandas as pd

API_BASE = 'https://api.binance.com'

def fetch_historical_data(symbol, interval='1m', limit=1000):
    """
    Fetch historical klines for a given symbol and interval from Binance API.
    """
    url = f"{API_BASE}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for failed requests
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def fetch_current_price(symbol):
    """
    Fetch the current price for a given symbol from Binance API.
    """
    url = f"{API_BASE}/api/v3/ticker/price"
    params = {'symbol': symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return float(data['price'])
