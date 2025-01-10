# src/tools/api_interface.py

import pandas as pd
from binance.client import Client

class APIInterface:
    def __init__(self, exchange="binance"):
        """
        Initialize the APIInterface with the selected exchange.
        Currently, only Binance is supported.
        """
        if exchange == "binance":
            self.client = Client()  # Assumes environment variables for API keys are set

    def fetch_historical_data(self, symbol, start_date="1 Jan 2017"):
        """
        Fetch historical data for a given symbol starting from the specified date.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            start_date (str): Start date for fetching historical data (e.g., '1 Jan 2017').

        Returns:
            pd.DataFrame: Historical data with columns ['open', 'high', 'low', 'close', 'volume'].
        """
        print(f"Fetching historical data for {symbol} starting from {start_date}...")
        klines = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date)

        # Convert raw kline data to a DataFrame
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Format the DataFrame
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data = data.astype(float)  # Ensure numeric types for calculations

        print(f"Fetched {len(data)} rows of data for {symbol}.")
        return data

    def fetch_current_price(self, symbol):
        """
        Fetch the current price for a given symbol.

        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').

        Returns:
            float: The current price of the symbol.
        """
        print(f"Fetching current price for {symbol}...")
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        print(f"Current price for {symbol}: {current_price}")
        return current_price
