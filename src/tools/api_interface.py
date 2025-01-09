# src/tools/api_interface.py

from src.tools.api_binance import fetch_historical_data as binance_fetch_historical_data
from src.tools.api_binance import fetch_current_price as binance_fetch_current_price

class APIInterface:
    def __init__(self, exchange="binance"):
        self.exchange = exchange

    def fetch_historical_data(self, symbol, interval='1m', limit=1000):
        if self.exchange == "binance":
            return binance_fetch_historical_data(symbol, interval, limit)
        else:
            raise NotImplementedError(f"Exchange {self.exchange} not supported yet.")

    def fetch_current_price(self, symbol):
        if self.exchange == "binance":
            return binance_fetch_current_price(symbol)
        else:
            raise NotImplementedError(f"Exchange {self.exchange} not supported yet.")
