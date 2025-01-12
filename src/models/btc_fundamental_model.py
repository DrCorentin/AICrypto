# src/models/btc_fundamental_model.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class BTCFundamentalModel:
    def __init__(self):
        """
        Initialize the BTC Fundamental Model with a linear regression model.
        """
        self.model = LinearRegression()

    def stock_to_flow(self, stock, flow):
        """
        Compute Stock-to-Flow ratio.
        """
        return stock / flow

    def nvt_ratio(self, market_cap, tx_volume):
        """
        Compute Network Value to Transactions (NVT) ratio.
        """
        return market_cap / tx_volume

    def mining_cost(self, hash_rate, energy_cost, block_reward):
        """
        Estimate the cost of mining one Bitcoin.
        """
        return (hash_rate * energy_cost) / block_reward

    def metcalfe_value(self, user_count):
        """
        Compute Bitcoin value using Metcalfe's Law.
        """
        return user_count ** 2

    def fit(self, X, y):
        """
        Train the composite model using fundamental factors.

        Args:
            X (ndarray): Features array of shape (n_samples, n_features).
            y (ndarray): Target array of actual Bitcoin prices.
        """
        self.model.fit(X, y)
        print("Model trained. Coefficients:", self.model.coef_)

    def predict(self, X):
        """
        Predict Bitcoin prices based on input features.

        Args:
            X (ndarray): Features array of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted Bitcoin prices.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model performance using Mean Squared Error (MSE).

        Args:
            X (ndarray): Features array.
            y (ndarray): Actual Bitcoin prices.

        Returns:
            float: Mean Squared Error.
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        print(f"Mean Squared Error: {mse:.2f}")
        return mse

    def visualize(self, dates, actual_prices, predicted_prices):
        """
        Visualize actual vs predicted Bitcoin prices.

        Args:
            dates (ndarray): Dates corresponding to the prices.
            actual_prices (ndarray): Actual Bitcoin prices.
            predicted_prices (ndarray): Predicted Bitcoin prices.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual_prices, label="Actual Prices", color="blue")
        plt.plot(dates, predicted_prices, label="Predicted Prices", color="orange")
        plt.title("Actual vs Predicted Bitcoin Prices")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        plt.show()
