# main.py

from src.models.btc_model import preprocess_data_btc, load_or_build_model, train_model, predict_price, calculate_confidence_interval
from src.tools.api_interface import APIInterface
from sklearn.model_selection import train_test_split
import pandas as pd
import signal
import sys
import json
import os

# Portfolio management
def save_portfolio(balance, holdings, filepath="portfolio.json"):
    portfolio = {"balance_eur": balance, "crypto_holdings": holdings}
    with open(filepath, "w") as f:
        json.dump(portfolio, f)
    print("Portfolio saved.")

def load_portfolio(filepath="portfolio.json"):
    try:
        with open(filepath, "r") as f:
            portfolio = json.load(f)
        print("Portfolio loaded.")
        return portfolio["balance_eur"], portfolio["crypto_holdings"]
    except FileNotFoundError:
        print("No saved portfolio found. Initializing new portfolio.")
        return 50.0, 0.0  # Default starting values

def save_preprocessed_data(X, y, scaler, filepath="data/preprocessed_data.csv"):
    """
    Save preprocessed features, targets, and scaler to a file.
    """
    data = pd.DataFrame(X, columns=["Feature_" + str(i) for i in range(X.shape[1])])
    data["Target"] = y
    data.to_csv(filepath, index=False)
    print(f"Preprocessed data saved to {filepath}.")

def load_preprocessed_data(filepath="data/preprocessed_data.csv"):
    """
    Load preprocessed features and targets from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No preprocessed data found at {filepath}. Please preprocess the data first.")
    data = pd.read_csv(filepath)
    X = data.drop(columns=["Target"]).to_numpy()
    y = data["Target"].to_numpy()
    print(f"Preprocessed data loaded from {filepath}.")
    return X, y

# Initialize global variables
balance_eur, crypto_holdings = load_portfolio()
symbol = "BTCUSDT"
model = None  # Placeholder for model

# Initialize API
api = APIInterface(exchange="binance")

def signal_handler(sig, frame):
    """
    Handle interrupt signal (Ctrl+C) to save the model and portfolio.
    """
    print("\nTraining interrupted. Saving the model and portfolio...")
    if model:
        model.save('models/btc_model_interrupted.h5')
    save_portfolio(balance_eur, crypto_holdings)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    """
    Main function to run the learning process for BTC and manage portfolio.
    """
    global balance_eur, crypto_holdings, model  # Access global portfolio variables

    print("Starting Crypto Hedge Fund Model...")

    # Path for preprocessed data
    preprocessed_data_path = "data/preprocessed_data.csv"

    # Fetch or load historical data for BTC
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data...")
        X, y = load_preprocessed_data(preprocessed_data_path)
    else:
        print(f"Fetching historical data for {symbol}...")
        historical_data = api.fetch_historical_data(symbol)

        # Preprocess data for the BTC model
        print(f"Preprocessing data for {symbol}...")
        X, y, scaler = preprocess_data_btc(historical_data)

        # Save preprocessed data for future use
        save_preprocessed_data(X, y, scaler, preprocessed_data_path)

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or build the BTC model
    print(f"Loading or building the model for {symbol}...")
    model = load_or_build_model(X.shape)

    # Train the BTC model with validation data
    model = train_model(model, X_train, y_train, scaler, X_val=X_val, y_val=y_val, epochs=100, batch_size=32)

    # Predict price using the BTC model
    X_test = X[-1].reshape(1, X.shape[1], X.shape[2])  # Last sequence as test input
    predicted_price = predict_price(model, X_test, scaler)

    # Fetch the current price
    current_price = api.fetch_current_price(symbol)

    # Determine if the model predicted the correct direction
    previous_price = historical_data['close'].iloc[-2]
    actual_movement = "UP" if current_price > previous_price else "DOWN"
    predicted_movement = "UP" if predicted_price > previous_price else "DOWN"
    is_correct = actual_movement == predicted_movement

    # Calculate confidence interval
    predictions = [predict_price(model, X_test, scaler) for _ in range(100)]
    confidence_interval = calculate_confidence_interval(predictions)

    # Display results
    print(f"Previous Price: {previous_price:.2f}")
    print(f"Current Price: {current_price:.2f}")
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Actual Movement: {actual_movement}")
    print(f"Predicted Movement: {predicted_movement}")
    print(f"Movement Prediction Correct: {is_correct}")
    print(f"Confidence Interval: {confidence_interval[0]:.2f} to {confidence_interval[1]:.2f}")
    print(f"Difference: {abs(predicted_price - current_price):.2f}")

    # Trading logic
    if predicted_price > confidence_interval[1]:  # Strong BUY signal
        if balance_eur >= current_price:
            crypto_to_buy = balance_eur / current_price
            crypto_holdings += crypto_to_buy
            balance_eur = 0.0  # All balance spent
            print(f"Trading Signal: BUY - Bought {crypto_to_buy:.6f} BTC")
            save_portfolio(balance_eur, crypto_holdings)
        else:
            print("Trading Signal: BUY - Insufficient balance to buy BTC")
    elif predicted_price < confidence_interval[0]:  # Strong SELL signal
        if crypto_holdings > 0:
            earnings = crypto_holdings * current_price
            balance_eur += earnings
            print(f"Trading Signal: SELL - Sold {crypto_holdings:.6f} BTC for {earnings:.2f} EUR")
            crypto_holdings = 0.0  # All holdings sold
            save_portfolio(balance_eur, crypto_holdings)
        else:
            print("Trading Signal: SELL - No crypto holdings to sell")
    else:
        print("Trading Signal: HOLD - No action taken")

    # Portfolio summary
    print("Portfolio Summary:")
    print(f"Balance (EUR): {balance_eur:.2f}")
    print(f"Crypto Holdings (BTC): {crypto_holdings:.6f}")

    # Log output
    print("BTC learning process completed.")

if __name__ == "__main__":
    main()
