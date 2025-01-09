# main.py

from src.agents.analysis_agent import AnalysisAgent
from src.models.btc_model import preprocess_data_btc, build_btc_model
from src.tools.api_interface import APIInterface

def main():
    """
    Main function to run the learning process for BTC.
    """
    print("Starting Crypto Hedge Fund Model...")

    # Initialize API and agent
    api = APIInterface(exchange="binance")
    agent = AnalysisAgent()

    # Fetch historical data for BTC
    symbol = "BTCUSDT"
    print(f"Fetching historical data for {symbol}...")
    historical_data = api.fetch_historical_data(symbol)

    # Preprocess data for the BTC model
    print(f"Preprocessing data for {symbol}...")
    X, y, scaler = preprocess_data_btc(historical_data)

    # Build the BTC model
    print(f"Building the model for {symbol}...")
    model = build_btc_model(X.shape)

    # Train the BTC model
    print(f"Training the model for {symbol}...")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # Predict price using the BTC model
    print(f"Predicting future prices for {symbol}...")
    predicted_price = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    # Fetch the current price
    current_price = api.fetch_current_price(symbol)
    print(f"Current Price: {current_price:.2f}")
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Difference: {abs(predicted_price - current_price):.2f}")

    # Log output
    print("BTC learning process completed.")

if __name__ == "__main__":
    main()
