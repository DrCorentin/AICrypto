# src/agents/analysis_agent.py

class AnalysisAgent:
    def __init__(self, exchange="binance"):
        self.api = APIInterface(exchange)

    def identify_high_potential_cryptos(self, symbols):
        """
        Identify cryptocurrencies with high ROI potential.
        """
        recommendations = []
        for symbol in symbols:
            print(f"Processing {symbol}...")
            # Fetch and preprocess data
            df = self.api.fetch_historical_data(symbol)
            X, y, scaler = preprocess_data(df)
            # Build and train model
            model = build_model(X.shape)
            print(f"Training model for {symbol}...")
            model = train_model(model, X, y)
            # Predict future price
            print(f"Predicting price for {symbol}...")
            predicted_price = predict_price(model, df, scaler, log=True)
            current_price = self.api.fetch_current_price(symbol)
            # Calculate expected ROI
            roi = (predicted_price - current_price) / current_price
            recommendations.append({
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_roi': roi
            })
            print(f"- Current Price: {current_price:.2f}")
            print(f"- Predicted Price: {predicted_price:.2f}")
            print(f"- Expected ROI: {roi:.5f}")
        return recommendations
