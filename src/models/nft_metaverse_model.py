# src/models/nft_metaverse_model.py

def preprocess_data_nft(df, sentiment_data):
    """
    Preprocess price and sentiment data for NFT/metaverse token analysis.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    scaled_sentiment = scaler.fit_transform(np.array(sentiment_data).reshape(-1, 1))

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_prices) - 1):
        price_window = scaled_prices[i - LOOKBACK:i, 0]
        sentiment_window = scaled_sentiment[i - LOOKBACK:i, 0]
        combined_features = np.column_stack((price_window, sentiment_window))
        X.append(combined_features)
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 2))
    return X, y, scaler

def build_nft_model(input_shape):
    """
    Build an NFT/metaverse token model integrating sentiment and price data.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
