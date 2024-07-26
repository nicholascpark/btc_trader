import pandas as pd
import numpy as np
from ta import add_all_ta_features
from datetime import datetime, timedelta
import yfinance as yf

def create_dataset(symbol, start_date, end_date, window_size=30):
    # Fetch historical data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate all technical indicators
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )
    
    # Calculate price-based features
    data['price_pct_change'] = data['Close'].pct_change()
    data['price_log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['price_momentum'] = data['Close'] - data['Close'].shift(5)
    
    # Create labels based on future price movements
    data['future_return'] = data['Close'].pct_change(periods=1).shift(-1)
    data['action'] = np.where(data['future_return'] > 0.01, 'buy',
                     np.where(data['future_return'] < -0.01, 'sell', 'hold'))
    
    # Create sequences of data
    sequences = []
    labels = []
    
    for i in range(len(data) - window_size):
        sequence = data.iloc[i:i+window_size]
        label = data.iloc[i+window_size]['action']
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels, data.columns

# Example usage
symbol = "BTC-USD"
start_date = "2022-01-01"
end_date = "2024-03-31"

sequences, labels, feature_names = create_dataset(symbol, start_date, end_date)

print(f"Dataset created with {len(sequences)} samples")
print(f"Sample sequence shape: {sequences[0].shape}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Number of features: {len(feature_names)}")
print("Features:", feature_names.tolist())