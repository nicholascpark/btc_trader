import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings("ignore")

class TradingDataset(Dataset):
    def __init__(self, sequences, future_prices):
        self.sequences = sequences
        self.future_prices = future_prices
        
        price_based_features = [
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'volume_vwap', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
            'volatility_kcc', 'volatility_kch', 'volatility_kcl',
            'volatility_dcl', 'volatility_dch', 'volatility_dcm',
            'volatility_atr', 'trend_sma_fast', 'trend_sma_slow',
            'trend_ema_fast', 'trend_ema_slow', 'trend_ichimoku_a',
            'trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',
            'trend_psar_up', 'trend_psar_down', 'others_dr', 'others_dlr',
            'price_pct_change', 'price_log_return', 'price_momentum'
        ]
        self.price_based_features = [f for f in price_based_features if f in sequences[0].columns]
        self.indicator_features = [col for col in sequences[0].columns if col not in self.price_based_features]
        
        # Scale price-based features across unique timestamps
        all_timestamps = np.unique([seq.index for seq in sequences])
        price_data = np.vstack([seq.loc[seq.index.isin(all_timestamps), self.price_based_features].values for seq in sequences])
        self.price_scaler = StandardScaler()
        self.price_scaler.fit(price_data)
        
        # Scale indicator features across samples (per time step)
        self.indicator_scalers = [StandardScaler() for _ in range(len(sequences[0]))]
        for i in range(len(sequences[0])):
            indicator_data = np.array([seq.iloc[i][self.indicator_features].values for seq in sequences])
            self.indicator_scalers[i].fit(indicator_data)
        
        # Create a separate scaler for future prices
        self.future_price_scaler = StandardScaler()
        self.future_price_scaler.fit(np.array(future_prices).reshape(-1, 1))
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Scale price-based features
        scaled_price_data = self.price_scaler.transform(sequence[self.price_based_features])
        
        # Scale indicator features
        scaled_indicator_data = np.array([
            self.indicator_scalers[i].transform(sequence.iloc[i][self.indicator_features].values.reshape(1, -1)).flatten()
            for i in range(len(sequence))
        ])
        
        # Combine scaled data
        scaled_sequence = np.hstack((scaled_price_data, scaled_indicator_data))
        
        # Scale future price using the separate future price scaler
        future_price = self.future_price_scaler.transform([[self.future_prices[idx]]])[0]
        
        return torch.FloatTensor(scaled_sequence), torch.FloatTensor(future_price)
    
    def __len__(self):
        return len(self.sequences)
    

# In your main code, before training:
def check_data(loader):
    for sequences, future_prices in loader:
        if torch.isnan(sequences).any() or torch.isnan(future_prices).any():
            print("NaN values detected in the data!")
            return False
    return True


def safe_divide(a, b, fill_value=0):
    return np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)

def safe_log(x):
    return np.log(np.where(x > 0, x, np.nan))

def create_dataset(symbol, start_date, end_date, window_size=30):
    # Fetch historical data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Ensure we have data
    if data.empty:
        raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
    
    # Forward fill any missing values
    data = data.ffill()
    
    # Calculate all technical indicators
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )
    
    # Custom indicators
    data['SMA_short'] = data['Close'].rolling(window=10, min_periods=1).mean()
    data['SMA_long'] = data['Close'].rolling(window=30, min_periods=1).mean()
    data['EMA_short'] = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    data['EMA_long'] = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    
    macd = MACD(data['Close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    
    rsi = RSIIndicator(data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    
    bb = BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_middle'] = bb.bollinger_mavg()
    
    stoch = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stoch_k'] = stoch.stoch()
    data['Stoch_d'] = stoch.stoch_signal()
    
    # Calculate price-based features safely
    data['price_pct_change'] = safe_divide(data['Close'] - data['Close'].shift(1), data['Close'].shift(1))
    data['price_log_return'] = safe_log(data['Close'] / data['Close'].shift(1))
    data['price_momentum'] = data['Close'] - data['Close'].shift(5)
    
    # Create labels based on future price movements
    data['future_return'] = data['Close'].pct_change(periods=1).shift(-1)
    
    # Drop any remaining NaN values
    data = data.dropna()
    
    # Create sequences of data
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequence = data.iloc[i:i+window_size]
        label = data.iloc[i+window_size]['Close']
        sequences.append(sequence)
        labels.append(label)
    
    print(f"Number of features: {len(data.columns)}")
    print(f"Number of sequences: {len(sequences)}")
    
    return sequences, labels, data.columns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, hidden_size)
        attn_weights = torch.softmax(self.attention(x), dim=0)
        context = torch.sum(x * attn_weights, dim=0)
        return context


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1000, 1, hidden_size))
        self.layers = nn.ModuleList([SimpleAttention(hidden_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
        x = self.embedding(x)  # (seq_len, batch_size, hidden_size)
        x = x + self.position_encoding[:x.size(0), :, :]
        
        for layer in self.layers:
            x = layer(x)
        
        output = self.fc(x)
        return output.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, future_prices in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences.transpose(0, 1))
            loss = criterion(outputs, future_prices)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected. Skipping batch.")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, future_prices in val_loader:
                outputs = model(sequences.transpose(0, 1))
                loss = criterion(outputs, future_prices)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small. Stopping training.")
            break


# Assume sequences, future_prices are obtained from the modified create_dataset function
sequences, future_prices, feature_names = create_dataset("BTC-USD", "2022-01-01", "2024-03-31")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(sequences, future_prices, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = TradingDataset(X_train, y_train)
test_dataset = TradingDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize and train the model
input_dim = X_train[0].shape[1]  # Number of features
model = TransformerModel(input_dim)
train_model(model, train_loader, test_loader)

# Check data before training
if check_data(train_loader) and check_data(test_loader):
    print("Data looks good. Starting training...")
    train_model(model, train_loader, test_loader)
else:
    print("Please check your data preprocessing steps.")

def predict_and_act(model, sequence, current_price, price_scaler, future_price_scaler, threshold=0.04):
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(1)
        predicted_price_scaled = model(sequence_tensor).item()
        predicted_price = future_price_scaler.inverse_transform([[predicted_price_scaled]])[0][0]
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > threshold:
            return "buy", predicted_price
        elif price_change < -threshold:
            return "sell", predicted_price
        else:
            return "hold", predicted_price


# Example prediction
last_sequence = sequences[-1]
last_sequence_scaled = train_dataset.price_scaler.transform(last_sequence[train_dataset.price_based_features])
current_price = last_sequence['Close'].iloc[-1]
action, predicted_price = predict_and_act(model, last_sequence_scaled, current_price, train_dataset.price_scaler)
print(f"Predicted action for the last sequence: {action}")
print(f"Current price: ${current_price:.2f}, Predicted price: ${predicted_price:.2f}")

# Calculate accuracy on test set
threshold = 0.04
correct = 0
total = 0
true_prices = []
predicted_prices = []

# In your accuracy calculation loop:
for sequence, future_price in test_loader:
    current_price = test_dataset.price_scaler.inverse_transform(sequence[:, -1, sequence.shape[2]//2].numpy().reshape(-1, 1)).flatten()
    
    predicted_action, predicted_price = predict_and_act(model, sequence.squeeze(), current_price[0], 
                                                        test_dataset.price_scaler, 
                                                        test_dataset.future_price_scaler)
    
    actual_price = test_dataset.future_price_scaler.inverse_transform(future_price.numpy().reshape(-1, 1)).flatten()[0]
    
    actual_price_change = (actual_price - current_price) / current_price
    actual_action = np.where(actual_price_change > threshold, "buy",
                             np.where(actual_price_change < -threshold, "sell", "hold"))[0]
    
    correct += (predicted_action == actual_action)
    total += 1
    
    true_prices.append(actual_price)
    predicted_prices.append(predicted_price)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Calculate additional metrics
mse = np.mean((np.array(true_prices) - np.array(predicted_prices))**2)
mae = np.mean(np.abs(np.array(true_prices) - np.array(predicted_prices)))
mape = np.mean(np.abs((np.array(true_prices) - np.array(predicted_prices)) / np.array(true_prices))) * 100

print(f"Mean Squared Error: ${mse:.2f}")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(true_prices, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.title('True vs Predicted Bitcoin Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price ($)')
plt.show()