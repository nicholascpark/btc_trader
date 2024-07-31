import yfinance as yf
from datetime import datetime, timedelta
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from utils.preprocessing import safe_divide, safe_log


def create_dataset(symbol, start_date, end_date, window_size=30, burn_in_period=100):
    # Calculate the adjusted start date to account for the burn-in period
    adjusted_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=burn_in_period)).strftime("%Y-%m-%d")
    
    # Fetch historical data with the adjusted start date
    data = yf.download(symbol, start=adjusted_start_date, end=end_date)
    
    # Ensure we have data
    if data.empty:
        raise ValueError(f"No data available for {symbol} between {adjusted_start_date} and {end_date}")
    
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
    
    # Remove the burn-in period data
    data = data[data.index >= start_date]
    
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