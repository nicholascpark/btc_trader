import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler


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