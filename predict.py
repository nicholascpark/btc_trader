import torch
import numpy as np
from models.transformer import TransformerModel
from data.dataset import TradingDataset
from data.data_loader import create_dataset
from config import MODEL_CONFIG, PREDICTION_CONFIG
from utils.visualization import plot_price_comparison
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def predict_and_act(model, sequence, current_price, price_scaler, future_price_scaler, threshold=PREDICTION_CONFIG['threshold']):
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

def get_latest_data(symbol, lookback_days=PREDICTION_CONFIG['lookback_days']):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def prepare_sequence(data, window_size=PREDICTION_CONFIG['window_size']):
    sequences, _, _ = create_dataset(None, None, None, window_size=window_size)
    return sequences[-1]  # Return the last sequence

def main():
    # Load the trained model
    model = TransformerModel(PREDICTION_CONFIG['input_dim'], **MODEL_CONFIG)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Get the latest data
    latest_data = get_latest_data(PREDICTION_CONFIG['symbol'])
    
    # Prepare the sequence for prediction
    sequence = prepare_sequence(latest_data)
    
    # Create a dummy dataset to use its scalers
    dummy_dataset = TradingDataset([sequence], [0])
    
    # Get the current price
    current_price = latest_data['Close'].iloc[-1]
    
    # Make prediction
    action, predicted_price = predict_and_act(
        model, 
        sequence, 
        current_price, 
        dummy_dataset.price_scaler, 
        dummy_dataset.future_price_scaler
    )
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted price: ${predicted_price:.2f}")
    print(f"Recommended action: {action}")
    
    # Visualize recent price history and prediction
    recent_prices = latest_data['Close'].values
    predicted_prices = np.append(recent_prices[:-1], predicted_price)
    plot_price_comparison(recent_prices, predicted_prices)

if __name__ == "__main__":
    main()