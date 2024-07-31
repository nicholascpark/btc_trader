import torch
import numpy as np
from torch.utils.data import DataLoader
from data.data_loader import create_dataset
from data.dataset import TradingDataset
from models.transformer import TransformerModel
from sklearn.model_selection import train_test_split
from config import MODEL_CONFIG, EVALUATION_CONFIG
from predict import predict_and_act

def evaluate_model(model, test_loader, test_dataset):
    threshold = EVALUATION_CONFIG['threshold']
    correct = 0
    total = 0
    true_prices = []
    predicted_prices = []

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

    accuracy = 100 * correct / total
    mse = np.mean((np.array(true_prices) - np.array(predicted_prices))**2)
    mae = np.mean(np.abs(np.array(true_prices) - np.array(predicted_prices)))
    mape = np.mean(np.abs((np.array(true_prices) - np.array(predicted_prices)) / np.array(true_prices))) * 100

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    return accuracy, mse, mae, mape

def main():
    # Load the trained model
    model = TransformerModel(EVALUATION_CONFIG['input_dim'], **MODEL_CONFIG)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Load and prepare the test data
    sequences, future_prices, _ = create_dataset(EVALUATION_CONFIG['symbol'], EVALUATION_CONFIG['start_date'], EVALUATION_CONFIG['end_date'])
    _, X_test, _, y_test = train_test_split(sequences, future_prices, test_size=0.2, random_state=42)
    
    test_dataset = TradingDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=EVALUATION_CONFIG['batch_size'], shuffle=False)

    # Evaluate the model
    evaluate_model(model, test_loader, test_dataset)

if __name__ == "__main__":
    main()