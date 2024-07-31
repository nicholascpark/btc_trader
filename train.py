import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import create_dataset
from data.dataset import TradingDataset
from models.transformer import TransformerModel
from sklearn.model_selection import train_test_split
from config import MODEL_CONFIG, TRAINING_CONFIG

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-5)
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

def main():
    sequences, future_prices, feature_names = create_dataset(TRAINING_CONFIG['symbol'], TRAINING_CONFIG['start_date'], TRAINING_CONFIG['end_date'])
    X_train, X_test, y_train, y_test = train_test_split(sequences, future_prices, test_size=0.2, random_state=42)
    
    train_dataset = TradingDataset(X_train, y_train)
    test_dataset = TradingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    
    input_dim = X_train[0].shape[1]
    model = TransformerModel(input_dim, **MODEL_CONFIG)
    train_model(model, train_loader, test_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    main()