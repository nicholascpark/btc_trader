MODEL_CONFIG = {
    'hidden_size': 32,
    'num_layers': 1
}

TRAINING_CONFIG = {
    'symbol': "BTC-USD",
    'start_date': "2022-01-01",
    'end_date': "2024-03-31",
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4
}

PREDICTION_CONFIG = {
    'threshold': 0.04,
    'input_dim': 106, 
    'symbol': "BTC-USD",
    'lookback_days': 100,
    'window_size': 30
}

EVALUATION_CONFIG = {
    'symbol': "BTC-USD",
    'start_date': "2022-01-01",
    'end_date': "2024-03-31",
    'batch_size': 32,
    'threshold': 0.04,
    'input_dim': 106
}