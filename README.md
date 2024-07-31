# Trading Bot

This project implements a machine learning-based trading bot that predicts cryptocurrency prices and suggests trading actions.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Modules](#modules)
6. [Contributing](#contributing)
7. [License](#license)

## Project Structure

```
project_root/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── dataset.py
│
├── models/
│   ├── __init__.py
│   ├── transformer.py
│   └── attention.py
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── train.py
├── predict.py
├── evaluate.py
├── config.py
├── main.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

Edit the `config.py` file to adjust the model parameters, training settings, and evaluation criteria:

- `MODEL_CONFIG`: Set the model architecture parameters
- `TRAINING_CONFIG`: Configure training data source, date range, and hyperparameters
- `PREDICTION_CONFIG`: Set up prediction parameters
- `EVALUATION_CONFIG`: Define evaluation settings

## Usage

Use the `main.py` script to run the bot in different modes:

1. Training mode:
   ```
   python main.py --mode train
   ```

2. Prediction mode:
   ```
   python main.py --mode predict
   ```

3. Evaluation mode:
   ```
   python main.py --mode evaluate
   ```

You can also use the short form `-m` instead of `--mode`.

## Modules

- `data/`: Contains scripts for data loading and dataset creation
- `models/`: Implements the Transformer model and attention mechanism
- `utils/`: Includes utility functions for preprocessing and visualization
- `train.py`: Handles the model training process
- `predict.py`: Makes predictions using the trained model
- `evaluate.py`: Evaluates the model's performance
- `config.py`: Centralizes configuration parameters
- `main.py`: Entry point for running the bot in different modes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
