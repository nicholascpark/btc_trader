import matplotlib.pyplot as plt

def plot_price_comparison(true_prices, predicted_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, label='True Prices')
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.legend()
    plt.title('True vs Predicted Bitcoin Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.show()