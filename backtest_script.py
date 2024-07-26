from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from ml_trader import AdvancedMLTrader, ALPACA_CREDS

def run_backtest(start_date, end_date):
    broker = Alpaca(ALPACA_CREDS)
    strategy = AdvancedMLTrader(name='mlstrat', broker=broker, parameters={"symbol": "BTC-USD", "cash_at_risk": 0.5})
    
    backtest = strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={"symbol": "BTC-USD", "cash_at_risk": 0.5}
    )
    
    return backtest, AdvancedMLTrader.num_trades

def backtest_multiple_periods(start_year, end_year):
    results = {}
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = datetime(year, month, 1)
            end_date = start_date + timedelta(days=30)  # Approximately one month
            
            if end_date > datetime.now():
                break
            
            print(f"Backtesting for period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
            backtest_result, num_trades = run_backtest(start_date, end_date)
            
            results[start_date.strftime('%Y-%m')] = {
                'backtest_result': backtest_result,
                'num_trades': num_trades
            }
            
            AdvancedMLTrader.num_trades = 0  # Reset num_trades for the next period
    
    return results

# Run backtests for the past few years
start_year = 2022
end_year = 2023

results = backtest_multiple_periods(start_year, end_year)

# Print or analyze results
for period, data in results.items():
    print(f"\nPeriod: {period}")
    print(f"Number of trades: {data['num_trades']}")
    print(f"Backtest results: {data['backtest_result']}")
    # Add more detailed analysis of backtest results as needed