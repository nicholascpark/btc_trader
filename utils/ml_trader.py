import logging
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

class MLTrader(Strategy):
    num_trades = 0

    def initialize(self, symbol:str="BTC-USD", cash_at_risk:float=0.5):
        # Trading parameters
        self.symbol = symbol
        self.sleeptime = "1H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.take_profit_percentage = 0.05
        self.stop_loss_percentage = 0.03

        # Sentiment analysis parameters
        self.sentiment_threshold = 0.95
        self.n_days_for_news = 1

        # Technical indicator parameters
        self.sma_short, self.sma_long = 10, 30
        self.ema_short, self.ema_long = 12, 26
        self.rsi_period = 14
        self.rsi_overbought, self.rsi_oversold = 70, 30
        self.macd_fast, self.macd_slow, self.macd_signal = 12, 26, 9
        self.bb_period, self.bb_std = 20, 2
        self.stoch_k, self.stoch_d = 14, 3
        self.stoch_overbought, self.stoch_oversold = 80, 20

        # API setup
        self.api = REST(
            base_url="https://paper-api.alpaca.markets",
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_API_SECRET")
        )

    @staticmethod
    def to_scalar(x):
        return x.iloc[-1] if isinstance(x, pd.Series) else x

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 3)
        return cash, last_price, quantity

    def get_sentiment(self):
        today = self.get_datetime()
        n_days_prior = today - Timedelta(days=self.n_days_for_news)
        try:
            news = self.api.get_news(symbol="BTC/USD", start=n_days_prior.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
            news = [ev.__dict__["_raw"]["headline"] for ev in news]
            return estimate_sentiment(news)
        except Exception as e:
            logging.error(f"Error fetching sentiment: {e}")
            return None, None

    def get_technical_indicators(self):
        bars = self.get_historical_prices(self.symbol, 100)
        print("bars:", bars)
        df = bars.df

        indicators = {}
        # Calculate custom indicators
        indicators['SMA_short'] = df['close'].rolling(window=self.sma_short).mean()
        indicators['SMA_long'] = df['close'].rolling(window=self.sma_long).mean()
        indicators['EMA_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        indicators['EMA_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        
        # MACD
        macd = MACD(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)
        indicators['MACD'] = macd.macd()
        indicators['MACD_signal'] = macd.macd_signal()
        
        # RSI
        indicators['RSI'] = RSIIndicator(df['close'], self.rsi_period).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], self.bb_period, self.bb_std)
        indicators['BB_middle'] = bb.bollinger_mavg()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], self.stoch_k, self.stoch_d)
        indicators['Stoch_k'] = stoch.stoch()
        indicators['Stoch_d'] = stoch.stoch_signal()

        # Add all TA features
        ta_features = add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="volume",
            fillna=True
        )

        return pd.concat([df, pd.DataFrame(indicators), ta_features], axis=1).iloc[-1]

    def check_buy_conditions(self, indicators, sentiment, probability):
        return (
            sentiment == "positive" and 
            probability > self.sentiment_threshold and
            self.to_scalar(indicators['SMA_short']) > self.to_scalar(indicators['SMA_long']) and
            self.to_scalar(indicators['EMA_short']) > self.to_scalar(indicators['EMA_long']) and
            self.to_scalar(indicators['RSI']) < self.rsi_overbought and
            self.to_scalar(indicators['MACD']) > self.to_scalar(indicators['MACD_signal']) and
            self.to_scalar(indicators['close']) > self.to_scalar(indicators['BB_middle']) and
            self.to_scalar(indicators['Stoch_k']) > self.to_scalar(indicators['Stoch_d']) and
            self.to_scalar(indicators['Stoch_k']) < self.stoch_overbought and
            self.to_scalar(indicators['trend_ichimoku_a']) < self.to_scalar(indicators['close']) and
            self.to_scalar(indicators['trend_ichimoku_b']) < self.to_scalar(indicators['close'])
        )

    def check_sell_conditions(self, indicators, sentiment, probability):
        return (
            sentiment == "negative" and 
            probability > self.sentiment_threshold and
            self.to_scalar(indicators['SMA_short']) < self.to_scalar(indicators['SMA_long']) and
            self.to_scalar(indicators['EMA_short']) < self.to_scalar(indicators['EMA_long']) and
            self.to_scalar(indicators['RSI']) > self.rsi_oversold and
            self.to_scalar(indicators['MACD']) < self.to_scalar(indicators['MACD_signal']) and
            self.to_scalar(indicators['close']) < self.to_scalar(indicators['BB_middle']) and
            self.to_scalar(indicators['Stoch_k']) < self.to_scalar(indicators['Stoch_d']) and
            self.to_scalar(indicators['Stoch_k']) > self.stoch_oversold and
            self.to_scalar(indicators['trend_ichimoku_a']) > self.to_scalar(indicators['close']) and
            self.to_scalar(indicators['trend_ichimoku_b']) > self.to_scalar(indicators['close'])
        )

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        try:
            indicators = self.get_technical_indicators()
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return
        
        # print("indicators:", indicators)

        if cash > last_price and probability is not None and sentiment is not None:
            current_position = self.get_position(self.symbol)
            
            if self.check_buy_conditions(indicators, sentiment, probability):
                if current_position is None or current_position.quantity < 0:
                    self.submit_buy_order(quantity, last_price)
            elif self.check_sell_conditions(indicators, sentiment, probability):
                if current_position is None or current_position.quantity > 0:
                    self.submit_sell_order(quantity, last_price)

    def submit_buy_order(self, quantity, last_price):
        self.sell_all()
        order = self.create_order(
            self.symbol,
            quantity,
            "buy",
            type="bracket",
            take_profit_price=last_price * (1 + self.take_profit_percentage),
            stop_loss_price=last_price * (1 - self.stop_loss_percentage)
        )
        self.submit_order(order)
        self.last_trade = "buy"
        MLTrader.num_trades += 1  # Increment trade counter


    def submit_sell_order(self, quantity, last_price):
        self.sell_all()
        order = self.create_order(
            self.symbol,
            quantity,
            "sell",
            type="bracket",
            take_profit_price=last_price * (1 - self.take_profit_percentage),
            stop_loss_price=last_price * (1 + self.stop_loss_percentage)
        )
        self.submit_order(order)
        self.last_trade = "sell"
        MLTrader.num_trades += 1  # Increment trade counter


start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 31)

broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbol": "BTC-USD", "cash_at_risk": .5})

backtest =strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "BTC-USD", "cash_at_risk": .5}
)
print(backtest)
print("num_trades:", MLTrader.num_trades)

# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
