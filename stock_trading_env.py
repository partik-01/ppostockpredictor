# stock_trading_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
import ta

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, max_shares=10, transaction_fee_percent=0.001):
        super(StockTradingEnv, self).__init__()
        
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        print("Starting data preprocessing...")
        print("Original DataFrame head:")
        print(df.head())
        print("\nOriginal DataFrame types:")
        print(df.dtypes)
        
        try:
            # Drop the S.N. column if it exists
            if 'S.N.' in df.columns:
                df = df.drop('S.N.', axis=1)
            
            # Helper function to safely convert numeric strings with commas
            def safe_numeric_convert(x):
                if isinstance(x, str):
                    return float(x.replace(',', '').replace('%', ''))
                return float(x)
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Ltp', 'Qty', 'Turnover']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(safe_numeric_convert)
            
            # Handle percentage change column
            if '% Change' in df.columns:
                df['% Change'] = df['% Change'].apply(safe_numeric_convert)
            
            # Convert date format
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            
            # Sort by date
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            print("\nProcessed DataFrame head:")
            print(df.head())
            print("\nProcessed DataFrame types:")
            print(df.dtypes)
            
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            print("DataFrame columns:", df.columns.tolist())
            import traceback
            traceback.print_exc()
            raise
            
        if len(df) < 100:
            raise ValueError("Not enough historical data. Need at least 100 data points.")
        
        self.df = df
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.transaction_fee_percent = transaction_fee_percent
        
        # Calculate technical indicators
        self.df['Returns'] = self.df['Ltp'].pct_change().fillna(0)
        self.df['MACD'] = ta.trend.MACD(self.df['Ltp']).macd().fillna(0)
        self.df['Signal'] = ta.trend.MACD(self.df['Ltp']).macd_signal().fillna(0)
        self.df['RSI'] = ta.momentum.RSIIndicator(self.df['Ltp']).rsi().fillna(50)
        self.df['SMA_20'] = self.df['Ltp'].rolling(window=20, min_periods=1).mean()
        self.df['SMA_50'] = self.df['Ltp'].rolling(window=50, min_periods=1).mean()
        self.df['EMA_20'] = ta.trend.EMAIndicator(self.df['Ltp'], window=20).ema_indicator().fillna(0)
        
        # Add volatility measures
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std().fillna(0)
        self.df['ATR'] = ta.volatility.AverageTrueRange(self.df['High'], self.df['Low'], self.df['Ltp']).average_true_range().fillna(0)
        
        # Volume indicators
        self.df['Volume_SMA'] = self.df['Qty'].rolling(window=20).mean().fillna(0)
        self.df['Volume_Ratio'] = (self.df['Qty'] / self.df['Volume_SMA']).fillna(1)
        
        # Momentum indicators
        self.df['MFI'] = ta.volume.MFIIndicator(self.df['High'], self.df['Low'], self.df['Ltp'], self.df['Qty']).money_flow_index().fillna(50)
        
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.max_steps = len(df) - 1
        self.current_step = None
        self.balance = None
        self.shares_held = None
        self.purchase_prices = None
        self.portfolio_history = None
        self.window_size = 20

    def calculate_reward(self, action, current_price, next_price, shares_bought=0, shares_sold=0):
        """Enhanced reward function with multiple components"""
        reward = 0
        
        # Time decay factor to encourage earlier profitable actions
        time_factor = 0.99 ** (self.current_step / self.max_steps)
        
        # Calculate smoothed price change
        price_change_pct = (next_price - current_price) / current_price
        smooth_price_change = np.tanh(price_change_pct)  # Limit extreme values
        
        # Calculate portfolio value change
        portfolio_value = self.balance + (self.shares_held * current_price)
        prev_portfolio_value = self.portfolio_history[-1]
        portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Base reward calculation
        if action == 0:  # Hold
            if self.shares_held > 0:
                reward = smooth_price_change * self.shares_held
            else:
                reward = -smooth_price_change  # Reward for avoiding losses in downtrend
        
        elif action == 1:  # Buy
            if shares_bought > 0:
                # Transaction cost penalty
                reward -= self.transaction_fee_percent
                
                # Price momentum component
                reward += smooth_price_change
                
                # Buy timing reward based on technical indicators
                current_rsi = self.df.iloc[self.current_step]['RSI']
                if current_rsi < 30:  # Oversold condition
                    reward += 0.1
                
                # Volume consideration
                volume_ratio = self.df.iloc[self.current_step]['Volume_Ratio']
                if volume_ratio > 1.5:  # High volume buy
                    reward += 0.1
        
        elif action == 2:  # Sell
            if shares_sold > 0:
                # Calculate profit/loss percentage
                avg_purchase_price = np.mean(self.purchase_prices) if self.purchase_prices else current_price
                profit_pct = (current_price - avg_purchase_price) / avg_purchase_price
                reward += np.tanh(profit_pct)  # Bounded profit reward
                
                # Sell timing reward based on technical indicators
                current_rsi = self.df.iloc[self.current_step]['RSI']
                if current_rsi > 70:  # Overbought condition
                    reward += 0.1
        
        # Portfolio value change component
        reward += np.tanh(portfolio_change)
        
        # Scale reward based on position size and time
        position_size = (self.shares_held * current_price) / portfolio_value if portfolio_value > 0 else 0
        reward *= (1 + position_size) * time_factor
        
        return reward

    def _next_observation(self):
        # Get the current window of data
        obs = np.array([
            self.df.iloc[self.current_step]['Ltp'],
            self.df.iloc[self.current_step]['Returns'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['Signal'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['SMA_20'],
            self.df.iloc[self.current_step]['SMA_50'],
            self.df.iloc[self.current_step]['EMA_20'],
            self.df.iloc[self.current_step]['Volatility'],
            self.df.iloc[self.current_step]['ATR'],
            self.df.iloc[self.current_step]['Volume_Ratio'],
            self.df.iloc[self.current_step]['MFI'],
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held / self.max_shares,   # Normalized position size
            self.portfolio_value() / self.initial_balance  # Normalized portfolio value
        ], dtype=np.float32)
        
        # Normalize the observation values
        obs[0] = obs[0] / 1000  # Price
        obs[4] = obs[4] / 100   # RSI
        obs[5] = obs[5] / 1000  # SMA_20
        obs[6] = obs[6] / 1000  # SMA_50
        
        return obs

    def portfolio_value(self):
        return self.balance + (self.shares_held * self.current_price())

    def current_price(self):
        return float(self.df.iloc[self.current_step]['Ltp'])

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.purchase_prices = []
        self.portfolio_history = [self.initial_balance]
        return self._next_observation()

    def step(self, action):
        self.current_step += 1
        
        current_price = self.current_price()
        next_price = float(self.df.iloc[min(self.current_step + 1, len(self.df)-1)]['Ltp'])
        
        shares_bought = 0
        shares_sold = 0
        
        # Execute action
        if action == 1 and self.balance >= current_price and self.shares_held < self.max_shares:  # Buy
            shares_to_buy = min(self.max_shares - self.shares_held, 
                              self.balance // current_price)
            shares_to_buy = int(shares_to_buy)
            shares_bought = shares_to_buy
            
            purchase_cost = shares_to_buy * current_price * (1 + self.transaction_fee_percent)
            self.balance -= purchase_cost
            self.shares_held += shares_to_buy
            self.purchase_prices.extend([current_price] * shares_to_buy)
            
        elif action == 2 and self.shares_held > 0:  # Sell
            shares_to_sell = self.shares_held
            shares_sold = shares_to_sell
            
            sell_revenue = shares_to_sell * current_price * (1 - self.transaction_fee_percent)
            self.balance += sell_revenue
            self.shares_held = 0
            self.purchase_prices = []
        
        # Calculate reward
        reward = self.calculate_reward(action, current_price, next_price, shares_bought, shares_sold)
        
        # Update portfolio history
        self.portfolio_history.append(self.portfolio_value())
        
        # Calculate done flag
        done = self.current_step >= self.max_steps - 1
        
        info = {
            'portfolio_value': self.portfolio_value(),
            'shares_held': self.shares_held,
            'balance': self.balance,
            'current_price': current_price,
            'action': action,
            'reward': reward
        }
        
        return self._next_observation(), reward, done, info

    def render(self, mode='human'):
        profit = self.portfolio_value() - self.initial_balance
        
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Current price: ${self.current_price():.2f}')
        print(f'Portfolio value: ${self.portfolio_value():.2f}')
        print(f'Profit: ${profit:.2f} ({(profit/self.initial_balance)*100:.2f}%)')
        print(f'Current RSI: {self.df.iloc[self.current_step]["RSI"]:.2f}')
        print('-' * 50)