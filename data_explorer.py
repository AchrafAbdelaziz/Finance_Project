import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not installed - using matplotlib for plots")

class StockDataExplorer:
    def __init__(self, ticker, period="6mo", interval="1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df = None
        self.initial_capital = None
        self.backtest_results = None

    # Data Pipeline Methods
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance with error handling"""
        try:
            stock = yf.Ticker(self.ticker)
            self.df = stock.history(
                period=self.period,
                interval=self.interval,
                auto_adjust=False  # Keep 'Adj Close' column
            )
            if self.df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            print(f" Successfully fetched {len(self.df)} records for {self.ticker}")
            return self.df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    def clean_data(self):
        """Clean data and handle missing values"""
        if self.df is not None:
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            print(f" Removed {initial_rows - len(self.df)} rows with missing values")
        else:
            print(" No data to clean. Call fetch_data() first")

    # Feature Engineering
    # def feature_engineering(self, ema_windows=[12, 26], rsi_window=14, 
    #                       macd_fast=12, macd_slow=26, macd_signal=9):
    #     """Add technical indicators to DataFrame"""
    #     if self.df is None:
    #         print("⚠️ No data. Call fetch_data() first")
    #         return
        
        
    #     self.df['Return'] = self.df['Adj Close'].pct_change()
        
    #     ema_windows = [12, 26]  # or any other list of periods
    #     for window in ema_windows:
    #         self.df[f'EMA{window}'] = self.df['Adj Close'].ewm(span=window, adjust=False).mean()
    #     # for window in ma_windows:
    #     #     self.df[f'MA{window}'] = self.df['Adj Close'].rolling(window).mean()
        
        
    #     self.df['Volatility20'] = self.df['Return'].rolling(20).std()
        
    #     # RSI
    #     delta = self.df['Adj Close'].diff()
    #     gain = delta.clip(lower=0)
    #     loss = -delta.clip(upper=0)
    #     avg_gain = gain.rolling(rsi_window).mean()
    #     avg_loss = loss.rolling(rsi_window).mean()
    #     rs = avg_gain / avg_loss
    #     self.df['RSI'] = 100 - (100 / (1 + rs))
        
    #     # MACD
    #     ema_fast = self.df['Adj Close'].ewm(span=macd_fast, adjust=False).mean()
    #     ema_slow = self.df['Adj Close'].ewm(span=macd_slow, adjust=False).mean()
    #     self.df['MACD'] = ema_fast - ema_slow
    #     self.df['Signal_Line'] = self.df['MACD'].ewm(span=macd_signal, adjust=False).mean()
        
    #     print("🔧 Feature engineering complete")
    #     return self.df
    def feature_engineering(self, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
        if self.df is None:
            print("No data. Call fetch_data() first")
            return
        
        # Price Returns
        self.df['Return'] = self.df['Adj Close'].pct_change()
        
        # EMAs (12 and 26 for MACD components)
        ema_windows = [12, 26]
        for window in ema_windows:
            self.df[f'EMA{window}'] = self.df['Adj Close'].ewm(span=window, adjust=False).mean()
        
        # Volatility (EMA-based)
        self.df['Volatility20'] = self.df['Return'].ewm(span=20).std()
        
        # RSI (EMA-smoothed)
        delta = self.df['Adj Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=rsi_window).mean()  # EMA smoothing
        avg_loss = loss.ewm(span=rsi_window).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (EMA 12/26/9)
        ema_fast = self.df['Adj Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = self.df['Adj Close'].ewm(span=macd_slow, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=macd_signal, adjust=False).mean()
        
        print("Feature engineering complete (EMA-only)")
        return self.df
        # Signal Generation
    def generate_signals(self):
            """Generate buy/sell signals using MACD crossover strategy"""
            if self.df is None:
                print("No data. Call fetch_data() first")
                return
            
            # Ensure MACD columns exist
            if 'MACD' not in self.df.columns or 'Signal_Line' not in self.df.columns:
                print("Missing MACD/Signal Line. Run feature_engineering() first")
                return
            
            # Generate signals
            self.df['Signal'] = 0
            # Buy when MACD crosses ABOVE Signal Line
            self.df.loc[self.df['MACD'] > self.df['Signal_Line'], 'Signal'] = 1
            # Sell when MACD crosses BELOW Signal Line
            self.df['Position'] = self.df['Signal'].diff()
            
            print("Generated signals using MACD crossover (12/26/9 EMA)")

    # Backtesting Engine
    def backtest_strategy(self, initial_capital=10000):
        """Backtest trading strategy with portfolio simulation"""
        if self.df is None or 'Position' not in self.df.columns:
            print("⚠️ Missing data or signals. Run fetch_data() and generate_signals() first")
            return
        
        self.initial_capital = initial_capital
        df = self.df.copy()
        
        # Initialize portfolio columns
        df['Shares'] = 0
        df['Cash'] = initial_capital
        df['Total'] = initial_capital
        df['Returns'] = 0.0
        
        for i in range(1, len(df)):
            # Previous values
            prev_shares = df['Shares'].iloc[i-1]
            prev_cash = df['Cash'].iloc[i-1]
            price = df['Adj Close'].iloc[i]
            
            # Buy Signal
            if df['Position'].iloc[i] == 1:
                buyable_shares = prev_cash // price
                df.at[df.index[i], 'Shares'] = prev_shares + buyable_shares
                df.at[df.index[i], 'Cash'] = prev_cash - (buyable_shares * price)
            
            # Sell Signal
            elif df['Position'].iloc[i] == -1:
                df.at[df.index[i], 'Cash'] = prev_cash + (prev_shares * price)
                df.at[df.index[i], 'Shares'] = 0
            
            # No change
            else:
                df.at[df.index[i], 'Shares'] = prev_shares
                df.at[df.index[i], 'Cash'] = prev_cash
            
            # Update portfolio value
            df.at[df.index[i], 'Total'] = df.at[df.index[i], 'Cash'] + \
                                         (df.at[df.index[i], 'Shares'] * price)
            df.at[df.index[i], 'Returns'] = df.at[df.index[i], 'Total'] / \
                                          df.at[df.index[i-1], 'Total'] - 1
        
        self.backtest_results = df
        print("📊 Backtest complete")
        return df

    # Performance Metrics
    def calculate_performance(self):
        """Calculate key performance metrics"""
        if self.backtest_results is None:
            print("⚠️ No backtest results. Run backtest_strategy() first")
            return
        
        df = self.backtest_results
        metrics = {}
        
        # Total Return
        metrics['Total Return (%)'] = (df['Total'][-1] / self.initial_capital - 1) * 100
        
        # Annualized Return
        days = (df.index[-1] - df.index[0]).days
        metrics['Annualized Return (%)'] = ((df['Total'][-1] / self.initial_capital) ** (365/days) - 1) * 100
        
        # Max Drawdown
        cumulative_max = df['Total'].cummax()
        drawdown = (df['Total'] - cumulative_max) / cumulative_max
        metrics['Max Drawdown (%)'] = drawdown.min() * 100
        
        # Sharpe Ratio (assuming risk-free rate=0)
        metrics['Sharpe Ratio'] = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252)
        
        return pd.Series(metrics).round(2)

    # Visualization Methods
    def plot_price(self, interactive=False, save_path=None):
        """Plot price data with indicators"""
        if self.df is None:
            print("⚠️ No data to plot")
            return
        
        buys = sells = None
        if 'Position' in self.df.columns:
            buys = self.df[self.df['Position'] == 1]
            sells = self.df[self.df['Position'] == -1]
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            # ... rest of plotly code unchanged ...
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['Adj Close'], label='Price', color='black')
            
            # Plot MAs
            for col in self.df.columns:
                if col.startswith('EMA'):
                    plt.plot(self.df[col], label=col, alpha=0.7)
            
            # Plot signals if they exist
            if buys is not None and sells is not None:
                plt.scatter(
                    buys.index, buys['Adj Close'],
                    marker='^', color='green', s=100, label='Buy'
                )
                plt.scatter(
                    sells.index, sells['Adj Close'],
                    marker='v', color='red', s=100, label='Sell'
                )
            
            plt.title(f"{self.ticker} Price Analysis")
            plt.legend()
            plt.grid(True)
            if save_path:
                plt.savefig(save_path)
            plt.show()
    def plot_performance(self):
        """Plot backtest results"""
        if self.backtest_results is None:
            print("⚠️ No backtest results to plot")
            return
        
        df = self.backtest_results
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['Total'], label='Portfolio Value')
        plt.title(f"Strategy Performance ({self.ticker})")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.grid(True)
        plt.legend()
        plt.show()

