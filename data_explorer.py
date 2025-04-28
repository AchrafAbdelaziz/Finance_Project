import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


class StockDataExplorer:
    def __init__(self, ticker, period="6mo", interval="1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df = None

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(period=self.period, interval=self.interval, auto_adjust = False)
        print(f"[INFO] Data fetched for {self.ticker} with {len(self.df)} records.")
        return self.df

    def clean_data(self):
        if self.df is not None:
            before = len(self.df)
            self.df.dropna(inplace=True)
            after = len(self.df)
            print(f"[INFO] Cleaned data. Dropped {before - after} rows with missing values.")
        else:
            print("[ERROR] No data. Call fetch_data() first.")

    def feature_engineering(self):
        if self.df is not None:
            # Calculate daily returns
            self.df['Return'] = self.df['Adj Close'].pct_change()
            
            # 20-day Moving Average (roughly 1 month)
            self.df['MA20'] = self.df['Adj Close'].rolling(window=20).mean()
            
            # 50-day Moving Average
            self.df['MA50'] = self.df['Adj Close'].rolling(window=50).mean()
            
            # 20-day Volatility
            self.df['Volatility20'] = self.df['Return'].rolling(window=20).std()

            print("[INFO] Feature engineering completed.")
            return self.df
        else:
            print("[ERROR] No data. Call fetch_data() first.")

    def show_summary(self):
        if self.df is not None:
            print("[SUMMARY]")
            print(self.df.describe())
        else:
            print("[ERROR] No data. Call fetch_data() first.")

    def plot_volume(self):
        if self.df is not None:
            self.df['Volume'].plot(title=f"{self.ticker} - Volume Traded", figsize=(10, 5), color='orange')
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.grid(True)
            plt.show()
        else:
            print("[ERROR] No data to plot.")

    def plot_returns(self):
        if self.df is not None:
            self.df['Return'].plot(title=f"{self.ticker} - Daily Returns", figsize=(10, 5), color='green')
            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.grid(True)
            plt.show()
        else:
            print("[ERROR] No returns to plot. Did you run feature_engineering()?")
    def generate_signals(self):
        if self.df is not None:
            # Generate signals (1 for buy, -1 for sell)
            self.df['Signal'] = 0  # Default no signal
            self.df['Signal'][20:] = [1 if self.df['MA20'][i] > self.df['MA50'][i] else 0 for i in range(20, len(self.df))]
            self.df['Position'] = self.df['Signal'].diff()
            print("[INFO] Trading signals generated.")
        else:
            print("[ERROR] No data. Call fetch_data() first.")
    def plot_price(self):
        
        if self.df is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['Adj Close'], label="Adj Close", color='black', alpha=0.5)
            plt.plot(self.df['MA20'], label="20-Day MA", color='blue', alpha=0.75)
            plt.plot(self.df['MA50'], label="50-Day MA", color='red', alpha=0.75)
            
            # Plot Buy signals
            plt.plot(self.df[self.df['Position'] == 1].index, 
                     self.df['MA20'][self.df['Position'] == 1], 
                     '^', markersize=10, color='g', lw=0, label="Buy Signal")

            # Plot Sell signals
            plt.plot(self.df[self.df['Position'] == -1].index, 
                     self.df['MA20'][self.df['Position'] == -1], 
                     'v', markersize=10, color='r', lw=0, label="Sell Signal")
            
            plt.title(f"{self.ticker} - Adjusted Close with Buy/Sell Signals")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("[ERROR] No data to plot.")
