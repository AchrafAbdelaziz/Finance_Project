import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, List, Dict
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed - using matplotlib for plots")

    class StockDataExplorer:
        def __init__(self, ticker: str, period: str = "6mo", interval: str = "1d"):
            self.ticker = ticker
            self.period = period
            self.interval = interval
            self.df: Optional[pd.DataFrame] = None
            self.initial_capital: Optional[float] = None
            self.backtest_results: Optional[pd.DataFrame] = None

        # Data Pipeline Methods
        def fetch_data(self) -> Optional[pd.DataFrame]:
            """Fetch stock data from Yahoo Finance with validation"""
            try:
                stock = yf.Ticker(self.ticker)
                self.df = stock.history(
                    period=self.period,
                    interval=self.interval,
                    auto_adjust=False,  # Keep 'Adj Close' column
                    actions=False  # Exclude dividends and stock splits
                )
                
                if self.df.empty:
                    raise ValueError(f"No data found for {self.ticker}")
                    
                # Validate required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                if not all(col in self.df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in self.df.columns]
                    raise ValueError(f"Missing critical columns: {missing}")
                
                logger.info(f"Successfully fetched {len(self.df)} records for {self.ticker}")
                return self.df
                
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                return None

        def clean_data(self, method: str = 'drop') -> None:
            """Handle missing values with multiple methods"""
            if self.df is None:
                logger.warning("No data to clean. Call fetch_data() first")
                return
                
            initial_rows = len(self.df)
            if method == 'ffill':
                self.df.ffill(inplace=True)
            elif method == 'interpolate':
                self.df.interpolate(method='time', inplace=True)
            else:  # default to drop
                self.df.dropna(inplace=True)
                
            logger.info(f"Data cleaning: {initial_rows - len(self.df)} rows removed/processed")

        # Feature Engineering
        def feature_engineering(
            self,
            rsi_window: int = 14,
            macd_fast: int = 12,
            macd_slow: int = 26,
            macd_signal: int = 9,
            ema_windows: List[int] = [12, 20, 26]
        ) -> Optional[pd.DataFrame]:
            """Add technical indicators with validation"""
            if self.df is None:
                logger.warning("No data. Call fetch_data() first")
                return None
                
            # Parameter validation
            if macd_fast >= macd_slow:
                raise ValueError("MACD fast window must be smaller than slow window")
            if rsi_window < 1:
                raise ValueError("RSI window must be ≥1")

            try:
                # Price Returns
                self.df['Return'] = self.df['Adj Close'].pct_change()
                
                # EMAs
                for window in ema_windows:
                    self.df[f'EMA{window}'] = self.df['Adj Close'].ewm(
                        span=window, 
                        adjust=False
                    ).mean()
                
                # Volatility (20-day rolling)
                self.df['Volatility20'] = self.df['Return'].rolling(20).std()
                
                # RSI (EMA-smoothed)
                delta = self.df['Adj Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(span=rsi_window).mean()
                avg_loss = loss.ewm(span=rsi_window).mean()
                rs = avg_gain / avg_loss
                self.df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD (EMA-based)
                ema_fast = self.df['Adj Close'].ewm(span=macd_fast, adjust=False).mean()
                ema_slow = self.df['Adj Close'].ewm(span=macd_slow, adjust=False).mean()
                self.df['MACD'] = ema_fast - ema_slow
                self.df['Signal_Line'] = self.df['MACD'].ewm(
                    span=macd_signal, 
                    adjust=False
                ).mean()
                
                logger.info("Feature engineering complete")
                return self.df
                
            except Exception as e:
                logger.error(f"Feature engineering failed: {str(e)}")
                return None

        # Signal Generation
        def generate_signals(self) -> None:
            if self.df is None or 'MACD' not in self.df.columns or 'RSI' not in self.df.columns:
                logger.warning("Required indicators not available.")
                return
            
            self.df['Signal'] = 0
            
            # Example buy condition: MACD > Signal line and RSI < 30
            buy_condition = (self.df['MACD'] > self.df['Signal_Line']) & (self.df['RSI'] < 30)
            sell_condition = (self.df['MACD'] < self.df['Signal_Line']) & (self.df['RSI'] > 70)

            self.df.loc[buy_condition, 'Signal'] = 1
            self.df.loc[sell_condition, 'Signal'] = -1
            
            self.df['Position'] = self.df['Signal'].diff().shift(1)
            logger.info("Generated combined MACD+RSI signals.")
        # Backtesting Engine
        def backtest_strategy(
            self,
            initial_capital: float = 10000,
            transaction_cost: float = 0.0005,
            execution_mode: str = 'close'
        ) -> Optional[pd.DataFrame]:
            """Backtest strategy with realistic trade execution"""
            if self.df is None or 'Position' not in self.df.columns:
                logger.warning("Missing data/signals. Complete previous steps first")
                return None
                
            # Validate execution mode
            valid_modes = ['close', 'next_open']
            if execution_mode not in valid_modes:
                raise ValueError(f"Invalid execution mode. Use: {valid_modes}")

            self.initial_capital = initial_capital
            df = self.df.copy()
            
            # Initialize portfolio columns
            df['Shares'] = 0.0
            df['Cash'] = float(initial_capital)
            df['Total'] = df['Cash'] + (df['Shares'] * df['Adj Close'])
            df['Returns'] = 0.0
            
            for i in range(1, len(df)):
                prev_shares = df['Shares'].iloc[i-1]
                prev_cash = df['Cash'].iloc[i-1]
                
                # Get execution price based on mode
                if execution_mode == 'next_open':
                    price = df['Open'].iloc[i]
                else:  # 'close'
                    price = df['Adj Close'].iloc[i]
                
                # Execute trades
                position = df['Position'].iloc[i]
                
                if position == 1:  # Buy
                    # Calculate maximum buyable shares with transaction cost
                    buyable_shares = prev_cash / (price * (1 + transaction_cost))
                    cost = buyable_shares * price * (1 + transaction_cost)
                    
                    df.at[df.index[i], 'Shares'] = prev_shares + buyable_shares
                    df.at[df.index[i], 'Cash'] = prev_cash - cost
                    
                elif position == -1:  # Sell
                    # Sell all shares with transaction cost
                    sale_value = prev_shares * price * (1 - transaction_cost)
                    df.at[df.index[i], 'Cash'] = prev_cash + sale_value
                    df.at[df.index[i], 'Shares'] = 0
                    
                else:  # Hold
                    df.at[df.index[i], 'Shares'] = prev_shares
                    df.at[df.index[i], 'Cash'] = prev_cash
                
                # Update portfolio value
                df.at[df.index[i], 'Total'] = (
                    df.at[df.index[i], 'Cash'] + 
                    (df.at[df.index[i], 'Shares'] * df['Adj Close'].iloc[i])
                )
                
                # Calculate daily returns
                df.at[df.index[i], 'Returns'] = (
                    df.at[df.index[i], 'Total'] / 
                    df.at[df.index[i-1], 'Total'] - 1
                )

            self.backtest_results = df
            logger.info("Backtest complete with transaction costs")
            return df

        # Performance Metrics
        def calculate_performance(
            self, 
            risk_free_rate: float = 0.02
        ) -> Optional[pd.Series]:
            """Calculate comprehensive performance metrics"""
            if self.backtest_results is None:
                logger.warning("No backtest results available")
                return None
                
            df = self.backtest_results
            metrics = {}
            
            # Returns
            total_return = (df['Total'][-1] / self.initial_capital - 1) * 100
            metrics['Total Return (%)'] = total_return
            
            days = (df.index[-1] - df.index[0]).days
            annualized_return = ((df['Total'][-1] / self.initial_capital) ** (365/days) - 1) * 100
            metrics['Annualized Return (%)'] = annualized_return
            
            # Risk Metrics
            cumulative_max = df['Total'].cummax()
            drawdown = (df['Total'] - cumulative_max) / cumulative_max
            metrics['Max Drawdown (%)'] = drawdown.min() * 100
            metrics['Volatility (ann.) (%)'] = df['Returns'].std() * np.sqrt(252) * 100
            
            # Risk-Adjusted Returns
            excess_returns = df['Returns'] - (risk_free_rate / 252)
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            metrics['Sharpe Ratio'] = sharpe
            
            # Trade Metrics
            trades = df[df['Position'] != 0]
            metrics['Total Trades'] = len(trades)
            metrics['Win Rate (%)'] = (trades['Returns'] > 0).mean() * 100
            
            return pd.Series(metrics).round(2)

        # Visualization Methods
        def plot_price(
            self, 
            interactive: bool = False, 
            save_path: Optional[str] = None
        ) -> None:
            """Plot price data with technical indicators"""
            if self.df is None:
                logger.warning("No data to plot")
                return
                
            # Prepare signals
            buys = sells = None
            if 'Position' in self.df.columns:
                buys = self.df[self.df['Position'] == 1]
                sells = self.df[self.df['Position'] == -1]

            if interactive and PLOTLY_AVAILABLE:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_heights=[0.7, 0.3])
                
                # Price and EMAs
                fig.add_trace(go.Scatter(
                    x=self.df.index, y=self.df['Adj Close'],
                    name='Price', line=dict(color='black')
                ), row=1, col=1)
                
                for col in self.df.columns:
                    if col.startswith('EMA'):
                        fig.add_trace(go.Scatter(
                            x=self.df.index, y=self.df[col],
                            name=col, line=dict(width=1)
                        ), row=1, col=1)
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=self.df.index, y=self.df['MACD'],
                    name='MACD', line=dict(color='blue')
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=self.df.index, y=self.df['Signal_Line'],
                    name='Signal', line=dict(color='orange')
                ), row=2, col=1)
                
                # Signals
                if buys is not None:
                    fig.add_trace(go.Scatter(
                        x=buys.index, y=buys['Adj Close'],
                        mode='markers', name='Buy',
                        marker=dict(symbol='triangle-up', color='green', size=10)
                    ), row=1, col=1)
                    
                if sells is not None:
                    fig.add_trace(go.Scatter(
                        x=sells.index, y=sells['Adj Close'],
                        mode='markers', name='Sell',
                        marker=dict(symbol='triangle-down', color='red', size=10)
                    ), row=1, col=1)
                
                fig.update_layout(
                    title=f"{self.ticker} Technical Analysis",
                    hovermode="x unified",
                    showlegend=True
                )
                fig.show()
                
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Price and EMAs
                ax1.plot(self.df['Adj Close'], label='Price', color='black')
                for col in self.df.columns:
                    if col.startswith('EMA'):
                        ax1.plot(self.df[col], label=col, alpha=0.7)
                
                if buys is not None:
                    ax1.scatter(
                        buys.index, buys['Adj Close'],
                        marker='^', color='green', s=100, label='Buy'
                    )
                if sells is not None:
                    ax1.scatter(
                        sells.index, sells['Adj Close'],
                        marker='v', color='red', s=100, label='Sell'
                    )
                ax1.set_title(f"{self.ticker} Price Analysis")
                ax1.legend()
                ax1.grid(True)
                
                # MACD
                ax2.plot(self.df['MACD'], label='MACD', color='blue')
                ax2.plot(self.df['Signal_Line'], label='Signal', color='orange')
                ax2.set_title("MACD")
                ax2.grid(True)
                ax2.legend()
                
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path)
                plt.show()

        def plot_performance(self) -> None:
            """Plot backtest performance vs benchmark"""
            if self.backtest_results is None:
                logger.warning("No performance data to plot")
                return
                
            df = self.backtest_results
            
            plt.figure(figsize=(12, 6))
            df['Total'].plot(label='Strategy')
            (self.df['Adj Close'] / self.df['Adj Close'].iloc[0] * self.initial_capital).plot(
                label='Buy & Hold', 
                linestyle='--'
            )
            plt.title(f"Strategy Performance vs Buy & Hold ({self.ticker})")
            plt.xlabel("Date")
            plt.ylabel("Value ($)")
            plt.grid(True)
            plt.legend()
            plt.show()

