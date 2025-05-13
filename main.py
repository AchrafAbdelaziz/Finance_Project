from data_explorer import StockDataExplorer
from matplotlib import pyplot as plt


explorer = StockDataExplorer("GOOG", period="5y")
explorer.fetch_data()
explorer.clean_data(method='interpolate')
explorer.feature_engineering()
explorer.generate_signals()
# Backtest with 0.05% transaction costs
explorer.backtest_strategy(
    initial_capital=10000,
    transaction_cost=0.0005,
    execution_mode='next_open'
)
explorer.show_data()
# Show results
print("\nPerformance Metrics:")
print(explorer.calculate_performance())
explorer.plot_price(interactive=True)
explorer.plot_performance()
# Print critical metrics
print("\nStrategy Performance:")

# Compare to buy-and-hold
buy_hold_return = (explorer.df['Adj Close'].iloc[-1]/explorer.df['Adj Close'].iloc[0]-1)*100
print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
print(explorer.calculate_performance())



# Example usage with market events:
major_events = {
    "Fed Meeting": "2023-07-26",
    "Earnings": "2023-08-01",
    "CPI Report": "2023-08-10"
}


explorer.create_killer_chart(major_events=major_events)
