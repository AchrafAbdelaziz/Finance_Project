from data_explorer import StockDataExplorer
from matplotlib import pyplot as plt


explorer = StockDataExplorer("AAPL", period="1y")
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
# Show results
print("\nPerformance Metrics:")
print(explorer.calculate_performance())
explorer.plot_price(interactive=True)
explorer.plot_performance()