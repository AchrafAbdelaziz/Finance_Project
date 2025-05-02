from data_explorer import StockDataExplorer
from matplotlib import pyplot as plt

# analyzer = StockDataExplorer("AAPL", period="5y")
# analyzer.fetch_data()
# analyzer.clean_data()
# analyzer.feature_engineering()

# # Generate signals and backtest
# # analyzer.show_data()
# analyzer.generate_signals()
# analyzer.backtest_strategy(initial_capital=10000)

# # Show results
# print("\n=== Performance Metrics ===")
# print(analyzer.calculate_performance())

# print("\n=== Price Plot ===")
# analyzer.plot_price(interactive=True)

# print("\n=== Performance Plot ===")
# analyzer.plot_performance()
explorer = StockDataExplorer("AAPL", period="5y")
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