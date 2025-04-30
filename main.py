from data_explorer import StockDataExplorer
from matplotlib import pyplot as plt

analyzer = StockDataExplorer("AAPL", period="5y")
analyzer.fetch_data()
analyzer.clean_data()
analyzer.feature_engineering()

# Generate signals and backtest
analyzer.generate_signals()
analyzer.backtest_strategy(initial_capital=10000)

# Show results
print("\n=== Performance Metrics ===")
print(analyzer.calculate_performance())

print("\n=== Price Plot ===")
analyzer.plot_price(interactive=True)

print("\n=== Performance Plot ===")
analyzer.plot_performance()