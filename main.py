from data_explorer import StockDataExplorer
from matplotlib import pyplot as plt
# Step 1: Create an object
explorer = StockDataExplorer("MSFT")

# Step 2: Fetch the data
explorer.fetch_data()

# Step 3: Clean the data
explorer.clean_data()

# Step 4: Engineer new features
explorer.feature_engineering()

# Step 5: Explore the data
explorer.show_summary()
explorer.generate_signals()
# Step 6: Plot various graphs
explorer.plot_price()
explorer.plot_volume()
explorer.plot_returns()

