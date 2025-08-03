# %% [markdown]
# # Causality Analysis of Crypto Data
#
# ## Goal:
# * Determine causal relationships between BTC and other major crypto currencies
# * Analyze the stability of these relationships over time


# %%
import glob
import os
# Import causality analysis tools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

# Add the project root to the python path
try:
    # This works when the script is run directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
except NameError:
    # This works when run in an interactive environment like Jupyter
    # Assumes the notebook is in notebooks/python or notebooks/jupyter
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.analysis.time_varying_granger import (plot_tvgc_results,
                                               rolling_granger_causality,
                                               summarize_tvgc_results)

# Set style for better visualizations
sns.set_theme()
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 6]

# %% [markdown]
# ## Data Loading
# First, let's load our data files containing the cryptocurrency data.


# %%
def load_all_crypto_data(data_dir=os.path.join(project_root, "data", "processed")):
    all_data = {}
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split("_")[0]
        # Get symbol from filename
        print(f"Loading {file}...")
        df = pq.read_table(file).to_pandas()
        all_data[symbol] = df
    return all_data


# %% [markdown]
#  Additional Analysis: Summary Statistics

# %%
# Load the data

crypto_data = load_all_crypto_data()

# %%
# Calculate returns for each crypto
returns_data = {}
for symbol, df in crypto_data.items():
    returns = pd.DataFrame()
    returns["timestamp"] = df["timestamp"]
    returns["returns"] = np.log(df["close"].astype(float)).diff()
    returns_data[symbol] = returns

# %%
# Create a combined returns dataframe
combined_returns = pd.DataFrame()
for symbol, returns in returns_data.items():
    combined_returns[symbol] = returns["returns"]
combined_returns.index = list(returns_data.values())[0]["timestamp"]
# %% [markdown]
# ## Pairwise Granger Causality
# We will now perform pairwise Granger causality tests to identify potential causal relationships between the cryptocurrencies.

# %%
# Initialize the analyzer
analyzer = GrangerCausalityAnalyzer(combined_returns, max_lags=10)

# Run pairwise causality analysis
pairwise_results = analyzer.run_pairwise_causality()

# Print summary of significant results
print("Pairwise Granger Causality Results (Significant at 5%):")
print(pairwise_results[pairwise_results["significant"]])

# %%
# Get summary statistics
summary_stats = analyzer.get_summary_statistics()
print("\nCausality Summary Statistics:")
# %% [markdown]
# ## Multivariate Granger Causality
# Now, let's perform a multivariate Granger causality test to see which variables Granger-cause Bitcoin (BTC) in a multivariate context.

# %%
# Run multivariate causality analysis for BTC
target_crypto = "BTCUSDT"
test_stats, coef_pvals, optimal_lag = analyzer.run_multivariate_causality(
    target=target_crypto
)

print(
    f"\nMultivariate Granger Causality for {target_crypto} (Optimal Lag: {optimal_lag}):"
)

# Create a DataFrame for the results
multivariate_results = pd.DataFrame({"P-Value": test_stats})

# %% [markdown]
# ## Time-Varying Granger Causality
# Finally, let's analyze the time-varying Granger causality between Bitcoin (BTC) and Ethereum (ETH) to see if the relationship is stable over time.

# %%
# Define the pair to test
pair_to_test = ('BTCUSDT', 'SOLUSDT')

# Run time-varying Granger causality analysis
tvgc_results = rolling_granger_causality(
    y=combined_returns[pair_to_test[1]],
    x=combined_returns[pair_to_test[0]],
    window_size=500,
)

# Plot the results
plot_tvgc_results({f"{pair_to_test[0]} -> {pair_to_test[1]}": tvgc_results})

# Summarize the results
summary = summarize_tvgc_results(
    {f"{pair_to_test[0]} -> {pair_to_test[1]}": tvgc_results}
)
print("\nTime-Varying Granger Causality Summary:")
print(summary)
