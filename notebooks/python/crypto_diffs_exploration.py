# %% [markdown]
# # First Exploratory Data Analysis of Crypto Data (DIFFS)
#
# ## Plotting
# * Plot data  in *diffs* (log returns, simple returns, price differences)
# * Plot Correlations in diffs
# * ACFs and CCFss in diffs
# * Summary Statistics
#
# ## Goal: All things needed for a pre-causal analysis

# %%

import glob
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from statsmodels.tsa.stattools import acf, ccf, pacf

# Set style for better visualizations
sns.set_theme()
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 6]

# %%


def load_and_prepare_data(data_dir="../data/processed/"):
    # Load raw data
    all_data = {}
    returns_data = {}

    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split("_")[0]
        df = pq.read_table(file).to_pandas()

        # Calculate log returns
        df["log_returns"] = np.log(df["close"]).diff()

        # Calculate simple returns
        df["simple_returns"] = df["close"].pct_change()

        # Calculate absolute difference
        df["price_diff"] = df["close"].diff()

        all_data[symbol] = df
        returns_data[symbol] = df[
            ["timestamp", "log_returns", "simple_returns", "price_diff"]
        ].copy()

    return all_data, returns_data


# %%
def plot_acf_analysis(returns_data, diff_type="log_returns", max_lags=50):
    """
    Plot ACF for each cryptocurrency's returns/differences
    diff_type: 'log_returns', 'simple_returns', or 'price_diff'
    """
    for symbol, df in returns_data.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f"ACF Analysis of {diff_type} for {symbol}")

        series = df[diff_type].dropna()

        # ACF
        acf_values = acf(series, nlags=max_lags)
        ax1.stem(range(len(acf_values)), acf_values)
        ax1.axhline(y=0, linestyle="-", color="black")
        ax1.axhline(y=1.96 / np.sqrt(len(series)), linestyle="--", color="gray")
        ax1.axhline(y=-1.96 / np.sqrt(len(series)), linestyle="--", color="gray")
        ax1.set_title(f"Autocorrelation Function")

        # PACF
        pacf_values = pacf(series, nlags=max_lags)
        ax2.stem(range(len(pacf_values)), pacf_values)
        ax2.axhline(y=0, linestyle="-", color="black")
        ax2.axhline(y=1.96 / np.sqrt(len(series)), linestyle="--", color="gray")
        ax2.axhline(y=-1.96 / np.sqrt(len(series)), linestyle="--", color="gray")
        ax2.set_title(f"Partial Autocorrelation Function")

        plt.tight_layout()
        plt.show()


# %% [markdown]
# ## Cross-Correlation Analysis of Differences


# %%
def plot_ccf_analysis(returns_data, diff_type="log_returns", max_lags=50):
    """
    Plot CCF between pairs of cryptocurrencies
    diff_type: 'log_returns', 'simple_returns', or 'price_diff'
    """
    # Get all pairs of cryptocurrencies
    pairs = list(itertools.combinations(returns_data.keys(), 2))

    for symbol1, symbol2 in pairs:
        # Get the return series
        series1 = returns_data[symbol1][diff_type].dropna()
        series2 = returns_data[symbol2][diff_type].dropna()

        # Calculate CCF
        ccf_values = ccf(series1, series2, adjusted=False)

        # Plot
        plt.figure(figsize=(15, 5))
        plt.stem(
            range(-max_lags, max_lags + 1),
            ccf_values[max_lags - max_lags : max_lags + max_lags + 1],
        )
        plt.axhline(y=0, linestyle="-", color="black")
        plt.axhline(y=1.96 / np.sqrt(len(series1)), linestyle="--", color="gray")
        plt.axhline(y=-1.96 / np.sqrt(len(series1)), linestyle="--", color="gray")

        plt.title(f"Cross-Correlation of {diff_type}: {symbol1} vs {symbol2}")
        plt.xlabel("Lag")
        plt.ylabel("CCF")

        # Find significant lags
        threshold = 1.96 / np.sqrt(len(series1))
        sig_lags = np.where(np.abs(ccf_values) > threshold)[0] - max_lags
        if len(sig_lags) > 0:
            print(f"\nSignificant lags between {symbol1} and {symbol2}:")
            for lag in sig_lags:
                corr = ccf_values[lag + max_lags]
                if lag < 0:
                    print(
                        f"{symbol2} leads {symbol1} by {abs(lag)} periods "
                        f"(correlation: {corr:.3f})"
                    )
                elif lag > 0:
                    print(
                        f"{symbol1} leads {symbol2} by {lag} periods "
                        f"(correlation: {corr:.3f})"
                    )

        plt.tight_layout()
        plt.show()


# %% [markdown]
# ## Run Analysis

# %%
# Load data
all_data, returns_data = load_and_prepare_data()

# Plot ACF analysis for log returns
print("ACF Analysis of Log Returns")
plot_acf_analysis(returns_data, diff_type="log_returns")

# Plot CCF analysis for log returns
print("\nCCF Analysis of Log Returns")
plot_ccf_analysis(returns_data, diff_type="log_returns")

# %% [markdown]
# ## Additional Analysis: Returns Distribution


# %%
def plot_returns_distribution(returns_data, diff_type="log_returns"):
    plt.figure(figsize=(15, 5))

    for symbol, df in returns_data.items():
        sns.kdeplot(data=df[diff_type].dropna(), label=symbol)

    plt.title(f"Distribution of {diff_type}")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


plot_returns_distribution(returns_data, "log_returns")
