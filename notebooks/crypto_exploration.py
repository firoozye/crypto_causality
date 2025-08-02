# %% [markdown]
# # First Exploratory Data Analysis of Crypto Data
#
# ## Plotting
# * Plot data
# * Plot Correlations
# * ACFs and PACFs
# * Summary Statistics
#
# ## Goal: All things needed for a pre-causal analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import ccf
import itertools
import glob
import os
import pyarrow.parquet as pq

# Set style for better visualizations
sns.set_theme()
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 6]

# %% [markdown]
# ## Multiple Y-Axes Approach


# %%
def plot_multiple_y_axes(crypto_data):
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Create additional axes that share the same x-axis
    axes = [ax1]
    for i in range(len(crypto_data) - 1):
        axes.append(ax1.twinx())

    # Offset the right spine of each additional axis
    # This creates space between the y-axes
    for i, ax in enumerate(axes[2:], start=2):
        ax.spines['right'].set_position(('outward', 60 * (i - 1)))

    # Plot each cryptocurrency on its own y-axis
    colors = plt.cm.husl(np.linspace(0, 1, len(crypto_data)))
    for (symbol, df), ax, color in zip(crypto_data.items(), axes, colors):
        line = ax.plot(df['timestamp'], df['close'], label=symbol, color=color)
        ax.set_ylabel(f'{symbol} Price (USDT)', color=color)
        ax.tick_params(axis='y', labelcolor=color)

    # Add title and adjust layout
    plt.title('Cryptocurrency Prices (Multiple Y-Axes)')
    ax1.set_xlabel('Time')

    # Create combined legend
    lines = []
    labels = []
    for ax in axes:
        ax_line, ax_label = ax.get_legend_handles_labels()
        lines.extend(ax_line)
        labels.extend(ax_label)

    plt.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Subplot Grid Approach


# %%
def plot_subplot_grid(crypto_data):
    # Calculate grid dimensions
    n_plots = len(crypto_data)
    n_cols = 2  # You can adjust this
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle('Cryptocurrency Prices', fontsize=16, y=1.02)

    # Flatten axes array for easier iteration
    if n_rows > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes] if n_cols == 1 else axes

    # Plot each cryptocurrency
    for (symbol, df), ax in zip(crypto_data.items(), axes_flat):
        ax.plot(df['timestamp'], df['close'], label=symbol)
        ax.set_title(f'{symbol} Price')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (USDT)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    # Remove any empty subplots
    for idx in range(len(crypto_data), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Data Loading
# First, let's load our data files containing the cryptocurrency data.


# %%
def load_all_crypto_data(data_dir="../data/processed/"):
    all_data = {}
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split('_')[0]
        # Get symbol from filename
        print(f"Loading {file}...")
        df = pq.read_table(file).to_pandas()
        all_data[symbol] = df
    return all_data


# %% [markdown]
#  Additional Analysis: Summary Statistics

# Load the data
crypto_data = load_all_crypto_data()

# Debug data loading
print("\nLoaded symbols:", list(crypto_data.keys()))
for symbol, df in crypto_data.items():
    print(f"\nShape of {symbol} data: ", df.shape)
    print(f"Columns: ", df.columns.tolist())
    print(f"Sample of {symbol} data: ")
    print(df[['timestamp', 'close']].head())


# %%
# Plot both versions
print("Plotting multiple y-axes version...")
plot_multiple_y_axes(crypto_data)

# %%
print("\nPlotting subplot grid version...")
plot_subplot_grid(crypto_data)



# %%
# 2. Returns Calculation and Correlation Analysis
# Calculate returns for each crypto
returns_data = {}
for symbol, df in crypto_data.items():
    returns = pd.DataFrame()
    returns['timestamp'] = df['timestamp']
    returns['returns'] = np.log(df['close'].astype(float)).diff()
    returns_data[symbol] = returns

# %%
# Create a combined returns dataframe
combined_returns = pd.DataFrame()
for symbol, returns in returns_data.items():
    combined_returns[symbol] = returns['returns']
combined_returns.index = list(returns_data.values())[0]['timestamp']

# %%
# Plot correlation heatmap of log returns
plt.figure(figsize=(12, 8))
sns.heatmap(
    combined_returns.corr(),
    annot=True,
    cmap='RdYlBu',
    center=0,
    fmt='.2f'
)
plt.title('Correlation Heatmap of Crypto Returns')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ACF and PACF Analysis
# Analyze the autocorrelation and partial autocorrelation
# functions for each cryptocurrency.

# %%
def plot_acf_pacf(series, symbol, lags=50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # ACF
    acf_values = acf(series.dropna(), nlags=lags)
    ax1.stem(range(len(acf_values)), acf_values)
    ax1.axhline(y=0, linestyle='-', color='black')
    ax1.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    ax1.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    ax1.set_title(f'Autocorrelation Function for {symbol}')

    # PACF
    pacf_values = pacf(series.dropna(), nlags=lags)
    ax2.stem(range(len(pacf_values)), pacf_values)
    ax2.axhline(y=0, linestyle='-', color='black')
    ax2.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    ax2.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    ax2.set_title(f'Partial Autocorrelation Function for {symbol}')

    plt.tight_layout()
    plt.show()

# %%
# Plot ACF and PACF for each crypto
for symbol in returns_data.keys():
    print(f"\nAnalyzing {symbol}")
    plot_acf_pacf(returns_data[symbol]['returns'], symbol)

# %% [markdown]
# ## Cross-Correlation Analysis
# Let's analyze the lead-lag relationships between different cryptocurrency pairs
# using cross-correlation functions. This will help us understand if price movements
# in one crypto tend to lead or lag another.
# 
# %%
def plot_ccf(series1, series2, name1, name2, lags=50):
    ccf_values = ccf(series1.dropna(), series2.dropna(), adjusted=False)

    # Plot CCF
    plt.figure(figsize=(15, 5))
    plt.stem(range(-lags, lags + 1), ccf_values[lags - lags:lags + lags + 1])
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(series1)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(series1)), linestyle='--', color='gray')
    plt.title(f'Cross-Correlation Function: {name1} vs {name2}')
    plt.xlabel('Lag')
    plt.ylabel('CCF')
    plt.tight_layout()
    plt.show()

    # Find significant lags
    threshold = 1.96 / np.sqrt(len(series1))
    significant_lags = np.where(np.abs(ccf_values) > threshold)[0] - lags
    if len(significant_lags) > 0:
        print(f"Significant lags between {name1} and {name2}:")
        for lag in significant_lags:
            if lag < 0:
                print(
                    f"{name2} leads {name1} by {abs(lag)} periods "
                    f"(correlation: {ccf_values[lag + lags]:.3f})"
                )
            elif lag > 0:
                print(
                    f"{name1} leads {name2} by {lag} periods "
                    f"(correlation: {ccf_values[lag + lags]:.3f})"
                )

# %%
# Get all unique pairs of cryptocurrencies
crypto_pairs = list(itertools.combinations(returns_data.keys(), 2))

# Plot CCF for each pair
for pair in crypto_pairs:
    symbol1, symbol2 = pair
    print(f"\nAnalyzing cross-correlation between {symbol1} and {symbol2}")
    plot_ccf(
        returns_data[symbol1]['returns'],
        returns_data[symbol2]['returns'],
        symbol1,
        symbol2
    )

# %% [markdown]
# ## ACF Analysis with Confidence Intervals
# Let's look at the autocorrelation structure of each cryptocurrency's returns,
# including confidence intervals to identify significant correlations.

# %%
def plot_detailed_acf(series, symbol, lags=50):
    # Calculate ACF with confidence intervals
    acf_values, confint = acf(series.dropna(), nlags=lags, alpha=0.05, fft=False)

    plt.figure(figsize=(15, 5))
    plt.stem(range(len(acf_values)), acf_values)
    plt.axhline(y=0, linestyle='-', color='black')

    # Plot confidence intervals
    plt.fill_between(
        range(len(acf_values)),
        confint[:, 0],
        confint[:, 1],
        alpha=0.1,
        color='blue'
    )

    plt.title(f'Autocorrelation Function for {symbol}')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.tight_layout()
    plt.show()

    # Print significant lags
    significant_lags = np.where(
        (acf_values < confint[:, 0]) |
        (acf_values > confint[:, 1])
    )[0]
    if len(significant_lags) > 0:
        print(f"\nSignificant autocorrelations for {symbol}:")
        for lag in significant_lags:
            print(f"Lag {lag}: {acf_values[lag]:.3f}")

# %%
# Plot ACF for each cryptocurrency
for symbol, returns in returns_data.items():
    print(f"\nAnalyzing autocorrelation for {symbol}")
    plot_detailed_acf(returns['returns'], symbol)

# %% [markdown]
# ## Summary of Lead-Lag Relationships
# Let's create a summary table of the strongest lead-lag relationships found.

# %%
def create_lead_lag_summary(returns_data, lags=50):
    summary_data = []

    for pair in itertools.combinations(returns_data.keys(), 2):
        symbol1, symbol2 = pair
        ccf_values = ccf(
            returns_data[symbol1]['returns'].dropna(),
            returns_data[symbol2]['returns'].dropna(),
            adjusted=False
        )

        # Find max correlation and corresponding lag
        max_corr_idx = np.argmax(np.abs(ccf_values))
        max_corr = ccf_values[max_corr_idx]
        max_lag = max_corr_idx - lags

        if max_lag < 0:
            leader = symbol2
            lagger = symbol1
            lag = abs(max_lag)
        else:
            leader = symbol1
            lagger = symbol2
            lag = max_lag

        summary_data.append({
            'Leader': leader,
            'Lagger': lagger,
            'Lag (minutes)': lag,
            'Correlation': max_corr
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values('Correlation', key=abs, ascending=False)

# %%
summary_df = create_lead_lag_summary(returns_data)
print("Summary of Strongest Lead-Lag Relationships:")
print(summary_df)

# %% [markdown]
# ## Summary Statistics and Volatility Analysis

# %%
# Summary Statistics
summary_stats = pd.DataFrame()
for symbol, returns in returns_data.items():
    stats = returns['returns'].describe()
    stats['skewness'] = returns['returns'].skew()
    stats['kurtosis'] = returns['returns'].kurtosis()
    summary_stats[symbol] = stats

print("\nSummary Statistics for Returns:")
print(summary_stats)

# %%
# Volatility Analysis
# Calculate rolling volatility (30-minute window)
plt.figure(figsize=(15, 10))
for symbol, returns in returns_data.items():
    vol = returns['returns'].rolling(window=30).std() * np.sqrt(30)
    plt.plot(returns['timestamp'], vol, label=symbol)

plt.title('30-Minute Rolling Volatility')
plt.xlabel('Time')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analysis
# * We have successfully loaded and analyzed the cryptocurrency data.
# * We visualized the price data using multiple y-axes and subplot grid approaches.
# * Calculated log returns and analyzed their correlations.
# * The Data are clearly non-stationary. This means that we should look at returns
#   instead of prices, before doing Granger causality tests
