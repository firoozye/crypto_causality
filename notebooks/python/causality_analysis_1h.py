# %% [markdown]
# # Causality Analysis of Crypto Data
#
#  This script performs a comprehensive analysis of causal relationships between major
#  cryptocurrencies using Granger causality tests. It includes pairwise, multivariate,
#  and time-varying Granger causality analyses to uncover potential causal influences
#  among cryptocurrencies. This notebook begins with the analysis of Kose and Unal
#  (see references).

# %% [markdown]
# ## References:
# * [Granger Causality (Wikipedia)](https://en.wikipedia.org/wiki/Granger_causality)
# * [Granger Causality: A Review (Annual Reviews)](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-040120-010930)
# * [Kose & Unal, Causal relationship among cryptocurrencies:  The effects of sampling interval and
# sample size](https://www.scopus.com/pages/publications/85149277679)
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
import networkx as nx
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
from src.data.data_processor import DataProcessor

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
#  ## Load Cryptocurrency Data

#   The data is expected to be in Parquet format, with each file named as
#  `<symbol>_data.parquet`.
#  The files should contain columns like `timestamp`, `open`, `high`, `low`, `close`,
#  and `volume`.
#  ### Note:
#  Note that all data is downloaded from Binance using the binance API, and the
#  src/data/downloader.py script.  This can be easily adapted to load different
#  frequencies or different symbols


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
# ## Resampling Data to Lower Frequencies
# To investigate causality at longer frequencies, we will resample the 1-minute data.

# %%
# Initialize DataProcessor (we only need it for the resample_data method)
data_processor = DataProcessor(data={})

# Resample combined returns to 1-hour frequency
# Note: This resampling is for the returns data. If you need to resample OHLCV data,
# you would do it on the individual crypto_data dataframes before calculating returns.
resampled_returns_1h = combined_returns.resample('1H').last().dropna()

print(f"Shape of 1-minute returns: {combined_returns.shape}")
print(f"Shape of 1-hour resampled returns: {resampled_returns_1h.shape}")

# Use the resampled data for analysis from now on
combined_returns = resampled_returns_1h
# %% [markdown]
# ## Pairwise Granger Causality
# We will now perform pairwise Granger causality tests to identify potential causal
# relationships between the cryptocurrencies.
# Eseentially this is fitting a model of the form:
# $\Delta x_t = \alpha + \sum_{i=1}^{p} \beta_i \Delta x_{t-i} + \sum_{j=1}^{q} \gamma_j
# \Delta y_{t-j} + \epsilon_t$
# where $x_t$ is the target variable (e.g., BTC returns) and $y_t$ is the predictor
# variable and $p$ and $q$ are the lags for the target and predictor variables, chosen
# using the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC).
# The causality test checks if the coefficients $\gamma_j$ are significantly different
# from zero, indicating that past values of $y_t$ help predict $x_t$.

# %%
# Initialize the analyzer
analyzer = GrangerCausalityAnalyzer(combined_returns, max_lags=3)

# Run pairwise causality analysis
pairwise_results = analyzer.run_pairwise_causality()

# Print summary of significant results
print("Pairwise Granger Causality Results (Significant at 5%):")
print(pairwise_results[pairwise_results["significant"]])

# %% [markdown]
# ### Visualization of Pairwise Granger Causality
# A directed graph can help visualize the significant pairwise causal relationships.


# %%
def plot_causality_network(
    pairwise_results, title="Significant Granger Causal Relationships"
):
    G = nx.DiGraph()

    # Add nodes for all unique cryptocurrencies
    all_cryptos = pd.concat(
        [pairwise_results["cause"], pairwise_results["effect"]]
    ).unique()
    for crypto in all_cryptos:
        G.add_node(crypto)

    # Add edges for significant relationships
    significant_relationships = pairwise_results[pairwise_results["significant"]]
    for _, row in significant_relationships.iterrows():
        G.add_edge(row["cause"], row["effect"], weight=row["min_p_value"])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8)  # positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=3000, alpha=0.6)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=30, edge_color="black", width=2.0
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title(title)
    plt.axis("off")
    plt.show()


# %%
plot_causality_network(pairwise_results)

# %%
# Get summary statistics
summary_stats = analyzer.get_summary_statistics()
print("\nCausality Summary Statistics:")
# %% [markdown]
# ## Multivariate Granger Causality
# Now, let's perform a multivariate Granger causality test to see which variables
# Granger-cause Bitcoin (BTC) in a multivariate context.
# For more details, see: [Multivariate Granger Causality (Wikipedia)](https://en.wikipedia.org/wiki/Granger_causality#Multivariate_analysis)
# Rather than testing pairwise relationships, this approach considers all variables
# simultaneously to determine their joint influence on the target variable, testing the
# significance of the coefficients in a multivariate regression model.

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
# ### Visualization of Multivariate Granger Causality
# A bar chart can effectively show the p-values of each variable's causal influence on the target.

# %%
plt.figure(figsize=(10, 6))
multivariate_results.sort_values(by="P-Value", ascending=True).plot(
    kind="bar", ax=plt.gca()
)
plt.axhline(y=0.05, color="r", linestyle="--", label="5% Significance Level")
plt.title(f"Multivariate Granger Causality P-Values for {target_crypto}")
plt.xlabel("Causing Variable")
plt.ylabel("P-Value")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Time-Varying Granger Causality
# Finally, let's analyze the time-varying Granger causality between Bitcoin (BTC) and a
# less liquid coin, e.g., Solana (SOL)  to see if the relationship is stable over time.
# For more details, see:
# [Time-varying Granger Causality (Wikipedia)](https://en.wikipedia.org/wiki/Granger_causality#Time-varying_Granger_causality)
# We note that this is a rolling window analysis, where we fit a Granger causality model
# and is very much in the spirit of adaptive filtering, where the model is updated with
# each observation (e.g., RLS, EWRLS, Kalman filter, etc). This is useful for
# non-statioanry data, where the relationships between variables may change over time.

# %%
# Define the pair to test
pair_to_test = ("BTCUSDT", "SOLUSDT")

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

# %% [markdown]
# ## Asymmetric Granger Causality
# We will now explore asymmetric Granger causality by decomposing returns into positive and negative components.
# This allows us to investigate if positive movements in one cryptocurrency have a different causal impact than negative movements.


# %%
def decompose_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose returns into positive and negative components."""
    df_decomposed = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_decomposed[f"{col}_pos"] = df[col].apply(lambda x: x if x > 0 else 0)
        df_decomposed[f"{col}_neg"] = df[col].apply(lambda x: x if x < 0 else 0)
    return df_decomposed


# Decompose combined returns
decomposed_returns = decompose_returns(combined_returns);

# Initialize analyzer for decomposed returns
asymmetric_analyzer = GrangerCausalityAnalyzer(decomposed_returns, max_lags=10)

# Run pairwise causality analysis on decomposed returns
asymmetric_pairwise_results = asymmetric_analyzer.run_pairwise_causality()

# Filter for significant asymmetric relationships
significant_asymmetric_results = asymmetric_pairwise_results[
    asymmetric_pairwise_results["significant"]
].copy()

# Display results
print("\nAsymmetric Granger Causality Results (Significant at 5%):")
print(significant_asymmetric_results)

# %% [markdown]
# ### Summarizing Asymmetric Relationships
# We can further analyze these results to identify clear asymmetric patterns.


# %%
def summarize_asymmetry(asymmetric_results: pd.DataFrame) -> pd.DataFrame:
    summary_list = []
    # Iterate through original crypto pairs
    for col1 in combined_returns.columns:
        for col2 in combined_returns.columns:
            if col1 == col2:
                continue

            # Check positive -> positive
            pos_pos = asymmetric_results[
                (asymmetric_results["cause"] == f"{col1}_pos")
                & (asymmetric_results["effect"] == f"{col2}_pos")
            ]
            # Check negative -> negative
            neg_neg = asymmetric_results[
                (asymmetric_results["cause"] == f"{col1}_neg")
                & (asymmetric_results["effect"] == f"{col2}_neg")
            ]
            # Check positive -> negative
            pos_neg = asymmetric_results[
                (asymmetric_results["cause"] == f"{col1}_pos")
                & (asymmetric_results["effect"] == f"{col2}_neg")
            ]
            # Check negative -> positive
            neg_pos = asymmetric_results[
                (asymmetric_results["cause"] == f"{col1}_neg")
                & (asymmetric_results["effect"] == f"{col2}_pos")
            ]

            # Example: A_pos causes B_neg, but A_neg does not cause B_pos
            if not pos_pos.empty and neg_neg.empty:
                summary_list.append(
                    {
                        "Cause": col1,
                        "Effect": col2,
                        "Type": "Positive to Positive (but not Negative to Negative)",
                        "P-Value": pos_pos["min_p_value"].iloc[0],
                    }
                )
            if pos_pos.empty and not neg_neg.empty:
                summary_list.append(
                    {
                        "Cause": col1,
                        "Effect": col2,
                        "Type": "Negative to Negative (but not Positive to Positive)",
                        "P-Value": neg_neg["min_p_value"].iloc[0],
                    }
                )
            if not pos_neg.empty and neg_pos.empty:
                summary_list.append(
                    {
                        "Cause": col1,
                        "Effect": col2,
                        "Type": "Positive to Negative (but not Negative to Positive)",
                        "P-Value": pos_neg["min_p_value"].iloc[0],
                    }
                )
            if pos_neg.empty and not neg_pos.empty:
                summary_list.append(
                    {
                        "Cause": col1,
                        "Effect": col2,
                        "Type": "Negative to Positive (but not Positive to Negative)",
                        "P-Value": neg_pos["min_p_value"].iloc[0],
                    }
                )

    return pd.DataFrame(summary_list)

asymmetric_summary = summarize_asymmetry(significant_asymmetric_results)
print("\nSummary of Asymmetric Causal Relationships:")
print(asymmetric_summary)

# %% [markdown]
# ## Conclusion
# This script had demonstrated how to perform a comprehensive causality analysis between
# crypto currencies. Some elements may require running offline due to time constraints.
# The analysis included:
# * Pairwise Granger causality tests to identify significant causal relationships
# * Multivariate Granger causality to assess the joint influence of multiple
# cryptocurrencies
# * Time-varying Granger causality to analyze the stability of relationships over time

# The primary conclusion of this is that there are significant causal relationships
# between cryptocurrencies from the higher liquidity ones to the lower liquidity ones,
# but that these change in time (as alt-coins become more liquid). This exploration
# feeds into the broader goal of finding suitable features for predicting movements in
# cryptocurrencies, which is a key part of developing trading strategies.

# %% [markdown]
# ## Further References
# ### Reading
# * [Firoozye Linktree](https://linktr.ee/nbfiroozye)
# * [Github](www.github.com/firoozye)
# * [Blog](https://firoozye.github.io/)
#
# ### Classes (at WBS Training)
# * [Fundamentals of Algo Trading](https://rebrand.ly/ndzrcki)
# * [Algorithmic Trading Certificate](https://rebrand.ly/cbadu1x)
#