# examples/tvgc_example.py

import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
from scipy import stats
from typing import Dict
import logging
import sys

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Time-varying Granger implementation from previous code
def rolling_granger_causality(y, x, max_lag=5, window_size=500):
    """
    Implements time-varying Granger causality test using rolling windows
    """
    if len(y) != len(x):
        raise ValueError("Series must be of equal length")
        
    # Initialize results storage
    results = []
    dates = []
    
    # Create lagged versions of both series
    y_lags = pd.DataFrame({f'y_lag_{i}': y.shift(i) for i in range(1, max_lag + 1)})
    x_lags = pd.DataFrame({f'x_lag_{i}': x.shift(i) for i in range(1, max_lag + 1)})
    
    # Combine all data
    data = pd.concat([y, y_lags, x_lags], axis=1)
    data = data.dropna()
    
    # Rolling window analysis
    for start in range(0, len(data) - window_size):
        window_data = data.iloc[start:start + window_size]
        
        # Restricted model (only y lags)
        y_window = window_data.iloc[:, 0]
        y_lags_window = window_data.iloc[:, 1:max_lag+1]
        restricted_model = sm.OLS(y_window, sm.add_constant(y_lags_window)).fit()
        rss_restricted = restricted_model.ssr
        
        # Unrestricted model (y lags and x lags)
        x_lags_window = window_data.iloc[:, max_lag+1:]
        full_x = pd.concat([y_lags_window, x_lags_window], axis=1)
        unrestricted_model = sm.OLS(y_window, sm.add_constant(full_x)).fit()
        rss_unrestricted = unrestricted_model.ssr
        
        # Calculate F-statistic
        n = window_size
        q = max_lag  # number of restrictions
        k = 2 * max_lag  # number of parameters in unrestricted model
        f_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - k - 1))
        p_value = 1 - stats.f.cdf(f_stat, q, n - k - 1)
        
        results.append({
            'date': data.index[start + window_size - 1],
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)

def analyze_tvgc(crypto_data, pairs_to_test, window_size=500, test_mode=False):
    """
    Analyze time-varying Granger causality for multiple cryptocurrency pairs
    
    Args:
        crypto_data: Dictionary of cryptocurrency data
        pairs_to_test: List of pairs to test
        window_size: Size of rolling window
        test_mode: If True, only analyze first 1000 data points for quick testing
    """
    tvgc_results = {}
    total_pairs = len(pairs_to_test) * 2  # Both directions
    completed = 0
    
    for pair in pairs_to_test:
        symbol1, symbol2 = pair
        print(f"\nAnalyzing TVGC for {symbol1} -> {symbol2} ({completed}/{total_pairs} pairs completed)")
        
        # Get returns data
        returns1 = crypto_data[symbol1]['returns']
        returns2 = crypto_data[symbol2]['returns']
        
        if test_mode:
            returns1 = returns1[:1000]  # Only use first 1000 points for testing
            returns2 = returns2[:1000]
            print("TEST MODE: Using first 1000 data points only")
        
        print(f"Processing forward causality: {symbol1} -> {symbol2}")
        forward_results = rolling_granger_causality(
            returns2.dropna(), 
            returns1.dropna(),
            window_size=window_size
        )
        completed += 1
        print(f"Forward causality complete. Initial p-value: {forward_results['p_value'].iloc[0]:.4f}")
        
        print(f"Processing backward causality: {symbol2} -> {symbol1}")
        backward_results = rolling_granger_causality(
            returns1.dropna(),
            returns2.dropna(),
            window_size=window_size
        )
        completed += 1
        print(f"Backward causality complete. Initial p-value: {backward_results['p_value'].iloc[0]:.4f}")
        
        tvgc_results[f"{symbol1}->{symbol2}"] = forward_results
        tvgc_results[f"{symbol2}->{symbol1}"] = backward_results
        
        # Print summary of this pair
        print(f"\nSummary for pair {symbol1}/{symbol2}:")
        print(f"{symbol1}->{symbol2} avg p-value: {forward_results['p_value'].mean():.4f}")
        print(f"{symbol2}->{symbol1} avg p-value: {backward_results['p_value'].mean():.4f}")
        
    return tvgc_results

def plot_tvgc_results(tvgc_results, title="Time-Varying Granger Causality Results"):
    """
    Plot the time-varying Granger causality results
    """
    plt.figure(figsize=(15, 10))
    
    for pair, results in tvgc_results.items():
        # Plot significance regions
        plt.fill_between(results['date'], 
                        0, 
                        1, 
                        where=results['significant'],
                        alpha=0.3,
                        label=f"{pair} (significant regions)")
        
        # Plot p-values
        plt.plot(results['date'], 
                results['p_value'], 
                label=f"{pair} (p-value)",
                alpha=0.7)
    
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% significance level')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('P-value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def summarize_tvgc_results(tvgc_results):
    """
    Generate summary statistics for TVGC results
    """
    summary_data = []
    
    for pair, results in tvgc_results.items():
        summary = {
            'Pair': pair,
            'Avg p-value': results['p_value'].mean(),
            'Min p-value': results['p_value'].min(),
            'Max p-value': results['p_value'].max(),
            'Significant Windows (%)': (results['significant'].sum() / len(results)) * 100,
            'Total Windows': len(results)
        }
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)

def load_and_prepare_data(data_dir: str = "./data/processed/") -> Dict[str, pd.DataFrame]:
    """Load and prepare cryptocurrency data from parquet files."""
    all_data = {}
    
    # Load all parquet files
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split('_')[0]
        print(f"Loading data for {symbol}...")
        
        # Read parquet file
        df = pq.read_table(file).to_pandas()
        
        # Calculate log returns if not already present
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close']).diff()
        
        all_data[symbol] = {
            'price': df['close'],
            'returns': df['log_returns']
        }
        
        print(f"Loaded {len(df)} rows for {symbol}")
    
    return all_data

def run_tvgc_analysis(data: Dict[str, pd.DataFrame], window_size: int = 500, test_mode: bool = False) -> None:
    """Run time-varying Granger causality analysis."""
    # Get all pairs of cryptocurrencies
    symbols = list(data.keys())
    pairs = list(combinations(symbols, 2))
    
    print(f"Analyzing {len(pairs)} cryptocurrency pairs...")
    
    # Run TVGC analysis
    tvgc_results = analyze_tvgc(data, pairs, window_size=window_size, test_mode=test_mode)
    
    # Create output directory if it doesn't exist
    output_dir = "./results/tvgc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    print("Plotting TVGC results...")
    plt.figure(figsize=(15, 10))
    plot_tvgc_results(tvgc_results, "Time-Varying Granger Causality Analysis")
    plt.savefig(os.path.join(output_dir, "tvgc_results.png"))
    plt.close()
    
    # Generate and save summary statistics
    print("Generating summary statistics...")
    summary = summarize_tvgc_results(tvgc_results)
    
    # Save summary to CSV
    summary.to_csv(os.path.join(output_dir, "tvgc_summary.csv"), index=False)
    
    # Print summary to console
    print("\nTime-Varying Granger Causality Summary:")
    print(summary.to_string(index=False))

def main():
    """Main execution function."""
    print("Starting TVGC analysis...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced data')
    args = parser.parse_args()
    
    # Load data
    data = load_and_prepare_data()
    print(f"\nLoaded data for {len(data)} cryptocurrencies:")
    for symbol, data_dict in data.items():
        print(f"{symbol}: {len(data_dict['returns'])} data points")
    
    # Run analysis
    window_size = 100 if args.test else 500  # Smaller window for test mode
    run_tvgc_analysis(data, window_size=window_size, test_mode=args.test)
    
    print("TVGC analysis completed!")

if __name__ == "__main__":
    main()
