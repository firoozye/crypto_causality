import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def rolling_granger_causality(y, x, max_lag=5, window_size=500):
    """
    Implements time-varying Granger causality test using rolling windows
    
    Parameters:
    -----------
    y : pd.Series
        Target variable (the variable being predicted)
    x : pd.Series
        Potential causal variable
    max_lag : int
        Maximum number of lags to test
    window_size : int
        Size of the rolling window
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing F-statistics and p-values for each window
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
        restricted_model = OLS(y_window, sm.add_constant(y_lags_window)).fit()
        rss_restricted = restricted_model.ssr
        
        # Unrestricted model (y lags and x lags)
        x_lags_window = window_data.iloc[:, max_lag+1:]
        full_x = pd.concat([y_lags_window, x_lags_window], axis=1)
        unrestricted_model = OLS(y_window, sm.add_constant(full_x)).fit()
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

def analyze_tvgc(crypto_data, pairs_to_test, window_size=500):
    """
    Analyze time-varying Granger causality for multiple cryptocurrency pairs
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary containing cryptocurrency return data
    pairs_to_test : list of tuples
        List of cryptocurrency pairs to test
    window_size : int
        Size of the rolling window
        
    Returns:
    --------
    dict
        Dictionary containing TVGC results for each pair
    """
    tvgc_results = {}
    
    for pair in pairs_to_test:
        symbol1, symbol2 = pair
        print(f"\nAnalyzing TVGC for {symbol1} -> {symbol2}")
        
        # Get returns data
        returns1 = crypto_data[symbol1]['returns']
        returns2 = crypto_data[symbol2]['returns']
        
        # Test both directions
        forward_results = rolling_granger_causality(
            returns2.dropna(), 
            returns1.dropna(),
            window_size=window_size
        )
        backward_results = rolling_granger_causality(
            returns1.dropna(),
            returns2.dropna(),
            window_size=window_size
        )
        
        tvgc_results[f"{symbol1}->{symbol2}"] = forward_results
        tvgc_results[f"{symbol2}->{symbol1}"] = backward_results
        
    return tvgc_results

def plot_tvgc_results(tvgc_results, title="Time-Varying Granger Causality Results"):
    """
    Plot the time-varying Granger causality results
    
    Parameters:
    -----------
    tvgc_results : dict
        Dictionary containing TVGC results for each pair
    title : str
        Title for the plot
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

# Calculate summary statistics for TVGC results
def summarize_tvgc_results(tvgc_results):
    """
    Generate summary statistics for TVGC results
    
    Parameters:
    -----------
    tvgc_results : dict
        Dictionary containing TVGC results for each pair
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics for each pair
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
