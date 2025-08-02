# examples/granger_analysis.py

import pandas as pd
import pyarrow.parquet as pq
import glob
import os
from src.analysis.granger_causality import GrangerCausalityAnalyzer

def load_and_prepare_data(data_dir="./data/processed/"):
    """Load and prepare cryptocurrency return data."""
    # Load all return series into a single DataFrame
    all_returns = {}
    
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split('_')[0]
        df = pq.read_table(file).to_pandas()
        all_returns[symbol] = np.log(df['close']).diff()
    
    # Combine into a single DataFrame
    returns_df = pd.DataFrame(all_returns)
    returns_df = returns_df.dropna()
    
    return returns_df

def main():
    # Load data
    returns_df = load_and_prepare_data()
    
    # Initialize analyzer
    analyzer = GrangerCausalityAnalyzer(
        data=returns_df,
        max_lags=10,
        test_type='ssr_chi2test'
    )
    
    # Run pairwise causality tests
    print("\nPairwise Granger Causality Results:")
    pairwise_results = analyzer.run_pairwise_causality(significance_level=0.05)
    print(pairwise_results)
    
    # Run multivariate causality test for BTC
    print("\nMultivariate Granger Causality Results for BTC:")
    test_stats, coef_pvals, opt_lag = analyzer.run_multivariate_causality(
        target='BTCUSDT'
    )
    print(f"Optimal lag order: {opt_lag}")
    print("\nTest statistics:")
    print(test_stats)
    print("\nCoefficient p-values:")
    print(coef_pvals)
    
    # Get summary statistics
    print("\nCausality Summary Statistics:")
    summary_stats = analyzer.get_summary_statistics()
    print(summary_stats)

if __name__ == "__main__":
    main()
