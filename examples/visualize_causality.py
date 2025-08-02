# examples/visualize_causality.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import glob
import os
from src.analysis.granger_causality import GrangerCausalityAnalyzer
from src.visualization.causality_viz import CausalityVisualizer

def load_and_prepare_data(data_dir="./data/processed/"):
    """Load and prepare cryptocurrency return data."""
    all_returns = {}
    
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split('_')[0]
        df = pq.read_table(file).to_pandas()
        all_returns[symbol] = np.log(df['close']).diff()
    
    returns_df = pd.DataFrame(all_returns)
    returns_df = returns_df.dropna()
    
    return returns_df

def main():
    # Load data
    returns_df = load_and_prepare_data()
    
    # Run Granger causality analysis
    analyzer = GrangerCausalityAnalyzer(
        data=returns_df,
        max_lags=10,
        test_type='ssr_chi2test'
    )
    
    causality_results = analyzer.run_pairwise_causality()
    summary_stats = analyzer.get_summary_statistics()
    
    # Create visualizations
    visualizer = CausalityVisualizer(significance_level=0.05)
    
    # 1. Heatmap of p-values
    print("Plotting causality heatmap...")
    visualizer.plot_causality_heatmap(
        causality_results,
        title='Granger Causality P-values Between Cryptocurrencies'
    )
    
    # 2. Network diagram
    print("Plotting causality network...")
    visualizer.plot_causality_network(
        causality_results,
        layout='spring',
        title='Cryptocurrency Causality Network'
    )
    
    # 3. Summary bar plots
    print("Plotting summary statistics...")
    visualizer.plot_summary_bars(
        summary_stats,
        title='Causality Influence Summary'
    )
    
    # 4. Lag distribution
    print("Plotting lag distribution...")
    visualizer.plot_lag_distribution(
        causality_results,
        title='Distribution of Optimal Lag Orders'
    )

if __name__ == "__main__":
    main()
