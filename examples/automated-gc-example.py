# examples/automated_gc.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import glob
from typing import Dict, Tuple, List
from src.visualization.causality_viz import CausalityVisualizer
import logging

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AutomatedGrangerAnalyzer:
    def __init__(self, data_dir: str = None):
        """Initialize analyzer with data directory path."""
        if data_dir is None:
            # Get the path relative to the script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(os.path.dirname(script_dir), "data", "processed")
        else:
            self.data_dir = data_dir
        logger.info(f"Initializing analyzer with data directory: {data_dir}")
        self.returns_data = self._load_data()
        logger.info(f"Loaded data for {len(self.returns_data.columns)} cryptocurrencies")
        self.visualizer = CausalityVisualizer(significance_level=0.05)
        
    def _load_data(self) -> pd.DataFrame:
        """Load and prepare return data from parquet files."""
        logger.info("Starting data loading process...")
        all_returns = {}
        
        # Get absolute path for better debugging
        abs_data_dir = os.path.abspath(self.data_dir)
        logger.info(f"Looking for parquet files in: {abs_data_dir}")
        
        # Check if directory exists
        if not os.path.exists(abs_data_dir):
            logger.error(f"Directory does not exist: {abs_data_dir}")
            return pd.DataFrame()
            
        parquet_files = glob.glob(os.path.join(abs_data_dir, "*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files")
        if len(parquet_files) == 0:
            logger.info("Contents of directory:")
            for item in os.listdir(abs_data_dir):
                logger.info(f"  {item}")
        
        for file in parquet_files:
            symbol = os.path.basename(file).split('_')[0]
            logger.info(f"Processing file for {symbol}")
            
            try:
                df = pq.read_table(file).to_pandas()
                logger.info(f"Loaded {len(df)} rows for {symbol}")
                
                if 'log_returns' not in df.columns:
                    logger.info(f"Calculating log returns for {symbol}")
                    df['log_returns'] = np.log(df['close']).diff()
                
                all_returns[symbol] = df['log_returns']
                logger.info(f"Successfully processed {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not all_returns:
            logger.error("No data was loaded!")
            return pd.DataFrame()
            
        returns_df = pd.DataFrame(all_returns).dropna()
        logger.info(f"Final DataFrame shape: {returns_df.shape}")
        logger.info(f"Columns: {returns_df.columns.tolist()}")
        return returns_df

    def test_pair_causality(
        self,
        cause: str,
        effect: str,
        max_lags: int = 10
    ) -> Tuple[Dict, int]:
        """Test Granger causality between a pair of cryptocurrencies."""
        logger.info(f"Testing causality: {cause} -> {effect}")
        
        try:
            pair_data = self.returns_data[[cause, effect]].dropna()
            logger.info(f"Testing pair with {len(pair_data)} observations")
            
            if len(pair_data) < max_lags + 2:
                logger.warning(f"Insufficient data for {cause}->{effect}")
                return None, None
            
            # Find optimal lag order using AIC
            model = VAR(pair_data)
            results = model.select_order(maxlags=max_lags)
            optimal_lag = results.aic.argmin() + 1
            logger.info(f"Optimal lag order: {optimal_lag}")
            
            # Run Granger causality test
            gc_results = grangercausalitytests(
                pair_data[[effect, cause]],
                maxlag=optimal_lag,
                verbose=False
            )
            
            test_stats = {}
            for test_type in ['ssr_chi2test', 'ssr_ftest']:
                test_stats[test_type] = {
                    'stat': gc_results[optimal_lag][0][test_type][0],
                    'pvalue': gc_results[optimal_lag][0][test_type][1]
                }
                logger.info(f"{test_type} p-value: {test_stats[test_type]['pvalue']:.4f}")
            
            return test_stats, optimal_lag
            
        except Exception as e:
            logger.error(f"Error in causality test {cause}->{effect}: {str(e)}")
            return None, None

    def analyze_all_pairs(
        self,
        significance_level: float = 0.05
    ) -> pd.DataFrame:
        """Analyze Granger causality for all cryptocurrency pairs."""
        results = []
        symbols = self.returns_data.columns
        total_pairs = len(symbols) * (len(symbols) - 1)
        logger.info(f"Starting analysis of {total_pairs} pairs")
        completed = 0
        
        for cause in symbols:
            for effect in symbols:
                if cause != effect:
                    completed += 1
                    logger.info(f"Processing pair {completed}/{total_pairs}: {cause}->{effect}")
                    
                    test_stats, opt_lag = self.test_pair_causality(cause, effect)
                    
                    if test_stats is not None and opt_lag is not None:
                        result = {
                            'cause': cause,
                            'effect': effect,
                            'optimal_lag': opt_lag,
                            'f_stat': test_stats['ssr_ftest']['stat'],
                            'p_value': test_stats['ssr_ftest']['pvalue'],
                            'significant': test_stats['ssr_ftest']['pvalue'] < significance_level
                        }
                        results.append(result)
                        logger.info(f"Result: {'Significant' if result['significant'] else 'Not significant'}")
        
        logger.info(f"Analysis completed. Found {len(results)} valid results.")
        return pd.DataFrame(results)

def main():
    """Main analysis pipeline."""
    logger.info("Starting Granger causality analysis")
    
    try:
        # Initialize analyzer
        analyzer = AutomatedGrangerAnalyzer()
        
        # Run analysis
        logger.info("Running pairwise causality tests...")
        results = analyzer.analyze_all_pairs()
        
        if results is None or len(results) == 0:
            logger.warning("No Granger causality results were generated.")
            return
            
        # Print all results
        print("\nAll Granger Causality Tests:")
        print(results.to_string())
        
        # Print significant results
        if 'significant' in results.columns:
            significant_results = results[results['significant'] == True]
            
            if len(significant_results) > 0:
                print("\nSignificant Granger Causality Relationships:")
                for _, row in significant_results.iterrows():
                    print(f"\n{row['cause']} -> {row['effect']}")
                    print(f"  Optimal lag: {row['optimal_lag']}")
                    print(f"  F-statistic: {row['f_stat']:.4f}")
                    print(f"  P-value: {row['p_value']:.4f}")
            else:
                print("\nNo significant Granger causality relationships found.")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()