# examples/run_analysis.py

import pandas as pd
import pyarrow.parquet as pq
import os
import logging
from src.data.processor import DataProcessor
from src.analysis.causality import CausalityAnalyzer
from src.analysis.outliers import OutlierAnalyzer
from src.analysis.stationarity import StationarityTester
from src.utils.helpers import calculate_returns
from src.config import PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load and prepare data from processed parquet files."""
    all_data = {}
    
    # Load all parquet files
    for file in os.listdir(PROCESSED_DATA_PATH):
        if file.endswith('.parquet'):
            symbol = file.split('_')[0]
            file_path = os.path.join(PROCESSED_DATA_PATH, file)
            df = pq.read_table(file_path).to_pandas()
            all_data[symbol] = df
    
    # Extract and combine return series
    returns_data = pd.DataFrame()
    for symbol, df in all_data.items():
        returns_data[symbol] = df['log_returns']
    
    return returns_data.dropna()

def run_stationarity_analysis(data: pd.DataFrame) -> None:
    """Run and print stationarity analysis."""
    logger.info("Running stationarity analysis...")
    
    tester = StationarityTester(data)
    summary = tester.get_stationarity_summary()
    
    print("\nStationarity Analysis Results:")
    print(summary)
    
    return summary

def run_outlier_analysis(data: pd.DataFrame) -> None:
    """Run and print outlier analysis."""
    logger.info("Running outlier analysis...")
    
    analyzer = OutlierAnalyzer(data)
    summary = analyzer.get_summary_statistics()
    
    print("\nOutlier Analysis Results:")
    print(summary)
    
    return summary

def run_causality_analysis(data: pd.DataFrame) -> None:
    """Run and print causality analysis."""
    logger.info("Running causality analysis...")
    
    analyzer = CausalityAnalyzer(data)
    results = analyzer.analyze_all_causality()
    metrics = analyzer.get_causality_metrics()
    
    print("\nCausality Analysis Results:")
    print("\nGranger Causality Summary:")
    print(results['granger'])
    
    print("\nCorrelation Structure:")
    print(results['correlation'])
    
    print("\nCausality Metrics:")
    print(metrics)
    
    return results, metrics

def main():
    """Main analysis pipeline."""
    # Load data
    logger.info("Loading data...")
    data = load_data()
    
    # Run analyses
    stationarity_results = run_stationarity_analysis(data)
    outlier_results = run_outlier_analysis(data)
    causality_results, causality_metrics = run_causality_analysis(data)
    
    # Save results if needed
    # ... (add code to save results)

if __name__ == "__main__":
    main()
