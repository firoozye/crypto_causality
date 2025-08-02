# src/analysis/stationarity.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, kpss
import logging

logger = logging.getLogger(__name__)

class StationarityTester:
    """Tests for stationarity in cryptocurrency return series."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize StationarityTester.
        
        Args:
            data: DataFrame with cryptocurrency returns
        """
        self.data = data
    
    def run_adf_test(
        self,
        regression: str = 'c'
    ) -> Dict[str, Dict[str, float]]:
        """
        Run Augmented Dickey-Fuller test for each series.
        
        Args:
            regression: Type of regression to include in test
            
        Returns:
            Dictionary with test results for each series
        """
        results = {}
        
        for column in self.data.columns:
            series = self.data[column].dropna()
            adf_result = adfuller(series, regression=regression)
            
            results[column] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4]
            }
        
        return results
    
    def run_kpss_test(
        self,
        regression: str = 'c'
    ) -> Dict[str, Dict[str, float]]:
        """
        Run KPSS test for each series.
        
        Args:
            regression: Type of regression to include in test
            
        Returns:
            Dictionary with test results for each series
        """
        results = {}
        
        for column in self.data.columns:
            series = self.data[column].dropna()
            kpss_result = kpss(series, regression=regression)
            
            results[column] = {
                'kpss_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3]
            }
        
        return results
    
    def get_stationarity_summary(self) -> pd.DataFrame:
        """Generate summary of stationarity tests."""
        adf_results = self.run_adf_test()
        kpss_results = self.run_kpss_test()
        
        summary = pd.DataFrame(index=self.data.columns)
        
        for column in self.data.columns:
            summary.loc[column, 'ADF_statistic'] = adf_results[column]['adf_statistic']
            summary.loc[column, 'ADF_p_value'] = adf_results[column]['p_value']
            summary.loc[column, 'KPSS_statistic'] = kpss_results[column]['kpss_statistic']
            summary.loc[column, 'KPSS_p_value'] = kpss_results[column]['p_value']
            
        summary['is_stationary'] = (
            (summary['ADF_p_value'] < 0.05) & 
            (summary['KPSS_p_value'] > 0.05)
        )
        
        return summary
