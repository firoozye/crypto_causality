import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple, Dict

class AutomatedGrangerCausality:
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
    
    def select_optimal_lag(self, data: pd.DataFrame) -> int:
        """
        Select optimal lag order using AIC for the VAR model
        """
        model = VAR(data)
        results = model.select_order(maxlags=self.max_lags)
        return results.aic
        
    def test_causality(self, x: pd.Series, y: pd.Series) -> Tuple[Dict, int]:
        """
        Perform automated Granger causality test
        
        Args:
            x: Potential cause variable
            y: Effect variable
            
        Returns:
            Tuple containing:
            - Dictionary with test results
            - Optimal lag order
        """
        # Prepare data
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Find optimal lag order using VAR model
        optimal_lag = self.select_optimal_lag(data)
        
        # Run Granger causality test with optimal lag
        gc_results = grangercausalitytests(
            data[['y', 'x']], # Note: y must come first
            maxlag=optimal_lag,
            verbose=False
        )
        
        # Extract results for the optimal lag
        test_stats = {}
        for test_type in ['ssr_chi2test', 'ssr_ftest']:
            test_stats[test_type] = {
                'stat': gc_results[optimal_lag][0][test_type][0],
                'pvalue': gc_results[optimal_lag][0][test_type][1]
            }
            
        return test_stats, optimal_lag

    def interpret_results(self, test_stats: Dict, significance_level: float = 0.05) -> str:
        """
        Interpret the Granger causality test results
        """
        # Use F-test results for interpretation
        pvalue = test_stats['ssr_ftest']['pvalue']
        
        if pvalue < significance_level:
            return f"Reject null hypothesis. Evidence of Granger causality (p-value: {pvalue:.4f})"
        else:
            return f"Fail to reject null hypothesis. No evidence of Granger causality (p-value: {pvalue:.4f})"

# Example usage:
"""
gc_tester = AutomatedGrangerCausality(max_lags=10)
test_stats, optimal_lag = gc_tester.test_causality(x_series, y_series)
interpretation = gc_tester.interpret_results(test_stats)
print(f"Optimal lag order: {optimal_lag}")
print(interpretation)
"""