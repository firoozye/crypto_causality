# src/analysis/outliers.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class OutlierAnalyzer:
    """Analyzes outliers in cryptocurrency return series."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        methods: List[str] = ['zscore', 'iqr', 'mad']
    ):
        """
        Initialize OutlierAnalyzer.
        
        Args:
            data: DataFrame with cryptocurrency returns
            methods: List of outlier detection methods to use
        """
        self.data = data
        self.methods = methods
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        valid_methods = ['zscore', 'iqr', 'mad']
        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(f"Method {method} not in {valid_methods}")
    
    def detect_outliers(
        self,
        threshold: float = 3.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers using specified methods.
        
        Args:
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier masks for each method
        """
        results = {}
        
        for method in self.methods:
            if method == 'zscore':
                results[method] = self._zscore_outliers(threshold)
            elif method == 'iqr':
                results[method] = self._iqr_outliers()
            elif method == 'mad':
                results[method] = self._mad_outliers(threshold)
        
        return results
    
    def _zscore_outliers(self, threshold: float) -> pd.DataFrame:
        """Detect outliers using z-score method."""
        z_scores = np.abs(stats.zscore(self.data))
        return pd.DataFrame(
            z_scores > threshold,
            columns=self.data.columns,
            index=self.data.index
        )
    
    def _iqr_outliers(self) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        return pd.DataFrame(
            (self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR)),
            columns=self.data.columns,
            index=self.data.index
        )
    
    def _mad_outliers(self, threshold: float) -> pd.DataFrame:
        """Detect outliers using Median Absolute Deviation method."""
        median = self.data.median()
        mad = stats.median_abs_deviation(self.data)
        return pd.DataFrame(
            np.abs(self.data - median) > threshold * mad,
            columns=self.data.columns,
            index=self.data.index
        )
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics of outliers."""
        outliers = self.detect_outliers()
        summary = pd.DataFrame()
        
        for method, mask in outliers.items():
            summary[f'{method}_count'] = mask.sum()
            summary[f'{method}_percentage'] = (mask.sum() / len(mask) * 100)
        
        return summary
