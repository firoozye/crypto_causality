# src/analysis/granger_causality.py
"""
Granger causality analysis for cryptocurrency returns.

The module provides
    Pairwise Granger Causality:
        Tests causality between all pairs of cryptocurrencies
        Finds optimal lag order for each pair
        Reports significance and p-values
    Multivariate Granger Causality:
        Uses VAR model to test causality in a multivariate setting
        Accounts for interactions between all variables
        Provides test statistics and coefficient p-values
        Automatically selects optimal lag order using AIC
    Summary Statistics:
        Counts significant causal relationships
        Shows which cryptocurrencies are most influential
        Identifies which ones are most affected by others

Key features:

    Handles missing data
    Validates inputs
    Provides detailed logging
    Flexible test statistic selection
    Configurable significance levels
    Optimal lag selection
    Type hints for better code clarity


"""


import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tools.eval_measures import aic, rmse
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrangerCausalityAnalyzer:
    """
    A class to analyze Granger causality relationships in cryptocurrency returns.

    Attributes:
        data (pd.DataFrame): DataFrame containing the return series
        max_lags (int): Maximum number of lags to test
        test_type (str): Type of test statistic to use ('ssr_chi2test', 'ssr_ftest', etc.)
    """

    def __init__(
        self, data: pd.DataFrame, max_lags: int = 10, test_type: str = "ssr_chi2test"
    ):
        """
        Initialize the GrangerCausalityAnalyzer.

        Args:
            data: DataFrame with cryptocurrency returns (each column is a crypto)
            max_lags: Maximum number of lags to test for causality
            test_type: Type of test statistic ('ssr_chi2test', 'ssr_ftest',
                      'ssr_chi2test', 'lrtest')
        """
        self.data = data
        self.max_lags = max_lags
        self.test_type = test_type
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if self.data.isnull().any().any():
            logger.warning("Data contains null values. These will be dropped.")
            self.data = self.data.dropna()

        valid_tests = ["ssr_chi2test", "ssr_ftest", "lrtest", "params_ftest"]
        if self.test_type not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}")

    def run_pairwise_causality(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Run pairwise Granger causality tests between all pairs of cryptocurrencies.

        Args:
            significance_level: P-value threshold for significance

        Returns:
            DataFrame containing test results for each pair
        """
        results = []
        pairs = combinations(self.data.columns, 2)

        for coin1, coin2 in pairs:
            # Test coin1 Granger-causing coin2
            c1_to_c2 = self._test_pair_causality(coin1, coin2)

            # Test coin2 Granger-causing coin1
            c2_to_c1 = self._test_pair_causality(coin2, coin1)

            results.extend(
                [
                    {
                        "cause": coin1,
                        "effect": coin2,
                        "min_p_value": c1_to_c2["min_p_value"],
                        "optimal_lag": c1_to_c2["optimal_lag"],
                        "significant": c1_to_c2["min_p_value"] < significance_level,
                    },
                    {
                        "cause": coin2,
                        "effect": coin1,
                        "min_p_value": c2_to_c1["min_p_value"],
                        "optimal_lag": c2_to_c1["optimal_lag"],
                        "significant": c2_to_c1["min_p_value"] < significance_level,
                    },
                ]
            )

        return pd.DataFrame(results)

    def _test_pair_causality(
        self, cause: str, effect: str
    ) -> Dict[str, Union[float, int]]:
        """
        Test Granger causality between a pair of cryptocurrencies.

        Args:
            cause: Name of potential causing variable
            effect: Name of potential effect variable

        Returns:
            Dictionary containing test results
        """
        data = self.data[[cause, effect]].dropna()
        test_results = grangercausalitytests(data, maxlag=self.max_lags, verbose=False)

        # Extract p-values for the specified test type
        p_values = [
            test_results[i + 1][0][self.test_type][1] for i in range(self.max_lags)
        ]

        min_p_value = min(p_values)
        optimal_lag = p_values.index(min_p_value) + 1

        return {"min_p_value": min_p_value, "optimal_lag": optimal_lag}

    def run_multivariate_causality(
        self, target: str, max_lag: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], int]:
        """
        Run multivariate Granger causality analysis using VAR model.

        Args:
            target: Target cryptocurrency to test causality for
            max_lag: Maximum lag order for VAR model (default: None, uses AIC)

        Returns:
            Tuple containing:
            - Dictionary of test statistics for each variable
            - Dictionary of coefficient p-values for each variable
            - Optimal lag order
        """
        # Prepare data
        data = self.data.copy()

        # Determine optimal lag order if not specified
        if max_lag is None:
            model = VAR(data)
            max_lag = model.select_order(maxlags=self.max_lags).aic

        # Fit VAR model
        model = VAR(data)
        results = model.fit(maxlags=max_lag)

        # Get test statistics and p-values for target variable
        target_idx = list(data.columns).index(target)

        # Extract coefficients and p-values for each variable
        coef_pvals = {}
        test_stats = {}

        for i, col in enumerate(data.columns):
            if col != target:
                # Get coefficients for this variable
                coefs = []
                pvals = []
                for lag in range(max_lag):
                    coef_idx = i + lag * len(data.columns)
                    coefs.append(results.params[coef_idx][target_idx])
                    pvals.append(results.pvalues[coef_idx][target_idx])

                coef_pvals[col] = pvals
                # Use F-test or Chi-square test statistic
                test_stats[col] = results.test_causality(
                    target, [col], kind="f"
                ).test_statistic

        return test_stats, coef_pvals, max_lag

    def get_summary_statistics(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Generate summary statistics of causality relationships.

        Args:
            significance_level: P-value threshold for significance

        Returns:
            DataFrame with summary statistics
        """
        results = self.run_pairwise_causality(significance_level)

        summary = pd.DataFrame(index=self.data.columns)
        summary["causes_count"] = (
            results[results["significant"]].groupby("cause").size()
        )
        summary["affected_by_count"] = (
            results[results["significant"]].groupby("effect").size()
        )
        summary["total_relationships"] = (
            summary["causes_count"] + summary["affected_by_count"]
        )

        return (
            summary.fillna(0)
            .astype(int)
            .sort_values("total_relationships", ascending=False)
        )
