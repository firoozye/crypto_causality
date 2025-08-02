# src/analysis/causality.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .granger_causality import GrangerCausalityAnalyzer
from statsmodels.stats.correlation_tools import corr_clust
from statsmodels.stats.diagnostic import acorr_ljungbox
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class CausalityAnalyzer:
    """
    Comprehensive causality analysis including Granger, correlation clustering,
    and transfer entropy methods.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        max_lags: int = 10,
        alpha: float = 0.05
    ):
        """
        Initialize CausalityAnalyzer.
        
        Args:
            data: DataFrame with cryptocurrency returns
            max_lags: Maximum number of lags to test
            alpha: Significance level for tests
        """
        self.data = data
        self.max_lags = max_lags
        self.alpha = alpha
        self.granger = GrangerCausalityAnalyzer(data, max_lags)
    
    def analyze_all_causality(self) -> Dict[str, pd.DataFrame]:
        """Run comprehensive causality analysis."""
        results = {}
        
        # Granger causality
        results['granger'] = self.granger.run_pairwise_causality(
            significance_level=self.alpha
        )
        
        # Correlation clustering
        results['correlation'] = self._analyze_correlation_structure()
        
        # Instantaneous causality
        results['instantaneous'] = self._analyze_instantaneous_causality()
        
        return results
    
    def _analyze_correlation_structure(self) -> pd.DataFrame:
        """Analyze correlation structure using hierarchical clustering."""
        # Calculate correlation matrix
        corr_matrix = self.data.corr()
        
        # Perform correlation clustering
        clustering = corr_clust(corr_matrix.values)
        
        # Create DataFrame with results
        clusters = pd.DataFrame(
            clustering,
            index=corr_matrix.index,
            columns=['cluster', 'p_value']
        )
        
        return clusters
    
    def _analyze_instantaneous_causality(self) -> pd.DataFrame:
        """Analyze instantaneous causality using Ljung-Box test."""
        results = []
        
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 >= col2:  # Only need upper triangle
                    continue
                    
                # Calculate residuals from VAR model
                series1 = self.data[col1]
                series2 = self.data[col2]
                
                # Perform Ljung-Box test on cross-correlation of residuals
                lb_test = acorr_ljungbox(
                    series1 * series2,
                    lags=self.max_lags,
                    return_df=True
                )
                
                results.append({
                    'series1': col1,
                    'series2': col2,
                    'lb_statistic': lb_test['lb_stat'].mean(),
                    'lb_pvalue': lb_test['lb_pvalue'].mean(),
                    'significant': lb_test['lb_pvalue'].mean() < self.alpha
                })
        
        return pd.DataFrame(results)
    
    def create_causality_network(self) -> nx.DiGraph:
        """Create a directed graph of causal relationships."""
        # Get Granger causality results
        granger_results = self.granger.run_pairwise_causality(
            significance_level=self.alpha
        )
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for col in self.data.columns:
            G.add_node(col)
        
        # Add edges for significant relationships
        significant_causes = granger_results[granger_results['significant']]
        for _, row in significant_causes.iterrows():
            G.add_edge(
                row['cause'],
                row['effect'],
                weight=-np.log10(row['min_p_value'])  # Edge weight based on p-value
            )
        
        return G
    
    def get_causality_metrics(self) -> pd.DataFrame:
        """Calculate various causality metrics for each cryptocurrency."""
        G = self.create_causality_network()
        
        metrics = pd.DataFrame(index=self.data.columns)
        
        # Calculate network metrics
        metrics['out_degree'] = pd.Series(dict(G.out_degree()))
        metrics['in_degree'] = pd.Series(dict(G.in_degree()))
        metrics['betweenness'] = pd.Series(nx.betweenness_centrality(G))
        metrics['pagerank'] = pd.Series(nx.pagerank(G))
        
        # Add descriptions
        metrics['is_causal_hub'] = metrics['out_degree'] > metrics['out_degree'].mean()
        metrics['is_effect_hub'] = metrics['in_degree'] > metrics['in_degree'].mean()
        
        return metrics
