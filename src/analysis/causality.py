# src/analysis/causality.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .granger_causality import GrangerCausalityAnalyzer
from statsmodels.stats.diagnostic import acorr_ljungbox
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)

class CausalityAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        max_lags: int = 10,
        alpha: float = 0.05
    ):
        self.data = data
        self.max_lags = max_lags
        self.alpha = alpha
        self.granger = GrangerCausalityAnalyzer(data, max_lags)
    
    def _analyze_correlation_structure(self) -> pd.DataFrame:
        """Analyze correlation structure using hierarchical clustering."""
        # Calculate correlation matrix
        corr_matrix = self.data.corr()
        
        # Convert correlation matrix to distance matrix
        dist_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        condensed_dist = squareform(dist_matrix)
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        # Form clusters
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        # Create DataFrame with results
        cluster_df = pd.DataFrame(
            index=corr_matrix.index,
            columns=['cluster']
        )
        cluster_df['cluster'] = clusters
        
        return cluster_df
    
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
    
    def _analyze_instantaneous_causality(self) -> pd.DataFrame:
        """Analyze instantaneous causality using Ljung-Box test."""
        results = []
        
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 >= col2:
                    continue
                    
                series1 = self.data[col1]
                series2 = self.data[col2]
                
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
        granger_results = self.granger.run_pairwise_causality(
            significance_level=self.alpha
        )
        
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
                weight=-np.log10(row['min_p_value'])
            )
        
        return G