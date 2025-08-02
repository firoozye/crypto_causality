# src/visualization/causality_viz.py
'''
This visualization module provides several ways to visualize the causality relationships:

    Causality Heatmap:
        Shows p-values for all pairs
        Color coding indicates strength of relationship
        Easy to spot patterns of influence
    Causality Network (Digraph):
        Directed graph showing significant relationships
        Edge thickness indicates strength of causality
        Node size can indicate importance
        Different layout options (spring, circular, etc.)
        Arrows show direction of causality
    Summary Bar Plots:
        Shows how many other cryptos each one influences
        Shows how many other cryptos influence each one
        Easy to identify most influential cryptocurrencies
    Lag Distribution:
        Shows distribution of optimal lag orders
        Helps understand typical time delays in causality

The digraph visualization (plot_causality_network) is particularly interesting because it:

    Shows direction of influence with arrows
    Uses edge thickness to show strength of relationship
    Offers different layout algorithms for better visualization
    Can handle complex networks of relationships
    Makes it easy to identify central/influential nodes



'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional
import logging

class CausalityVisualizer:
    """Visualization tools for Granger causality analysis results."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)

    def plot_causality_heatmap(
        self,
        causality_results: pd.DataFrame,
        title: Optional[str] = None
    ) -> None:
        """
        Create a heatmap of p-values for Granger causality relationships.

        Args:
            causality_results: DataFrame from GrangerCausalityAnalyzer.run_pairwise_causality()
            title: Optional title for the plot
        """
        # Pivot the results into a matrix
        pivot_df = causality_results.pivot(
            index='cause',
            columns='effect',
            values='min_p_value'
        )

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_df,
            annot=True,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=0.1,
            center=self.significance_level,
            fmt='.3f'
        )

        plt.title(title or 'Granger Causality P-values Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_causality_network(
        self,
        causality_results: pd.DataFrame,
        layout: str = 'spring',
        min_edge_width: float = 1.0,
        max_edge_width: float = 5.0,
        node_size: float = 2000,
        title: Optional[str] = None
    ) -> None:
        """
        Create a directed graph visualization of significant causal relationships.

        Args:
            causality_results: DataFrame from GrangerCausalityAnalyzer.run_pairwise_causality()
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
            min_edge_width: Minimum edge width for visualization
            max_edge_width: Maximum edge width for visualization
            node_size: Size of nodes in the graph
            title: Optional title for the plot
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        unique_cryptos = pd.concat([
            causality_results['cause'],
            causality_results['effect']
        ]).unique()
        G.add_nodes_from(unique_cryptos)

        # Add edges for significant relationships
        significant_results = causality_results[
            causality_results['min_p_value'] < self.significance_level
        ]

        # Calculate edge weights based on -log(p-value)
        max_weight = -np.log(significant_results['min_p_value'].min())
        min_weight = -np.log(self.significance_level)

        for _, row in significant_results.iterrows():
            weight = -np.log(row['min_p_value'])
            # Normalize weight for edge width
            edge_width = (
                (weight - min_weight)
                / (max_weight - min_weight)
                * (max_edge_width - min_edge_width)
                + min_edge_width
            )
            G.add_edge(
                row['cause'],
                row['effect'],
                weight=weight,
                width=edge_width
            )

        # Create plot
        plt.figure(figsize=(12, 8))

        # Set layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=node_size,
            alpha=0.7
        )

        # Draw edges with varying widths
        edges = G.edges(data=True)
        edge_widths = [d['width'] for _, _, d in edges]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=edge_widths,
            edge_color='gray',
            arrowsize=20,
            alpha=0.6
        )

        # Add labels
        nx.draw_networkx_labels(G, pos)

        plt.title(title or 'Granger Causality Network')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_summary_bars(
        self,
        summary_stats: pd.DataFrame,
        title: Optional[str] = None
    ) -> None:
        """
        Create bar plots showing causality influence summary statistics.

        Args:
            summary_stats: DataFrame from GrangerCausalityAnalyzer.get_summary_statistics()
            title: Optional title for the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot causes count
        summary_stats['causes_count'].plot(
            kind='bar',
            ax=ax1,
            color='skyblue'
        )
        ax1.set_title('Number of Cryptocurrencies Caused')
        ax1.set_ylabel('Count')
        plt.setp(ax1.get_xticklabels(), rotation=45)

        # Plot affected by count
        summary_stats['affected_by_count'].plot(
            kind='bar',
            ax=ax2,
            color='lightgreen'
        )
        ax2.set_title('Number of Cryptocurrencies Affected By')
        ax2.set_ylabel('Count')
        plt.setp(ax2.get_xticklabels(), rotation=45)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plot_lag_distribution(
        self,
        causality_results: pd.DataFrame,
        title: Optional[str] = None
    ) -> None:
        """
        Plot the distribution of optimal lag orders for significant relationships.

        Args:
            causality_results: DataFrame from GrangerCausalityAnalyzer.run_pairwise_causality()
            title: Optional title for the plot
        """
        significant_results = causality_results[
            causality_results['min_p_value'] < self.significance_level
        ]

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=significant_results,
            x='optimal_lag',
            bins=range(
                significant_results['optimal_lag'].min(),
                significant_results['optimal_lag'].max() + 2,
                1
            )
        )

        plt.title(title or 'Distribution of Optimal Lag Orders')
        plt.xlabel('Optimal Lag')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
