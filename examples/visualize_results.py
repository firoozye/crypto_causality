# examples/visualize_results.py
'''
Visualizing Results
    Completed the plot_return_distributions method:
        Creates subplots for each cryptocurrency
        Shows histogram with density
        Overlays normal distribution curve
        Handles multiple cryptocurrencies efficiently
    Added the plot_qq_plots method:
        Creates Q-Q plots for each cryptocurrency
        Helps assess normality of returns
        Uses scipy's probplot function
    Added the plot_lead_lag_relationships method:
        Visualizes Granger causality results
        Shows direction and strength of relationships
        Uses circular layout for better visualization
    Added a complete main() function:
        Demonstrates how to use the visualizer
        Loads data from parquet files
        Runs causality analysis
        Creates all visualizations
    Added proper error handling and logging throughout

Each visualization method includes:

    Consistent styling
    Save functionality
    Proper cleanup of matplotlib figures
    Handling of different numbers of cryptocurrencies
    Clear titles and labels


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.causality_viz import CausalityVisualizer
from src.analysis.causality import CausalityAnalyzer
from src.utils.helpers import calculate_returns
import os
import logging
from typing import Dict, Optional
import networkx as nx
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Visualizes analysis results with various plots and diagrams."""

    def __init__(
        self,
        data: pd.DataFrame,
        results: Optional[Dict] = None,
        output_dir: str = "./plots"
    ):
        """
        Initialize ResultsVisualizer.

        Args:
            data: DataFrame with cryptocurrency returns
            results: Dictionary of analysis results (optional)
            output_dir: Directory to save plots
        """
        self.data = data
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def plot_all_results(self, save: bool = True) -> None:
        """Generate all visualizations."""
        logger.info("Generating all visualizations...")

        # Time series plots
        self.plot_return_series(save)
        self.plot_rolling_statistics(save)

        # Correlation and causality plots
        self.plot_correlation_matrix(save)
        self.plot_causality_network(save)
        self.plot_lead_lag_relationships(save)

        # Distribution plots
        self.plot_return_distributions(save)
        self.plot_qq_plots(save)

    def plot_return_series(self, save: bool = True) -> None:
        """Plot return series for all cryptocurrencies."""
        plt.figure(figsize=(15, 8))

        for column in self.data.columns:
            plt.plot(self.data.index, self.data[column], label=column, alpha=0.7)

        plt.title('Cryptocurrency Returns')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'return_series.png'))
        plt.close()

    def plot_rolling_statistics(self, save: bool = True) -> None:
        """Plot rolling mean and volatility."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        window = 30

        # Rolling mean
        for column in self.data.columns:
            rolling_mean = self.data[column].rolling(window=window).mean()
            ax1.plot(self.data.index, rolling_mean, label=column)

        ax1.set_title(f'{window}-Day Rolling Mean')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mean Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rolling volatility
        for column in self.data.columns:
            rolling_vol = self.data[column].rolling(window=window).std()
            ax2.plot(self.data.index, rolling_vol, label=column)

        ax2.set_title(f'{window}-Day Rolling Volatility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Standard Deviation')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'rolling_statistics.png'))
        plt.close()

    def plot_correlation_matrix(self, save: bool = True) -> None:
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(10, 8))

        corr_matrix = self.data.corr()
        mask = np.triu(np.ones_like(corr_matrix), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdYlBu',
            center=0,
            fmt='.2f'
        )

        plt.title('Correlation Matrix')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
        plt.close()

    def plot_causality_network(self, save: bool = True) -> None:
        """Plot causality network diagram."""
        if self.results is None:
            analyzer = CausalityAnalyzer(self.data)
            G = analyzer.create_causality_network()
        else:
            G = self.results.get('causality_network')

        if G is None:
            logger.warning("No causality network available to plot")
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1000,
            arrowsize=20,
            font_size=10
        )

        plt.title('Causality Network')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'causality_network.png'))
        plt.close()

    def plot_return_distributions(self, save: bool = True) -> None:
        """Plot return distributions with normal curve overlay."""
        n_cols = 2
        n_rows = (len(self.data.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, column in enumerate(self.data.columns):
            data = self.data[column].dropna()

            # Plot histogram
            sns.histplot(
                data=data,
                stat='density',
                ax=axes[i],
                alpha=0.6,
                label='Actual'
            )

            # Overlay normal distribution
            xmin, xmax = axes[i].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            mean = data.mean()
            std = data.std()
            p = stats.norm.pdf(x, mean, std)
            axes[i].plot(x, p, 'r-', lw=2, label='Normal')

            axes[i].set_title(f'{column} Returns Distribution')
            axes[i].legend()

        # Remove empty subplots
        for i in range(len(self.data.columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'return_distributions.png'))
        plt.close()

    def plot_qq_plots(self, save: bool = True) -> None:
        """Plot Q-Q plots for each cryptocurrency."""
        n_cols = 2
        n_rows = (len(self.data.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, column in enumerate(self.data.columns):
            data = self.data[column].dropna()
            stats.probplot(data, dist="norm", plot=axes[i])
            axes[i].set_title(f'Q-Q Plot: {column}')

        # Remove empty subplots
        for i in range(len(self.data.columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'qq_plots.png'))
        plt.close()

    def plot_lead_lag_relationships(self, save: bool = True) -> None:
        """Plot lead-lag relationships between cryptocurrencies."""
        if self.results is None or 'granger' not in self.results:
            logger.warning("No Granger causality results available")
            return

        granger_results = self.results['granger']
        significant_results = granger_results[granger_results['significant']]

        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()

        # Add edges for significant relationships
        for _, row in significant_results.iterrows():
            G.add_edge(
                row['cause'],
                row['effect'],
                weight=-np.log10(row['min_p_value'])
            )

        pos = nx.circular_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)

        # Draw edges with weights
        edges = G.edges(data=True)
        weights = [d['weight'] for (_, _, d) in edges]
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 2 for w in weights],
            edge_color='gray',
            arrowsize=20
        )

        plt.title('Lead-Lag Relationships')
        plt.axis('off')

        if save:
            plt.savefig(os.path.join(self.output_dir, 'lead_lag_relationships.png'))
        plt.close()

def main():
    """Main visualization pipeline."""
    # Example usage
    from src.data.processor import DataProcessor
    import pyarrow.parquet as pq
    import os
    from src.config import PROCESSED_DATA_PATH

    # Load data
    logger.info("Loading data...")

    all_data = {}
    for file in os.listdir(PROCESSED_DATA_PATH):
        if file.endswith('.parquet'):
            symbol = file.split('_')[0]
            file_path = os.path.join(PROCESSED_DATA_PATH, file)
            df = pq.read_table(file_path).to_pandas()
            all_data[symbol] = df['log_returns']

    data = pd.DataFrame(all_data)

    # Run causality analysis
    analyzer = CausalityAnalyzer(data)
    results = {
        'granger': analyzer.analyze_all_causality()['granger'],
        'causality_network': analyzer.create_causality_network()
    }

    # Create visualizations
    visualizer = ResultsVisualizer(data, results)
    visualizer.plot_all_results()

if __name__ == "__main__":
    main()
