# src/data/processor.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging
from ..config import PROCESSED_DATA_PATH
from ..utils.helpers import calculate_returns

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes raw cryptocurrency data and creates derived features."""
    
    def __init__(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """
        Initialize DataProcessor.
        
        Args:
            data: DataFrame or dict of DataFrames with raw crypto data
        """
        self.data = data if isinstance(data, dict) else {'default': data}
    
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Process all datasets with standard transformations."""
        processed_data = {}
        
        for symbol, df in self.data.items():
            processed_df = self._process_single_dataset(df)
            processed_data[symbol] = processed_df
            
        return processed_data
    
    def _process_single_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply processing steps to a single dataset."""
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Convert timestamps if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Calculate returns
        df['log_returns'] = calculate_returns(df['close'], method='log')
        df['simple_returns'] = calculate_returns(df['close'], method='simple')
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        # Volatility (rolling standard deviation of returns)
        df['volatility'] = df['log_returns'].rolling(window=30).std()
        
        # Moving averages
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Forward fill price data
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].fillna(method='ffill')
        
        # Fill remaining missing values with 0
        df = df.fillna(0)
        
        return df
    
    def save_processed_data(self, symbol: str) -> None:
        """Save processed data to parquet format."""
        if symbol not in self.data:
            raise KeyError(f"No data found for symbol {symbol}")
            
        processed_df = self.process_data()[symbol]
        
        # Create table and save to parquet
        table = pa.Table.from_pandas(processed_df)
        output_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed.parquet")
        pq.write_table(table, output_path)
        
        logger.info(f"Saved processed data for {symbol} to {output_path}")
