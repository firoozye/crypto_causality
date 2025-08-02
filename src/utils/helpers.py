# src/utils/helpers.py

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def calculate_returns(
    prices: pd.Series,
    method: str = 'log'
) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Series of prices
        method: Return calculation method ('log' or 'simple')
        
    Returns:
        Series of returns
    """
    if method == 'log':
        return np.log(prices).diff()
    elif method == 'simple':
        return prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

def standardize_series(
    series: pd.Series,
    method: str = 'zscore'
) -> pd.Series:
    """
    Standardize a time series.
    
    Args:
        series: Input series
        method: Standardization method ('zscore' or 'minmax')
        
    Returns:
        Standardized series
    """
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

def rolling_statistics(
    series: pd.Series,
    window: int
) -> pd.DataFrame:
    """
    Calculate rolling statistics for a series.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        DataFrame with rolling statistics
    """
    stats = pd.DataFrame(index=series.index)
    
    stats['mean'] = series.rolling(window=window).mean()
    stats['std'] = series.rolling(window=window).std()
    stats['skew'] = series.rolling(window=window).skew()
    stats['kurt'] = series.rolling(window=window).kurt()
    
    return stats

def resample_data(
    data: pd.DataFrame,
    freq: str,
    agg_dict: Optional[dict] = None
) -> pd.DataFrame:
    """
    Resample time series data.
    
    Args:
        data: Input DataFrame
        freq: Resampling frequency (e.g., '1H', '1D')
        agg_dict: Dictionary of aggregation functions
        
    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    return data.resample(freq).agg(agg_dict)

def align_series(
    series_list: List[pd.Series],
    method: str = 'inner'
) -> List[pd.Series]:
    """
    Align multiple time series to common timestamps.
    
    Args:
        series_list: List of time series
        method: Join method ('inner' or 'outer')
        
    Returns:
        List of aligned series
    """
    df = pd.concat(series_list, axis=1)
    if method == 'inner':
        df = df.dropna()
    return [df[col] for col in df.columns]

def time_window_split(
    data: pd.DataFrame,
    train_size: Union[float, int],
    test_size: Optional[Union[float, int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split time series data into train/validation/test sets.
    
    Args:
        data: Input DataFrame
        train_size: Size of training set
        test_size: Size of test set
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    if isinstance(train_size, float):
        train_size = int(len(data) * train_size)
    
    if test_size is None:
        train = data.iloc[:train_size]
        val = data.iloc[train_size:]
        return train, val, None
    
    if isinstance(test_size, float):
        test_size = int(len(data) * test_size)
    
    train = data.iloc[:train_size]
    val = data.iloc[train_size:-test_size]
    test = data.iloc[-test_size:]
    
    return train, val, test
