
# API configuration
# src/config.py

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
# Data directories
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Trading pairs configuration
SYMBOLS = ['BTCUSDT',
           'ETHUSDT',
           'BNBUSDT',
           'XRPUSDT',
           'ADAUSDT',
           'DOGEUSDT'
           ]

# Time parameters
INTERVAL = '1m'  # Data granularity
START_DATE = '2024-01-01'
END_DATE = '2024-02-01'

# API configuration
BASE_URL = 'https://api.binance.com'

# Analysis parameters
DEFAULT_WINDOW = 30  # Default window size for rolling calculations
MAX_LAGS = 10  # Maximum lags for causality analysis
SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 8)
