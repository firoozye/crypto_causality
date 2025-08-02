# Crypto Causality

1. The Goal is to determine causal relationships between BTC and other major crypto currencies: ETH, BNB, XRP, ADA,DOGE,SOL
2. Which causes which? Is this relationship stable? Is it persistent over shorter horizons?
3. Can we get by with 1 min data? Is causality more pronounced at higher frequencies? 



---- 
## The Project structure

crypto_causality/
├── data/
│   ├── raw/                    # Store raw CSV files
│   └── processed/              # Store processed parquet files
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py      # Data download functionality
│   │   └── processor.py       # Data processing and cleaning
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outliers.py        # Outlier detection
│   │   └── causality.py       # Granger causality analysis
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks/
│   └── analysis.ipynb         # Jupyter notebook for exploration
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md


## Revised Project Structure

crypto_analysis/
│
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outliers.py
│   │   ├── causality.py 
│   │   └── stationarity.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── processor.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── causality_viz.py 
│   │   └── plots.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── examples/
│   ├── run_analysis.py
│   └── visualize_results.py
├── notebooks/
│   └── crypto_exploration.ipynb
├── tests/
│   └── test_analysis.py
├── README.md
└── requirements.txt


----
## Installation.
1. Run pip install -r requirements.txt to install the required packages
2. Run  ../make_dir_structure.sh to create the project structure
3. config.py -> src/config.py
4. downloader.py -> src/data/downloader.py
5. python -m src.data.downloader to download 1m of major Crypto crosses vs Tether (USDT) from binance at 1min intervals.
6. The notebooks folder contains basic EDA for the data
7. The analysis folder contains the granger causality analysis
8. The visualization folder has various visualizations of the data
9. The examples folders shows how to use the code


---
# Project Goal

1. Determine causality for data using Granger, Multivariate Granger and/or Time-varying Granger Causality
2. Determine appropriate lags for the data and appropriate causal structure (which causes which?)
3. Can you find a frequency where causality is more pronounced? Can you find a frequency where causality is less pronounced?
4. Can you determine if the relationship is stable over time?
5. Can you determine if it is assymmetric?
6. Can you use this to predict future prices/returns, etc? 
7. Develop a trading model based on this information. If it is time-varying, you can use EWRLS or RLS that I provided before.
8. Please try to provide as many stats as you think good for this trading model.
9. Note that you are only testing a set of features. A real model will include other sets of features as well.`(e.g., ARMA/Momentum/Mean-reversion + Granger Causality + Sentiment + LOB data etc)
10. Please write up a Notebook and or a report on your findings. Please link to your code base. Please keep the heavy-lifting outside the Notebook. (make it more readable).
11. Note that this is meant to show findings, not explorations. You are meant to find some form of working model, no matter what the horizon.
12. If it isn't working for 1min, you can do 1sec, 1hour, 1day, etc. Binance has it all.


---
## Referencees
* [BTCUSDT 1 min data including OHLCV](https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1m/)
* klines should have OHLCV data etc [Binance Public data](https://github.com/binance/binance-public-data)


