# %% [markdown]
# # Cryptocurrency Forecasting Model using EWRLS
#
# ## Goal:
# * Develop a forecasting model for cryptocurrency returns using Exponentially Weighted Recursive Least Squares (EWRLS).
# * Leverage insights from Granger causality analysis to select relevant features.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score

# Add the project root to the python path
try:
    # This works when the script is run directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    # This works when run in an interactive environment like Jupyter
    # Assumes the notebook is in notebooks/python or notebooks/jupyter
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ewrls.ewrls import EWRLSRidge

# Set style for better visualizations
sns.set_theme()
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 6]

# %% [markdown]
# ## Data Loading and Preparation
# We will load the processed cryptocurrency data and prepare it for forecasting.

# %%
def load_all_crypto_data(data_dir=os.path.join(project_root, 'data', 'processed')):
    all_data = {}
    for file in glob.glob(os.path.join(data_dir, "*.parquet")):
        symbol = os.path.basename(file).split('_')[0]
        print(f"Loading {file}...")
        df = pq.read_table(file).to_pandas()
        all_data[symbol] = df
    return all_data

# Load the data
crypto_data = load_all_crypto_data()

# Calculate returns for each crypto
returns_data = {}
for symbol, df in crypto_data.items():
    returns = pd.DataFrame()
    returns['timestamp'] = df['timestamp']
    returns['returns'] = np.log(df['close'].astype(float)).diff()
    returns_data[symbol] = returns

# Create a combined returns dataframe
combined_returns = pd.DataFrame()
for symbol, returns in returns_data.items():
    combined_returns[symbol] = returns['returns']
combined_returns.index = list(returns_data.values())[0]['timestamp']
combined_returns = combined_returns.dropna()

# %% [markdown]
# ## Feature Engineering
# Create lagged features for the forecasting model.

# %%
def create_lagged_features(df, target_col, lags=5):
    df_lagged = df.copy()
    for col in df.columns:
        for i in range(1, lags + 1):
            df_lagged[f'{col}_lag_{i}'] = df_lagged[col].shift(i)
    
    # Define target variable (next period's return)
    df_lagged['target'] = df_lagged[target_col].shift(-1)
    
    return df_lagged.dropna()

# Define target cryptocurrency
target_crypto = 'BTCUSDT'

# Create lagged features for all cryptocurrencies
forecasting_data = create_lagged_features(combined_returns, target_crypto, lags=5)

# Separate features (X) and target (y)
X = forecasting_data.drop(columns=['target'])
y = forecasting_data['target']

# %% [markdown]
# ## EWRLS Forecasting Model
# Implement a rolling window forecasting strategy using EWRLSRidge.

# %%
def run_ewrls_forecasting(X, y, window_size=1000, span=100, regularization=0.1):
    predictions = []
    actuals = []

    for i in range(window_size, len(X)):
        X_train = X.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]
        X_test = X.iloc[i:i+1]
        y_test = y.iloc[i:i+1]

        # Initialize and update EWRLS model
        model = EWRLSRidge(num_features=X_train.shape[1], span=span, regularization=regularization)
        model.update(y_train.values, X_train.values)

        # Make prediction
        prediction = model.generate_prediction(X_test.values[0])
        predictions.append(prediction)
        actuals.append(y_test.values[0])

    return pd.DataFrame({'Actual': actuals, 'Prediction': predictions}, index=y.index[window_size:len(X)])

# Run forecasting
forecast_results = run_ewrls_forecasting(X, y)

# %% [markdown]
# ## Evaluation
# Evaluate the performance of the forecasting model.

# %%
mse = mean_squared_error(forecast_results['Actual'], forecast_results['Prediction'])
r2 = r2_score(forecast_results['Actual'], forecast_results['Prediction'])

print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R-squared (R2): {r2:.6f}")

# %% [markdown]
# ## Plotting Forecasts
# Visualize the actual vs. predicted returns.

# %%
plt.figure(figsize=(15, 7))
plt.plot(forecast_results['Actual'], label='Actual Returns')
plt.plot(forecast_results['Prediction'], label='Predicted Returns', alpha=0.7)
plt.title(f'{target_crypto} Returns: Actual vs. Predicted')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.tight_layout()
plt.show()
