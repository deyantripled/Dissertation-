# ====================================================================================
# DEFINITIVE Backtest Script for Strategy A with Fixed Trade Size
# ====================================================================================
# This script correctly simulates Strategy A using a non-compounding, fixed
# position size for every trade.
#
# ENGINE: Mark-to-Market (M2M) for accurate daily equity tracking.
# POSITION SIZING: Fixed $100,000 per trade.
# ====================================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import quantstats as qs

# --- 1. Configuration ---
# Data and Model Paths
OUT_OF_SAMPLE_DATA_FILE = '/home/tripled/backtest_data/OUT OF SAMPLE TEST 15 PERCENT.csv'
ENTRY_MODEL_PATH = 'entry_optimizer_model.keras'
ENTRY_SCALER_PATH = 'entry_scaler_params.npy'
REPORT_FILENAME = 'report_Strategy_A_Fixed_100k_Trade.html'

# Backtest Parameters
LOOKBACK_WINDOW = 60
STOP_LOSS_ATR_MULT = 2.0
INITIAL_CAPITAL = 100000.0
# --- POSITION SIZING SET TO A FIXED $100,000 ---
FIXED_TRADE_SIZE_USD = 100000.0
COMMISSION_RATE = 0.0001 # 0.01% commission per transaction (entry and exit)
RISK_FREE_RATE = 0.02    # 2% annual risk-free rate for reporting

# AI Prediction Thresholds for STRATEGY A
ENTRY_CONFIDENCE_THRESHOLD = 0.42

# --- 2. Load Models and Scalers ---
print("--- Loading AI Models and Scalers ---")
try:
    entry_model = tf.keras.models.load_model(ENTRY_MODEL_PATH)
    entry_scaler_params = np.load(ENTRY_SCALER_PATH)
    entry_scaler = MinMaxScaler()
    entry_scaler.min_, entry_scaler.scale_ = entry_scaler_params[0], entry_scaler_params[1]
    print("Models and scalers loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load a required model or scaler file. Details: {e}")
    sys.exit(1)

# --- 3. Load and Prepare Data ---
print(f"\n--- Loading and preparing data from: {OUT_OF_SAMPLE_DATA_FILE} ---")
try:
    df = pd.read_csv(OUT_OF_SAMPLE_DATA_FILE, index_col='Datetime', parse_dates=True)
except FileNotFoundError:
    print(f"FATAL ERROR: The data file '{OUT_OF_SAMPLE_DATA_FILE}' was not found.")
    sys.exit(1)

print("--- Calculating base indicators... ---")
df['fast_ma'] = df['Close'].rolling(window=20).mean()
df['slow_ma'] = df['Close'].rolling(window=50).mean()
df['buy_signal'] = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
df['sell_signal'] = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
df['atr'] = (df['High'] - df['Low']).rolling(window=14).mean() # Simplified ATR
df.dropna(inplace=True)
entry_feature_columns = ['fast_ma', 'slow_ma', 'atr']

# --- 4. Mark-to-Market Backtesting Engine ---
print("\n--- Starting Backtest with M2M Engine and Fixed Trade Size ---")

# Portfolio state variables
cash = INITIAL_CAPITAL
position_size = 0.0  # In number of shares
entry_price = 0.0
stop_loss_price = 0.0

# Series to store the daily portfolio value
equity_curve = pd.Series(index=df.index, dtype=float)
equity_curve.iloc[:LOOKBACK_WINDOW] = INITIAL_CAPITAL

for i in range(LOOKBACK_WINDOW, len(df)):
    current_price = df['Close'].iloc[i]
    portfolio_value = equity_curve.iloc[i-1]

    # --- A. Check for Exit Conditions ---
    if position_size > 0:
        exit_triggered = False
        exit_price = 0

        if df['Low'].iloc[i] <= stop_loss_price:
            exit_triggered = True
            exit_price = stop_loss_price
        elif df['sell_signal'].iloc[i]:
            exit_triggered = True
            exit_price = current_price

        if exit_triggered:
            trade_value = position_size * exit_price
            commission = trade_value * COMMISSION_RATE
            cash += trade_value - commission
            position_size = 0.0
            entry_price = 0.0

    # --- B. Check for Entry Conditions ---
    if position_size == 0 and df['buy_signal'].iloc[i]:
        # Check if portfolio value can cover the fixed trade size
        if portfolio_value >= FIXED_TRADE_SIZE_USD:
            sequence = df.iloc[i-LOOKBACK_WINDOW:i][entry_feature_columns].values
            scaled_sequence = entry_scaler.transform(sequence).reshape(1, LOOKBACK_WINDOW, len(entry_feature_columns))
            entry_prediction = entry_model.predict(scaled_sequence, verbose=0)[0][0]

            if entry_prediction > ENTRY_CONFIDENCE_THRESHOLD:
                # Use the fixed dollar amount for the trade
                capital_to_invest = FIXED_TRADE_SIZE_USD
                commission = capital_to_invest * COMMISSION_RATE
                
                position_size = (capital_to_invest - commission) / current_price
                cash -= capital_to_invest
                
                entry_price = current_price
                stop_loss_price = entry_price - (df['atr'].iloc[i] * STOP_LOSS_ATR_MULT)

    # --- C. Record Daily Portfolio Value (Mark-to-Market) ---
    equity_curve.iloc[i] = cash + (position_size * current_price)

# --- 5. Generate QuantStats Report from True Equity Curve ---
print("\n--- Backtest Complete. Generating QuantStats report... ---")

daily_returns = equity_curve.pct_change().fillna(0)
daily_returns.name = "Strategy A (Fixed $100k Trade)"

qs.extend_pandas()
benchmark_returns = df['Close'].pct_change().fillna(0)
benchmark_returns.name = "Benchmark (Buy & Hold)"

try:
    qs.reports.html(
        returns=daily_returns,
        benchmark=benchmark_returns,
        rf=RISK_FREE_RATE,
        output=REPORT_FILENAME,
        title='Strategy A (Fixed $100k Trade Size vs. Buy & Hold)',
        download_filename=REPORT_FILENAME
    )
    print(f"Successfully generated and saved QuantStats report to: {REPORT_FILENAME}")
except Exception as e:
    print(f"Error generating QuantStats report: {e}")