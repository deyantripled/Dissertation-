# ==============================================================================
# DEFINITIVE Backtest Script for Strategy A (with DIAGNOSTICS)
# ==============================================================================
# This version includes a diagnostic print statement to show the AI's
# confidence for every single buy signal it evaluates.
# ==============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sys
import quantstats as qs

# --- 1. Configuration ---
OUT_OF_SAMPLE_DATA_FILE = '/home/tripled/backtest_data/OUT OF SAMPLE TEST 15 PERCENT.csv'
ENTRY_MODEL_PATH = 'entry_optimizer_model.keras'
ENTRY_SCALER_PATH = 'entry_scaler_params.npy'
REPORT_FILENAME = 'Report_LSTM-EMA-ATR.html'

# Backtest Parameters
LOOKBACK_WINDOW = 60
STOP_LOSS_ATR_MULT = 2.0
INITIAL_CAPITAL = 100000.0
TRADE_FRACTION = 1.0
COMMISSION_RATE = 0.0001
RISK_FREE_RATE = 0.02

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
fast_ma_period, slow_ma_period = 20, 50
df['fast_ma'] = df['Close'].rolling(window=fast_ma_period).mean()
df['slow_ma'] = df['Close'].rolling(window=slow_ma_period).mean()
df['buy_signal'] = (df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
df['sell_signal'] = (df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
df['tr1'] = df['High'] - df['Low']
df['tr2'] = np.abs(df['High'] - df['Close'].shift(1))
df['tr3'] = np.abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['true_range'].rolling(window=14).mean()
df.dropna(inplace=True)

# --- 4. Main Backtesting Loop ---
print("\n--- Starting Backtest Simulation with Diagnostics ---")
in_position = False
current_equity = INITIAL_CAPITAL
equity_history = [(df.index[0], INITIAL_CAPITAL)]

entry_price, stop_loss_price = 0, 0
trade_size_in_dollars = 0
entry_feature_columns = ['fast_ma', 'slow_ma', 'atr']

for i in range(LOOKBACK_WINDOW, len(df)):
    current_time = df.index[i]
    current_price = df['Close'].iloc[i]

    if in_position:
        exit_reason = None
        exit_price = 0
        if df['Low'].iloc[i] <= stop_loss_price:
            exit_reason = 'Stop-Loss'
            exit_price = stop_loss_price
        elif df['sell_signal'].iloc[i]:
            exit_reason = 'Sell Signal'
            exit_price = current_price
            
        if exit_reason:
            num_shares = trade_size_in_dollars / entry_price
            gross_pnl = (exit_price - entry_price) * num_shares
            entry_commission = trade_size_in_dollars * COMMISSION_RATE
            exit_value = exit_price * num_shares
            exit_commission = exit_value * COMMISSION_RATE
            total_commission = entry_commission + exit_commission
            net_pnl = gross_pnl - total_commission
            current_equity += net_pnl
            equity_history.append((current_time, current_equity))
            in_position = False
            trade_size_in_dollars = 0

    if not in_position and df['buy_signal'].iloc[i]:
        if current_equity <= 0: continue

        sequence = df.iloc[i-LOOKBACK_WINDOW:i][entry_feature_columns].values
        scaled_sequence = entry_scaler.transform(sequence).reshape(1, LOOKBACK_WINDOW, len(entry_feature_columns))
        entry_prediction = entry_model.predict(scaled_sequence, verbose=0)[0][0]
        
        # <<< --- THIS IS THE CRITICAL DIAGNOSTIC LINE --- >>>
        print(f"[{current_time}] Buy Signal Detected. AI Confidence: {entry_prediction:.2%}")

        if entry_prediction > ENTRY_CONFIDENCE_THRESHOLD:
            in_position = True
            entry_price = current_price
            trade_size_in_dollars = current_equity * TRADE_FRACTION
            current_atr = df['atr'].iloc[i]
            stop_loss_price = entry_price - (current_atr * STOP_LOSS_ATR_MULT)
            print(f"    ---> Trade APPROVED and entered at {current_price:.2f}")

# --- 5. Generate Report ---
print("\n--- Backtest Complete. Preparing data for QuantStats report... ---")

if len(equity_history) <= 1:
    print("No trades were executed in this backtest.")
    sys.exit()

equity_df = pd.DataFrame(equity_history, columns=['Date', 'Equity']).set_index('Date')
equity_series = equity_df['Equity']
daily_equity = equity_series.resample('D').last().ffill()
daily_returns = daily_equity.pct_change().fillna(0)
daily_returns.name = "Strategy A (Compounding)"

qs.extend_pandas()
benchmark_prices = df['Close'].resample('D').last().ffill()
benchmark_returns = benchmark_prices.pct_change().fillna(0)
benchmark_returns.name = "Benchmark (Buy & Hold)"

print(f"--- Generating QuantStats report... ---")
try:
    qs.reports.html(
        returns=daily_returns,
        benchmark=benchmark_returns,
        rf=RISK_FREE_RATE,
        output=REPORT_FILENAME,
        title=f'Strategy A (Compounding at {TRADE_FRACTION*100:.0f}% Equity per Trade)',
        download_filename=REPORT_FILENAME
    )
    print(f"Successfully generated and saved QuantStats report to: {REPORT_FILENAME}")
except Exception as e:
    print(f"Error generating QuantStats report: {e}")