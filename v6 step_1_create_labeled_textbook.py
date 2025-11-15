# ==============================================================================
# Step 1 (Definitive Version): Create the Labeled Textbook
# ==============================================================================
# This script loads the raw 85% data and calculates all features and labels,
# saving the result to a new file for the next step.
# ==============================================================================

import pandas as pd
import numpy as np
import sys

# --- Configuration ---
RAW_DATA_FILE = '/home/tripled/backtest_data/IBKR_QQQ_10Y_1H_TEST.csv'
LABELED_OUTPUT_FILE = 'labeled_data_85_percent.csv'

# --- Load Raw Data ---
print(f"--- Loading raw data from: {RAW_DATA_FILE} ---")
try:
    df = pd.read_csv(RAW_DATA_FILE, index_col='Datetime', parse_dates=True)
    df.dropna(inplace=True)
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{RAW_DATA_FILE}' was not found.")
    sys.exit(1)
print("Raw data loaded successfully.")

# --- Calculate All Features and Labels ---
print("--- Calculating all features and labels... ---")
# ... (This is the full labeling logic from your original script)
fast_ma_period, slow_ma_period = 20, 50
df['fast_ma'] = df['Close'].rolling(window=fast_ma_period).mean()
df['slow_ma'] = df['Close'].rolling(window=slow_ma_period).mean()
lookback_window = 60
df['tr1'] = df['High'] - df['Low']
df['tr2'] = np.abs(df['High'] - df['Close'].shift(1))
df['tr3'] = np.abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['true_range'].rolling(window=14).mean()
df['return_2H'] = df['Close'].pct_change(periods=2)
df['return_5H'] = df['Close'].pct_change(periods=5)
df['return_60H'] = df['Close'].pct_change(periods=lookback_window)
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=lookback_window).mean()
df['atr_ratio'] = df['atr'] / df['atr'].rolling(window=lookback_window).mean()
df.dropna(inplace=True)
fomo_return_threshold = df['return_5H'].quantile(0.99)
correction_return_threshold = df['return_2H'].quantile(0.05)
capitulation_return_threshold = df['return_5H'].quantile(0.01)
df['psychological_label'] = 'Neutral'
fomo_condition = ((df['return_5H'] > fomo_return_threshold) & (df['return_60H'] > df['return_60H'].quantile(0.90)) & (df['volume_ratio'] > 1.3))
df.loc[fomo_condition, 'psychological_label'] = 'FOMO'
capitulation_condition = ((df['return_5H'] < capitulation_return_threshold) & (df['volume_ratio'] > 2.5) & (df['atr_ratio'] > 2.0))
downtrend_regime_condition = ((df['fast_ma'] < df['slow_ma'] * 0.995) & (df['atr_ratio'] > 1.2))
df.loc[capitulation_condition | downtrend_regime_condition, 'psychological_label'] = 'Panic'
base_correction_condition = ((df['return_2H'] < correction_return_threshold) & (df['atr_ratio'] > 1.5))
is_new_event_condition = ((df['psychological_label'].shift(1) != 'Correction') & (df['psychological_label'].shift(1) != 'Panic'))
df.loc[base_correction_condition & is_new_event_condition, 'psychological_label'] = 'Correction'
herd_condition = ((df['Close'] > df['slow_ma']) & (df['atr_ratio'] < 0.8) & (df['psychological_label'] == 'Neutral'))
df.loc[herd_condition, 'psychological_label'] = 'Herd'
print("Feature and label calculation complete.")

# --- Save the Labeled File ---
df.to_csv(LABELED_OUTPUT_FILE)
print(f"\n--- Step 1 Complete ---")
print(f"The labeled textbook has been saved to: {LABELED_OUTPUT_FILE}")