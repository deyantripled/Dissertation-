# ==============================================================================
# Final Training Script: AI Entry Optimizer
# ==============================================================================
# This definitive script trains ONE AI model to act as an intelligent filter for the
# mechanical EMA crossover buy signal, optimizing for low drawdown and high profit.
#
# 1. Loads the 85% historical data file.
# 2. Simulates all historical trades to label each 'buy_signal' as "Good" (1)
#    if it led to a profitable exit, or "Bad" (0) if it hit the stop-loss.
# 3. Trains an LSTM model on the patterns of 'fast_ma', 'slow_ma', and 'atr'.
# 4. Saves the final trained model and scaler for the backtesting script.
# ==============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

# --- 1. Configuration ---
DATA_FILE = '/home/tripled/backtest_data/IBKR_QQQ_10Y_1H_TEST_85_FROM_100_PERCENT.csv'
MODEL_SAVE_PATH = 'entry_optimizer_model.keras'
SCALER_SAVE_PATH = 'entry_scaler_params.npy'

# Parameters for labeling and training
LOOKAHEAD_PERIOD = 100  # How many hours to look into the future for an outcome
STOP_LOSS_ATR_MULT = 2.0
LOOKBACK_WINDOW = 60 # How many hours of indicator data the AI sees

# --- 2. Load Data and Calculate Base Indicators ---
print(f"--- Loading data from: {DATA_FILE} ---")
try:
    df = pd.read_csv(DATA_FILE, index_col='Datetime', parse_dates=True)
    df.dropna(inplace=True)
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{DATA_FILE}' was not found.")
    sys.exit(1)

print("--- Calculating base indicators (EMA, ATR, Signals)... ---")
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

# --- 3. Create the "Good Entry" Target Labels ---
print("--- Labeling historical buy signals as 'Good' (1) or 'Bad' (0)... ---")
df['good_entry'] = np.nan
buy_signal_indices = df[df['buy_signal']].index

for entry_time in buy_signal_indices:
    entry_price = df.loc[entry_time, 'Close']
    entry_atr = df.loc[entry_time, 'atr']
    stop_loss_price = entry_price - (STOP_LOSS_ATR_MULT * entry_atr)
    
    outcome_window_df = df.loc[entry_time:].iloc[1:LOOKAHEAD_PERIOD+1]
    if outcome_window_df.empty: continue

    hit_stop_loss = (outcome_window_df['Low'] <= stop_loss_price).any()
    hit_sell_signal = (outcome_window_df['sell_signal']).any()
    
    if hit_stop_loss and hit_sell_signal:
        stop_time = outcome_window_df[outcome_window_df['Low'] <= stop_loss_price].index[0]
        sell_time = outcome_window_df[outcome_window_df['sell_signal']].index[0]
        if sell_time < stop_time:
            df.loc[entry_time, 'good_entry'] = 1 # Profit via sell signal
        else:
            df.loc[entry_time, 'good_entry'] = 0 # Loss via stop-loss
    elif hit_sell_signal:
        df.loc[entry_time, 'good_entry'] = 1 # Profit via sell signal
    elif hit_stop_loss:
        df.loc[entry_time, 'good_entry'] = 0 # Loss via stop-loss
    else:
        df.loc[entry_time, 'good_entry'] = 0 # Trade timed out, treat as bad

# --- 4. Prepare Data for LSTM ---
print("--- Preparing data for AI training... ---")
signal_df = df[df['good_entry'].notna()].copy()
print(f"Found {len(signal_df)} labeled entry signals to train on.")
print("Distribution of labels:")
print(signal_df['good_entry'].value_counts(normalize=True))

feature_columns = ['fast_ma', 'slow_ma', 'atr']

def create_sequences_from_indices(full_df, signal_indices, features, target_series, lookback):
    X_seq, y_seq = [], []
    for idx in signal_indices:
        loc = full_df.index.get_loc(idx)
        if loc < lookback: continue
        X_seq.append(full_df.iloc[loc-lookback:loc][features].values)
        y_seq.append(target_series.loc[idx])
    return np.array(X_seq), np.array(y_seq)

X_data, y_data = create_sequences_from_indices(df, signal_df.index, feature_columns, signal_df['good_entry'], LOOKBACK_WINDOW)

# Split into training and validation sets for model training
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, shuffle=False, random_state=42)

# Scale the features
scaler = MinMaxScaler()
# Reshape for scaling: (num_samples * timesteps, features)
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
scaler.fit(X_train_flat)

# Transform and reshape back
X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_flat = X_val.reshape(-1, X_val.shape[-1])
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

print(f"Created sequences. X_train shape: {X_train_scaled.shape}, X_val shape: {X_val_scaled.shape}")

# --- 5. Build and Train the LSTM ---
print("--- Building and training the Entry Optimizer LSTM... ---")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK_WINDOW, len(feature_columns))), Dropout(0.2),
    LSTM(30), Dropout(0.2),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Calculated class weights: {class_weight_dict}")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict
)

# --- 6. Save the Final Model and Scaler ---
print(f"\n--- Training complete. Saving final assets. ---")
np.save(SCALER_SAVE_PATH, np.array([scaler.min_, scaler.scale_]))
print(f"Final model saved to: {MODEL_SAVE_PATH}")
print(f"Scaler parameters saved to: {SCALER_SAVE_PATH}")
print("\n--- Process Complete ---")