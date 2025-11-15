# ==============================================================================
#                      COMPLETE BACKTESTING & SIMULATION SCRIPT
# ==============================================================================
# This single file loads the AI model, runs a fast 10,000-trial Monte Carlo
# simulation, and generates a comprehensive 4-panel analysis dashboard.
# ==============================================================================

# --- REQUIRED IMPORTS ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# --- 1. Configuration and Parameters ---
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.0001
lookback_window = 60

# --- File Paths ---
MODEL_FILE = 'final_psychology_model.keras'
SCALER_FILE = 'scaler_params.npy'
ENCODER_FILE = 'label_encoder.npy'
TEST_DATA_FILE = '/home/tripled/backtest_data/OUT OF SAMPLE TEST 15 PERCENT.csv'

# --- 2. Load Core AI Assets and Test Data ---
print("--- Loading all required assets for final backtesting ---")
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    scaler_params = np.load(SCALER_FILE, allow_pickle=True)
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = scaler_params[0], scaler_params[1]
    label_classes = np.load(ENCODER_FILE, allow_pickle=True)
    test_df = pd.read_csv(TEST_DATA_FILE, index_col='Datetime', parse_dates=True)
except (FileNotFoundError, IOError) as e:
    print(f"FATAL ERROR: Could not load necessary files. {e}")
    sys.exit(1)

# --- 3. Calculate All Features from Scratch ---
print("\n--- Calculating all indicators on the test set from scratch... ---")
test_df['tr1'] = test_df['High'] - test_df['Low']
test_df['tr2'] = np.abs(test_df['High'] - test_df['Close'].shift(1))
test_df['tr3'] = np.abs(test_df['Low'] - test_df['Close'].shift(1))
test_df['true_range'] = test_df[['tr1', 'tr2', 'tr3']].max(axis=1)
test_df['atr'] = test_df['true_range'].rolling(window=14).mean()
test_df['return_2H'] = test_df['Close'].pct_change(periods=2)
test_df['return_5H'] = test_df['Close'].pct_change(periods=5)
test_df['return_60H'] = test_df['Close'].pct_change(periods=lookback_window)
test_df['volume_ratio'] = test_df['Volume'] / test_df['Volume'].rolling(window=lookback_window).mean()
test_df['atr_ratio'] = test_df['atr'] / test_df['atr'].rolling(window=lookback_window).mean()
test_df.dropna(inplace=True)
feature_columns = ['Close', 'Volume', 'atr', 'return_2H', 'return_5H', 'return_60H', 'volume_ratio', 'atr_ratio']
print(f"Final, untouched test data prepared: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")

# ==============================================================================
# --- 8. FASTER True Monte Carlo Simulation (Vectorized & Batched) ---
# ==============================================================================
print("\n--- Running FASTER True (Path-Generating) Monte Carlo Simulation ---")

def create_batched_windows(data, window_size):
    shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

log_returns = np.log(test_df['Close'] / test_df['Close'].shift(1))
mu = log_returns.mean()
sigma = log_returns.std()
initial_price = test_df['Close'].iloc[0]
simulation_days = len(test_df)
# =================================================
n_simulations = 10000  # <--- THE ONLY CHANGE
# =================================================

equity_curves = []
final_equities_gbm = []
print(f"Running {n_simulations} full backtest simulations (optimized)...")
print("NOTE: This will take a significant amount of time. Please be patient.")

for i in tqdm(range(n_simulations)):
    synthetic_prices = [initial_price]
    for _ in range(1, simulation_days):
        dt = 1
        random_shock = np.random.normal(0, 1)
        price_t = synthetic_prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shock)
        synthetic_prices.append(price_t)
    synthetic_df = pd.DataFrame(synthetic_prices, index=test_df.index, columns=['Close'])
    price_volatility = synthetic_df['Close'].pct_change().std() * 0.5
    synthetic_df['High'] = synthetic_df['Close'] * (1 + np.random.uniform(0, price_volatility, size=len(synthetic_df)))
    synthetic_df['Low'] = synthetic_df['Close'] * (1 - np.random.uniform(0, price_volatility, size=len(synthetic_df)))
    synthetic_df['Volume'] = np.random.choice(test_df['Volume'], size=len(synthetic_df), replace=True)
    synthetic_df['tr1'] = synthetic_df['High'] - synthetic_df['Low']
    synthetic_df['tr2'] = np.abs(synthetic_df['High'] - synthetic_df['Close'].shift(1))
    synthetic_df['tr3'] = np.abs(synthetic_df['Low'] - synthetic_df['Close'].shift(1))
    synthetic_df['true_range'] = synthetic_df[['tr1', 'tr2', 'tr3']].max(axis=1)
    synthetic_df['atr'] = synthetic_df['true_range'].rolling(window=14).mean()
    synthetic_df['return_2H'] = synthetic_df['Close'].pct_change(periods=2)
    synthetic_df['return_5H'] = synthetic_df['Close'].pct_change(periods=5)
    synthetic_df['return_60H'] = synthetic_df['Close'].pct_change(periods=lookback_window)
    synthetic_df['volume_ratio'] = synthetic_df['Volume'] / synthetic_df['Volume'].rolling(window=lookback_window).mean()
    synthetic_df['atr_ratio'] = synthetic_df['atr'] / synthetic_df['atr'].rolling(window=lookback_window).mean()
    synthetic_df.dropna(inplace=True)
    if len(synthetic_df) < lookback_window: continue
    feature_data = synthetic_df[feature_columns].values
    batched_windows = create_batched_windows(feature_data, lookback_window)
    n_samples, window_len, n_features = batched_windows.shape
    reshaped_for_scaling = batched_windows.reshape(n_samples * window_len, n_features)
    reshaped_for_scaling[~np.isfinite(reshaped_for_scaling)] = 0
    scaled_features = scaler.transform(reshaped_for_scaling)
    scaled_windows = scaled_features.reshape(n_samples, window_len, n_features)
    prediction_probs = model.predict(scaled_windows, verbose=0, batch_size=2048) # Increased batch size for performance
    predicted_indices = np.argmax(prediction_probs, axis=1)
    all_ai_states = label_classes[predicted_indices]
    position, entry_price, stop_loss_price = 0, 0, 0
    champion_equity_sim = [INITIAL_CAPITAL] * lookback_window
    ai_state_history = []
    persistence_filter = 3
    for j in range(len(all_ai_states)):
        sim_idx = j + lookback_window - 1
        current_price = synthetic_df['Close'].iloc[sim_idx]
        previous_price = synthetic_df['Close'].iloc[sim_idx - 1]
        current_capital = champion_equity_sim[-1]
        if position == 1: current_capital *= (current_price / previous_price)
        ai_state = all_ai_states[j]
        if position == 1 and (current_price <= stop_loss_price or ai_state in ['Panic', 'Correction']):
            exit_price = stop_loss_price if current_price <= stop_loss_price else current_price
            current_capital = champion_equity_sim[-1] * (exit_price / previous_price)
            current_capital *= (1 - COMMISSION_RATE)
            position = 0
        if position == 0:
            ai_state_history.append(ai_state)
            if len(ai_state_history) > persistence_filter: ai_state_history.pop(0)
            is_stable_herd = (len(ai_state_history) == persistence_filter and all(s == 'Herd' for s in ai_state_history))
            if is_stable_herd:
                position = 1
                entry_price = current_price
                current_capital *= (1 - COMMISSION_RATE)
                stop_loss_price = entry_price - (2 * synthetic_df['atr'].iloc[sim_idx])
        if position == 1 and ai_state in ['Herd', 'FOMO']:
            new_trailing_stop = current_price - (1 * synthetic_df['atr'].iloc[sim_idx])
            stop_loss_price = max(stop_loss_price, new_trailing_stop)
        champion_equity_sim.append(current_capital)
    final_equities_gbm.append(champion_equity_sim[-1])
    equity_curves.append(champion_equity_sim)

# ==============================================================================
# --- 9. Comprehensive Analysis Suite with Risk-Reward Profiling ---
# ==============================================================================
print("\nGenerating comprehensive analysis dashboard with Risk-Reward plot...")

final_returns = (np.array(final_equities_gbm) / INITIAL_CAPITAL - 1) * 100
mean_final_equity = np.mean(final_equities_gbm)
median_final_equity = np.median(final_equities_gbm)
percentile_5 = np.percentile(final_equities_gbm, 5)
percentile_95 = np.percentile(final_equities_gbm, 95)
probability_of_profit = (np.array(final_equities_gbm) > INITIAL_CAPITAL).mean() * 100
max_drawdowns = []
for curve in equity_curves:
    equity_series = pd.Series(curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdowns.append(drawdown.min())
max_drawdowns_pct = np.array(max_drawdowns) * 100

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(20, 16), gridspec_kw={'height_ratios': [2, 2]})
fig.suptitle('Comprehensive Monte Carlo Analysis (10000 Simulations)', fontsize=24)

ax1 = axes[0, 0]
ax1.hist(final_equities_gbm, bins=100, color='royalblue', alpha=0.8, edgecolor='white', linewidth=0.5) # More bins for smoother plot
ax1.axvline(INITIAL_CAPITAL, color='yellow', linestyle='-', linewidth=2, label=f'Initial Capital: ${INITIAL_CAPITAL:,.2f}')
ax1.axvline(mean_final_equity, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_final_equity:,.2f}')
ax1.axvline(median_final_equity, color='lime', linestyle=':', linewidth=3, label=f'Median: ${median_final_equity:,.2f}')
ax1.axvline(percentile_5, color='orange', linestyle='--', linewidth=2, label=f'5th Percentile: ${percentile_5:,.2f}')
ax1.axvline(percentile_95, color='cyan', linestyle='--', linewidth=2, label=f'95th Percentile: ${percentile_95:,.2f}')
ax1.set_title('Distribution of Final Equity', fontsize=16)
ax1.set_xlabel('Final Portfolio Value ($)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.2)

ax2 = axes[0, 1]
for i, curve in enumerate(equity_curves[:200]): # Show more sample paths
    ax2.plot(curve, color='cyan', alpha=0.1)
ax2.plot(equity_curves[0], color='cyan', alpha=0.5, label='Sample Paths')
ax2.axhline(INITIAL_CAPITAL, color='yellow', linestyle='-', linewidth=2, label='Initial Capital')
ax2.set_title('Sample of Simulated Equity Paths', fontsize=16)
ax2.set_xlabel('Trading Periods', fontsize=12)
ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.2)

ax3 = axes[1, 0]
colors = final_returns
scatter = ax3.scatter(max_drawdowns_pct, final_returns, c=colors, cmap='viridis', alpha=0.5, s=15) # smaller points
cbar = fig.colorbar(scatter, ax=ax3)
cbar.set_label('Final Return (%)', fontsize=10)
ax3.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
ax3.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
mean_dd = np.mean(max_drawdowns_pct)
mean_ret = np.mean(final_returns)
ax3.scatter(mean_dd, mean_ret, color='red', s=150, edgecolor='white', zorder=5, label=f'Mean Outcome\n(DD: {mean_dd:.2f}%, Ret: {mean_ret:.2f}%)')
ax3.set_title('Risk vs. Reward Profile', fontsize=16)
ax3.set_xlabel('Maximum Drawdown During Simulation (%)', fontsize=12)
ax3.set_ylabel('Final Return (%)', fontsize=12)
ax3.legend(loc='lower left')
ax3.grid(True, linestyle='--', alpha=0.2)

ax4 = axes[1, 1]
ax4.axis('off')
stats_text = (
    f"--- Key Performance Indicators ---\n\n"
    f"Probability of Profit: {probability_of_profit:.2f}%\n\n"
    f"Mean Final Equity: ${mean_final_equity:,.2f}\n"
    f"Mean Return: {mean_ret:.2f}%\n\n"
    f"Median Final Equity: ${median_final_equity:,.2f}\n"
    f"Median Return: {(median_final_equity/INITIAL_CAPITAL - 1):.2%}\n\n"
    f"5th Percentile Equity: ${percentile_5:,.2f}\n"
    f"95th Percentile Equity: ${percentile_95:,.2f}\n\n"
    f"Mean Maximum Drawdown: {mean_dd:.2f}%\n"
    f"Median Maximum Drawdown: {np.median(max_drawdowns_pct):.2f}%"
)
ax4.text(0.5, 0.5, stats_text, fontsize=14, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
print("Displaying comprehensive analysis dashboard...")
plt.show()