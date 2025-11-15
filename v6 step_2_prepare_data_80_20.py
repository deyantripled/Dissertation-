# ==============================================================================
# Step 2 (Definitive 80/20 Version): Preparing the Data
# ==============================================================================
# This script loads the file created by Step 1 and performs the 80/20 split,
# creating the final .npy files for model training.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import sys

# --- 1. Load the LABELED Data (The file created by Step 1) ---
LABELED_DATA_FILE = 'labeled_data_85_percent.csv' # <-- THIS IS THE CRITICAL CHANGE
print(f"--- Loading LABELED data from '{LABELED_DATA_FILE}' ---")
try:
    df = pd.read_csv(LABELED_DATA_FILE, index_col='Datetime', parse_dates=True)
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{LABELED_DATA_FILE}' was not found. Please run Step 1 first.")
    sys.exit(1)
print("Labeled data loaded successfully.")

# --- 2. Define Features (X) and Target (y) ---
target_column = 'psychological_label'
feature_columns = [
    'Close', 'Volume', 'atr', 'return_2H', 'return_5H',
    'return_60H', 'volume_ratio', 'atr_ratio'
]

# --- 3. THE DEFINITIVE 80/20 Chronological Split ---
split_ratio = 0.80
split_index = int(len(df) * split_ratio)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
print(f"\n--- 80/20 Chronological Split ---")
print(f"Training data: {len(train_df)} rows")
print(f"Testing data:  {len(test_df)} rows")

# --- 4. Encode the Target Variable ---
label_encoder = LabelEncoder()
label_encoder.fit(df[target_column])
train_y_encoded = label_encoder.transform(train_df[target_column])
test_y_encoded = label_encoder.transform(test_df[target_column])
print("\nTarget labels encoded.")

# --- 5. Scale the Feature Data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[feature_columns])
train_X_scaled = scaler.transform(train_df[feature_columns])
test_X_scaled = scaler.transform(test_df[feature_columns])
print("Numerical features scaled successfully.")

# --- 6. Create Time-Series Sequences ---
def create_sequences(features, target, lookback_window):
    X, y = [], []
    for i in range(lookback_window, len(features)):
        X.append(features[i-lookback_window:i])
        y.append(target[i])
    return np.array(X), np.array(y)

lookback_window = 60
X_train, y_train = create_sequences(train_X_scaled, train_y_encoded, lookback_window)
X_test, y_test = create_sequences(test_X_scaled, test_y_encoded, lookback_window)
print(f"\nSequences created.")

# --- 7. Save the Prepared Data ---
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('label_encoder.npy', label_encoder.classes_)
np.save('scaler_params.npy', np.array([scaler.min_, scaler.scale_]))

print("\n--- Step 2 Complete ---")
print("All final data sets have been saved to .npy files with the correct names and formats.")