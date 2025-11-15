# ==============================================================================
# Step 2.2 (Final Version): Training with Tuned Class Weights
# ==============================================================================
# This final training version uses manually tuned class weights to create a more
# balanced and effective model. The goal is to retain high sensitivity to rare
# but critical events ('Correction', 'FOMO', 'Panic') while reducing the number
# of false alarms on the more common states ('Neutral', 'Herd').
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

# --- 1. Load the Prepared Data ---
print("--- Loading pre-processed data from .npy files ---")
try:
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    label_classes = np.load('label_encoder.npy', allow_pickle=True)
except FileNotFoundError:
    print("FATAL ERROR: .npy data files not found. Please run the data preparation script first.")
    sys.exit(1)
print("Data loaded successfully.")

# --- 2. Define Model Parameters ---
num_classes = len(label_classes)
input_shape = (X_train.shape[1], X_train.shape[2]) # Should be (60, 8)
print(f"\nModel will be trained with Input Shape: {input_shape}, Output Classes: {num_classes}")

# --- 3. Build the LSTM Model ---
print("\n--- Building the LSTM Model ---")
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=input_shape),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Manually Define Tuned Class Weights ---
print("\n--- Using MANUALLY TUNED Class Weights for Final Model ---")

# These weights were determined through an iterative process to balance
# the model's sensitivity. The goal is to ensure rare, high-impact events
# are not ignored, without making the model overly reactive.
class_weight_dict = {
    0: 50.0,  # Correction (Index 0) - High weight for this rare warning sign
    1: 250.0, # FOMO (Index 1) - Extremely high weight for this very rare, critical top signal
    2: 5.0,   # Herd (Index 2) - Boosted to improve precision for the main entry signal
    3: 1.0,   # Neutral (Index 3) - Boosted from automatic baseline to give the model more confidence
    4: 8.0    # Panic (Index 4) - Boosted to ensure it's respected as a key exit signal
}

print("Tuned weights being applied:")
# Print the mapping for verification
for i, class_name in enumerate(label_classes):
    weight = class_weight_dict.get(i, 1.0) # Default to 1.0 if index is somehow missing
    print(f"  - Class '{class_name}' (Index {i}): Weight = {weight:.2f}")

# --- 5. Start Final Model Training ---
print("\n--- Starting Final Model Training ---")

# Define the final model filename
MODEL_FILENAME = 'final_psychology_model.keras'

# Callbacks to ensure we get the best possible model without overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_FILENAME, save_best_only=True, monitor='val_loss')

# The fit method is where the training happens.
# The model will learn from X_train/y_train and validate its performance
# against X_test/y_test after each epoch.
history = model.fit(
    X_train, y_train,
    epochs=100, # Set a high number; EarlyStopping will find the optimal point
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict, # <-- Applying our manually tuned weights
    verbose=1
)

print("\n--- Final Model Training Complete ---")
print(f"The final, tuned model has been saved as '{MODEL_FILENAME}'")
print("This model is now ready for evaluation (Step 2.3) and final backtesting (Step 3).")