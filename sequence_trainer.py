import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- Configuration ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
print(f"Found actions: {actions}")

# 1. Load Data
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

print("Loading data...")
# We need to determine the sequence length from the first file we find
first_seq_len = None

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]

    for file_name in files:
        window = np.load(os.path.join(action_path, file_name))

        # Check consistency
        if first_seq_len is None:
            first_seq_len = window.shape[0]  # e.g., 60 frames
            print(f"Detected sequence length: {first_seq_len} frames")

        if window.shape[0] != first_seq_len:
            print(f"WARNING: Skipping {file_name} in {action}. Length {window.shape[0]} != {first_seq_len}")
            continue

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Check if data loaded correctly
if X.shape[0] == 0:
    print("Error: No valid data found. Did you run the collector?")
    exit()

print(f"Data shape: {X.shape}")  # (Samples, Frames, Features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 2. Build LSTM Model
print("Building LSTM Model...")
model = Sequential()

# LSTM Layers
# return_sequences=True because the next layer is also an LSTM
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(first_seq_len, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # Output layer

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 3. Train
print("Starting training...")
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='logs')])

# 4. Save
model.summary()
model.save('rsl_sequence_model.h5')
print("Model saved as 'rsl_sequence_model.h5'")