import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = os.path.join('MP_Data_Improved')
actions = np.array([n for n in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, n))])
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
max_length = 0

print("Loading and padding data...")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for file_name in [f for f in os.listdir(action_path) if f.endswith('.npy')]:
        res = np.load(os.path.join(action_path, file_name))
        sequences.append(res)
        labels.append(label_map[action])
        if len(res) > max_length: max_length = len(res)

X = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float32')
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(max_length, 258)))

# --- CRITICAL FIX FOR MAC M-SERIES ---
# Changed activation from 'relu' to 'tanh'.
# 'tanh' is stable and prevents the NaN (Not a Number) error.
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
model.add(LSTM(64, return_sequences=False, activation='tanh'))

model.add(Dense(64, activation='relu')) # Dense layers are fine with relu
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=650, callbacks=[TensorBoard(log_dir='logs')])

model.save('rsl_improved_model.h5')
np.save('model_meta.npy', np.array([max_length]))
print(f"Model saved. Max length {max_length}.")