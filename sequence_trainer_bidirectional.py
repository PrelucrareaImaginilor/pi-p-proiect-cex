import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

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

model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
model.add(LSTM(64, return_sequences=False, activation='tanh'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_data=(X_test, y_test),
    callbacks=[TensorBoard(log_dir='logs'), early_stop]
)

model.save('rsl_improved_model.h5')
np.save('model_meta.npy', np.array([max_length]))
print(f"Model saved. Max length {max_length}.")


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# calcul metrici
start_time = time.time()
for _ in range(100):
    model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)
end_time = time.time()
avg_inference_time = (end_time - start_time) / 100
print(f"Average Inference Time per Sequence: {avg_inference_time*1000:.2f} ms")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

y_pred_probs = model.predict(X_train, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_train, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=actions))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_result_lstm.png')
print("Confusion matrix saved as 'confusion_matrix_result_lstm.png'")