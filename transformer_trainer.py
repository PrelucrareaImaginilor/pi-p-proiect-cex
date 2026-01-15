import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

DATA_PATH = os.path.join('MP_Data_Improved')
if not os.path.exists(DATA_PATH):
    print(f"Eroare: Folderul {DATA_PATH} nu exista.")
    exit()

actions = np.array([n for n in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, n))])
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
max_length = 0

print("Se incarca datele...")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    for file_name in files:
        res = np.load(os.path.join(action_path, file_name))
        sequences.append(res)
        labels.append(label_map[action])
        if len(res) > max_length:
            max_length = len(res)

print(f"Lungimea maxima a secventei gasita: {max_length}")

# Padding
X = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float32')
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer_model(input_shape, num_classes, max_len):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(128)(inputs)

    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = layers.Embedding(input_dim=max_len, output_dim=128)(positions)
    x = x + pos_emb

    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=512, dropout=0.1)
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=512, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


input_shape = (max_length, 258)
model = build_transformer_model(input_shape, len(actions), max_length)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# Callbacks
log_dir = os.path.join('logs', 'fit')
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stop = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=400,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, early_stop]
)

model.save('transformer_model.keras')
np.save('model_meta.npy', np.array([max_length]))

print(f"Model Transformer salvat cu succes (format .keras). Max length: {max_length}")


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
plt.savefig('confusion_matrix_result_tranformer.png')
print("Confusion matrix saved as 'confusion_matrix_result_tranformer.png'")