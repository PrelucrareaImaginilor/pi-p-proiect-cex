import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

csv_file_name = 'rsl_landmarks.csv'
try:
    data = pd.read_csv(csv_file_name)
except FileNotFoundError:
    print(f"Error: '{csv_file_name}' not found.")
    print("Please run '1_data_collector.py' first to create the dataset.")
    exit()

if data.empty:
    print("Error: The CSV file is empty. Please record some data first.")
    exit()

print(f"Loaded {len(data)} samples.")

X = data.drop('label', axis=1)
y = data['label']

feature_names = list(X.columns)
feature_file_name = 'rsl_features.joblib'
joblib.dump(feature_names, feature_file_name)
print(f"Feature names saved to '{feature_file_name}'.")


print("\nClass distribution:")
print(y.value_counts())

# split in 80% pt training, 20% pt testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")

print("Training RandomForestClassifier:")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Training complete.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_file_name = 'rsl_model.pkl'
joblib.dump(model, model_file_name)

print(f"\nModel saved to '{model_file_name}'.")
print("You can now run the live detector.")


print("\n\nFINAL MODEL EVALUATION (Static Random Forest)\n")

sample_input = X_test.iloc[[0]]

start_time = time.time()
for _ in range(100):
    model.predict(sample_input)
end_time = time.time()

avg_inference_time = (end_time - start_time) / 100
print(f"Average Inference Time per Frame: {avg_inference_time*1000:.4f} ms")

y_pred = model.predict(X_test)

labels = sorted(y.unique())

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.savefig('confusion_matrix_static.png')
print("Confusion matrix saved as 'confusion_matrix_static.png'")