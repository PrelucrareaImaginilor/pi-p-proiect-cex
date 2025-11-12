import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

# Separate features (X) and labels (y)
X = data.drop('label', axis=1) # All columns except 'label'
y = data['label']              # Only the 'label' column

# We save the column headers so the live detector can use them
feature_names = list(X.columns)
feature_file_name = 'rsl_features.joblib'
joblib.dump(feature_names, feature_file_name)
print(f"Feature names saved to '{feature_file_name}'.")


# Show class distribution (how many samples per letter)
print("\nClass distribution:")
print(y.value_counts())

# Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")

print("Training RandomForestClassifier:")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Training complete.")

# See how well the model performs on data it's never seen before (the test set)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

# Show a detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_file_name = 'rsl_model.pkl'
joblib.dump(model, model_file_name)

print(f"\nModel saved to '{model_file_name}'.")
print("You can now run the live detector.")