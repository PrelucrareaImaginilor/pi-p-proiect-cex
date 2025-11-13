# --- Suppress TensorFlow/MediaPipe Warnings ---
# v2: Stronger suppression
import warnings
import os

# Suppress TensorFlow C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress all Python UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to initialize absl logging to silence it before mediapipe does
try:
    from absl import logging

    logging.set_verbosity(logging.ERROR)
except ImportError:
    pass  # If absl isn't installed, we can't silence it
# --- End of Warning Suppression ---

import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import pandas as pd  # <-- Add this import

# --- 1. Load the Trained Model & Features ---
model_file_name = 'rsl_model.pkl'
feature_file_name = 'rsl_features.joblib'
try:
    model = joblib.load(model_file_name)
    feature_names = joblib.load(feature_file_name)  # <-- Load feature names
except FileNotFoundError as e:
    print(f"Error: Model or feature file not found. ({e})")
    print("Please run '2_train_model.py' first to train and save the model.")
    exit()

print("Model loaded successfully.")

# --- 2. Setup MediaPipe ---
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

# --- 3. Setup Display Box ---
# This is where we'll display the predicted letter
BOX_TOP_LEFT = (20, 150)
BOX_BOTTOM_RIGHT = (120, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 3
FONT_THICKNESS = 5
PREDICTION_COLOR = (255, 0, 0)  # Blue

# --- [MODIFIED] Toggles for Display ---
show_prediction = True  # Toggle with 'p'
show_landmarks = True  # Toggle with 'h'
show_fps = True  # Toggle with 'f'
print("--- Controls ---")
print("  p : Toggle Predictions")
print("  h : Toggle Hand Landmarks")
print("  f : Toggle FPS")
print("  q : Quit")
# --- [END MODIFIED] ---

current_prediction = ""
prediction_confidence = 0.0

while True:
    success, img = cap.read()
    if not success:
        continue

    # --- [MODIFIED] Handle Key Presses First ---
    # We check for keys at the start of the loop
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):  # Changed from 'f' to 'p'
        show_prediction = not show_prediction  # Toggle the boolean
        print(f"Show Predictions: {show_prediction}")
    if key == ord('h'):
        show_landmarks = not show_landmarks  # Toggle the boolean
        print(f"Show Landmarks: {show_landmarks}")
    if key == ord('f'):  # Added for FPS
        show_fps = not show_fps  # Toggle the boolean
        print(f"Show FPS: {show_fps}")
    # --- [END MODIFIED] ---

    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = holistic.process(imgRGB)

    # --- 4. Prediction Logic (This always runs in the background) ---
    try:
        # Helper function to normalize and flatten landmarks
        def get_normalized_landmarks(hand_landmarks):
            if not hand_landmarks:
                # Return a flat list of 63 zeros if hand is not detected
                return [0.0] * (21 * 3)

            landmarks_list = hand_landmarks.landmark

            # Normalize relative to the wrist (landmark 0)
            wrist = [landmarks_list[0].x, landmarks_list[0].y, landmarks_list[0].z]

            normalized_landmarks = []
            for lm in landmarks_list:
                normalized_landmarks.append(lm.x - wrist[0])
                normalized_landmarks.append(lm.y - wrist[1])
                normalized_landmarks.append(lm.z - wrist[2])
            return normalized_landmarks


        # --- 4a. Normalize Landmarks (Same as in data collector) ---
        right_hand_data = get_normalized_landmarks(results.right_hand_landmarks)
        left_hand_data = get_normalized_landmarks(results.left_hand_landmarks)

        # Combine all landmarks into one input feature vector
        input_data_list = right_hand_data + left_hand_data

        # --- 4b. Make Prediction ---
        # Create a pandas DataFrame with the feature names
        input_df = pd.DataFrame([input_data_list], columns=feature_names)

        # Get the prediction and the probabilities
        if not input_df.empty and np.any(input_df.values != 0):  # Check if the array is not all zeros
            prediction = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            current_prediction = prediction[0]
            prediction_confidence = np.max(probabilities)
        else:
            current_prediction = ""
            prediction_confidence = 0.0

    except Exception as e:
        print(f"Error during prediction: {e}")
        current_prediction = ""
        prediction_confidence = 0.0

    # --- [MODIFIED] 5. Display FPS (Now conditional) ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if show_fps:
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # --- [NEW] Conditional Landmark Drawing ---
    if show_landmarks:
        # Draw right hand
        if results.right_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                results.right_hand_landmarks,
                mpHolistic.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

        # Draw left hand
        if results.left_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                results.left_hand_landmarks,
                mpHolistic.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green for left
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    # --- [END NEW] ---

    # --- [NEW] Conditional Prediction Drawing ---
    if show_prediction:
        # --- 6. Display Prediction Box ---
        # Draw the box
        cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, PREDICTION_COLOR, FONT_THICKNESS)

        # Display the letter only if confidence is above a threshold
        if prediction_confidence > 0.65:  # Using the 0.65 from your script
            # Get text size to center it
            text_size = cv2.getTextSize(current_prediction, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            text_x = BOX_TOP_LEFT[0] + (BOX_BOTTOM_RIGHT[0] - BOX_TOP_LEFT[0] - text_size[0]) // 2
            text_y = BOX_BOTTOM_RIGHT[1] - (BOX_BOTTOM_RIGHT[1] - BOX_TOP_LEFT[1] - text_size[1]) // 2

            cv2.putText(img, current_prediction, (text_x, text_y), FONT, FONT_SCALE, PREDICTION_COLOR, FONT_THICKNESS)

        # Show the confidence
        cv2.putText(img, f'Conf: {int(prediction_confidence * 100)}%', (BOX_TOP_LEFT[0], BOX_TOP_LEFT[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # --- [END NEW] ---

    # --- 7. Show Image ---
    cv2.imshow("Image", img)

    # Note: key press logic was moved to the top of the loop
    # if cv2.waitKey(5) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()