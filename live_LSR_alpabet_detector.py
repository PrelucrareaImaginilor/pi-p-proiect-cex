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

# Prediction variables
current_prediction = ""
prediction_confidence = 0.0

while True:
    success, img = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = holistic.process(imgRGB)

    # --- 4. Prediction Logic ---
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
        # This prevents the sklearn UserWarning
        input_df = pd.DataFrame([input_data_list], columns=feature_names)

        # Get the prediction and the probabilities
        # Only predict if at least one hand is visible
        if not input_df.empty and np.any(input_df.values != 0):  # Check if the array is not all zeros
            # Use the DataFrame for prediction
            prediction = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            current_prediction = prediction[0]
            prediction_confidence = np.max(probabilities)
        else:
            # This is the correct place to set 0 confidence (when no hand is seen)
            current_prediction = ""
            prediction_confidence = 0.0

    except Exception as e:
        print(f"Error during prediction: {e}")
        current_prediction = ""
        prediction_confidence = 0.0

    # [--- DELETED ---]
    # I have removed the 'else:' block and the 'if not (results...)'
    # block that were here, as they were causing the bug.
    # [--- DELETED ---]

    # --- 5. Display FPS ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # --- 6. Display Prediction Box ---
    # Draw the box
    cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 255, 255), cv2.FILLED)
    cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, PREDICTION_COLOR, FONT_THICKNESS)

    # Display the letter only if confidence is above a threshold (e.g., 70%)
    if prediction_confidence > 0.65:
        # Get text size to center it
        text_size = cv2.getTextSize(current_prediction, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = BOX_TOP_LEFT[0] + (BOX_BOTTOM_RIGHT[0] - BOX_TOP_LEFT[0] - text_size[0]) // 2
        text_y = BOX_BOTTOM_RIGHT[1] - (BOX_BOTTOM_RIGHT[1] - BOX_TOP_LEFT[1] - text_size[1]) // 2

        cv2.putText(img, current_prediction, (text_x, text_y), FONT, FONT_SCALE, PREDICTION_COLOR, FONT_THICKNESS)

    # Show the confidence
    cv2.putText(img, f'Conf: {int(prediction_confidence * 100)}%', (BOX_TOP_LEFT[0], BOX_TOP_LEFT[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # --- 7. Show Image ---
    cv2.imshow("Image", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()