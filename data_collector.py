import cv2
import mediapipe as mp
import time
import os
import csv
import numpy as np

# --- Setup ---
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

# --- CSV Setup ---
# Define the landmarks to save (all 21 hand landmarks FOR EACH HAND)
# We will save (21 * 3) + (21 * 3) = 126 columns + 1 'label' column
num_landmarks = 21
base_landmark_names = [f'{i}_{axis}' for i in range(num_landmarks) for axis in ['x', 'y', 'z']]
# Add prefixes for right and left hands
landmark_names = [f'right_{name}' for name in base_landmark_names] + [f'left_{name}' for name in base_landmark_names]
csv_header = ['label'] + landmark_names

# CSV file to save the data
csv_file_name = 'rsl_landmarks.csv'

# Create the file and write the header if it doesn't exist
if not os.path.exists(csv_file_name):
    with open(csv_file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

print("Data collector started. Press 'q' to quit.")
print("Press 'a', 'b', 'c', etc., to record the sign for that letter.")

current_label = None

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

    # --- Landmark Extraction and Normalization ---
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

    # --- FPS Calculation ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # --- Check for Key Press ---
    key = cv2.waitKey(10) & 0xFF

    if key == ord('='):  # Changed 'fn' to 'q' as ord() requires a single character
        break

    # Check if the key is a letter (a-z)
    if key >= ord('a') and key <= ord('z'):
        current_label = chr(key)
        print(f"Recording data for: '{current_label}'")

    # --- Save Data ---
    if current_label and (results.right_hand_landmarks or results.left_hand_landmarks):
        try:
            # Helper function to normalize and flatten landmarks
            def get_normalized_landmarks(hand_landmarks):
                if not hand_landmarks:
                    # Return a flat list of 63 zeros if hand is not detected
                    return [0.0] * (num_landmarks * 3)

                landmarks_list = hand_landmarks.landmark

                # Normalize relative to the wrist (landmark 0)
                wrist = [landmarks_list[0].x, landmarks_list[0].y, landmarks_list[0].z]

                normalized_landmarks = []
                for lm in landmarks_list:
                    normalized_landmarks.append(lm.x - wrist[0])
                    normalized_landmarks.append(lm.y - wrist[1])
                    normalized_landmarks.append(lm.z - wrist[2])
                return normalized_landmarks


            # Get landmarks for both hands
            right_hand_data = get_normalized_landmarks(results.right_hand_landmarks)
            left_hand_data = get_normalized_landmarks(results.left_hand_landmarks)

            # Combine all landmarks into one row
            row_data = right_hand_data + left_hand_data

            # 3. Save to CSV
            with open(csv_file_name, mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([current_label] + row_data)

            # Reset label after saving one frame to avoid saving duplicates
            # (Hold the key down to record multiple frames)
            # For continuous saving, remove the line below
            print(f"Saved frame for '{current_label}'")
            # To save continuously while holding, comment out the next line
            current_label = None

        except Exception as e:
            print(f"Error processing landmarks: {e}")

    # Display the current recording label
    if current_label:
        cv2.putText(img, f'RECORDING: {current_label}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Image", img)

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to {csv_file_name}")