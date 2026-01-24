import cv2
import mediapipe as mp
import time
import os
import csv
import numpy as np

# Setup
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0


# 21 landmark uri pentru fiecare mana
# (21 * 3) + (21 * 3) = 126 coloane
num_landmarks = 21
base_landmark_names = [f'{i}_{axis}' for i in range(num_landmarks) for axis in ['x', 'y', 'z']]

landmark_names = [f'right_{name}' for name in base_landmark_names] + [f'left_{name}' for name in base_landmark_names]
csv_header = ['label'] + landmark_names

csv_file_name = 'rsl_landmarks.csv'

if not os.path.exists(csv_file_name):
    with open(csv_file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

print("Data collector started. Press '=' to quit.")
print("Press 'a', 'b', 'c', etc., to record the sign for that letter.")

current_label = None

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = holistic.process(imgRGB)

    if results.right_hand_landmarks:
        mpDraw.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
            mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

    if results.left_hand_landmarks:
        mpDraw.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green for left
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # calculc FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # salvare tasta apasata
    key = cv2.waitKey(10) & 0xFF

    if key == ord('='):  # '=' iesire
        break

    if key >= ord('a') and key <= ord('z'):
        current_label = chr(key)
        print(f"Recording data for: '{current_label}'")

    # salavre date
    if current_label and (results.right_hand_landmarks or results.left_hand_landmarks):
        try:
            # normalizare landmark uri relativa la incheietura
            def get_normalized_landmarks(hand_landmarks):
                if not hand_landmarks:
                    return [0.0] * (num_landmarks * 3)

                landmarks_list = hand_landmarks.landmark

                wrist = [landmarks_list[0].x, landmarks_list[0].y, landmarks_list[0].z]

                normalized_landmarks = []
                for lm in landmarks_list:
                    normalized_landmarks.append(lm.x - wrist[0])
                    normalized_landmarks.append(lm.y - wrist[1])
                    normalized_landmarks.append(lm.z - wrist[2])
                return normalized_landmarks


            right_hand_data = get_normalized_landmarks(results.right_hand_landmarks)
            left_hand_data = get_normalized_landmarks(results.left_hand_landmarks)

            row_data = right_hand_data + left_hand_data

            with open(csv_file_name, mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([current_label] + row_data)

            print(f"Saved frame for '{current_label}'")
            current_label = None

        except Exception as e:
            print(f"Error processing landmarks: {e}")

    if current_label:
        cv2.putText(img, f'RECORDING: {current_label}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Image", img)

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to {csv_file_name}")