import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import pandas as pd
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

model_file_name = 'rsl_model.pkl'
feature_file_name = 'rsl_features.joblib'
try:
    model = joblib.load(model_file_name)
    feature_names = joblib.load(feature_file_name)
except FileNotFoundError as e:
    print(f"Error: Model or feature file not found. ({e})")
    print("Please run '2_train_model.py' first to train and save the model.")
    exit()

print("Model loaded successfully.")

mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                                model_complexity=1,       # 1- balanced
                                refine_face_landmarks=False
)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

# caseta predictie
BOX_TOP_LEFT = (20, 150)
BOX_BOTTOM_RIGHT = (120, 250)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 3
FONT_THICKNESS = 5
PREDICTION_COLOR = (255, 0, 0)

current_prediction = ""
prediction_confidence = 0.0

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

    try:
        # normalizare landmark uri relativa la incheietura
        def get_normalized_landmarks(hand_landmarks):
            if not hand_landmarks:
                return [0.0] * (21 * 3)

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

        input_data_list = right_hand_data + left_hand_data

        # predictia
        input_df = pd.DataFrame([input_data_list], columns=feature_names)

        # face predictia daca exista macar o mana
        if not input_df.empty and np.any(input_df.values != 0):
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


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 255, 255), cv2.FILLED)
    cv2.rectangle(img, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, PREDICTION_COLOR, FONT_THICKNESS)

    if prediction_confidence > 0.65:
        text_size = cv2.getTextSize(current_prediction, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = BOX_TOP_LEFT[0] + (BOX_BOTTOM_RIGHT[0] - BOX_TOP_LEFT[0] - text_size[0]) // 2
        text_y = BOX_BOTTOM_RIGHT[1] - (BOX_BOTTOM_RIGHT[1] - BOX_TOP_LEFT[1] - text_size[1]) // 2

        cv2.putText(img, current_prediction, (text_x, text_y), FONT, FONT_SCALE, PREDICTION_COLOR, FONT_THICKNESS)

    cv2.putText(img, f'Conf: {int(prediction_confidence * 100)}%', (BOX_TOP_LEFT[0], BOX_TOP_LEFT[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()