import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data_Improved')

try:
    actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
except FileNotFoundError:
    print(f"Error: Folder {DATA_PATH} not found. Did you run the collector?")
    actions = []

model_path = 'rsl_improved_model.h5'
meta_path = 'model_meta.npy'

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    print("Error: Model or metadata not found. Run the trainer first.")
    exit()

print("Loading model...")
model = load_model(model_path)
max_length = int(np.load(meta_path)[0])
print(f"Model loaded. Expecting sequence length: {max_length}")


print("Warming up the AI brain... (This prevents the freeze later)")
dummy_input = np.zeros((1, max_length, 258))
model.predict(dummy_input, verbose=0)  # predictie fortata pentru warm-up
print("Model warm-up complete! Starting camera...")



def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if np.isnan(prob): continue

        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame



sequence = []
sentence = []
predictions = []
threshold = 0.8

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] * 5
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holistic)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-max_length:]

        hands_present = results.left_hand_landmarks or results.right_hand_landmarks

        if hands_present and len(sequence) >= 15:
            input_data = np.expand_dims(sequence, axis=0)
            pad_amt = max_length - len(sequence)
            if pad_amt > 0:
                input_data = np.pad(input_data, ((0, 0), (0, pad_amt), (0, 0)), mode='constant')

            res = model.predict(input_data, verbose=0)[0]

            if np.any(np.isnan(res)):
                print("Warning: Model returned NaN. Skipping frame.")
            else:
                best_idx = np.argmax(res)
                confidence = res[best_idx]

                image = prob_viz(res, actions, image, colors)

                if confidence > threshold:
                    predictions.append(best_idx)
                    if np.unique(predictions[-10:])[0] == best_idx:
                        current_word = actions[best_idx]
                        if len(sentence) > 0:
                            if current_word != sentence[-1]:
                                sentence.append(current_word)
                        else:
                            sentence.append(current_word)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        text_to_show = ' '.join(sentence)
        cv2.putText(image, text_to_show, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'c' to Clear", (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Live Detector', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            sentence = []
            predictions = []

cap.release()
cv2.destroyAllWindows()