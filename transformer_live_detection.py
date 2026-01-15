import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data_Improved')
MODEL_PATH = 'transformer_model.keras'
META_PATH = 'model_meta.npy'
IDLE_LABEL = 'idle'


try:
    actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
except FileNotFoundError:
    print(f"Eroare: Folderul {DATA_PATH} nu a fost gasit.")
    actions = []

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    print("Eroare: Modelul sau metadata lipsesc.")
    exit()

print("Se incarca modelul Transformer...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Eroare la incarcare: {e}")
    model = load_model(MODEL_PATH, custom_objects={
        "MultiHeadAttention": tf.keras.layers.MultiHeadAttention,
        "LayerNormalization": tf.keras.layers.LayerNormalization
    })

max_length = int(np.load(META_PATH)[0])
print(f"Model incarcat. Lungime secventa: {max_length}")

print("Warming up...")
dummy_input = np.zeros((1, max_length, 258))
model.predict(dummy_input, verbose=0) # predictie fortata pentru warm-up
print("Sistem gata!")



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



def check_hands_active(results):
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return False

    if results.pose_landmarks:
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        hip_level = (left_hip.y + right_hip.y) / 2

        hands_down = True
        if results.left_hand_landmarks:
            lh_wrist_y = results.left_hand_landmarks.landmark[0].y
            if lh_wrist_y < hip_level:
                hands_down = False

        if results.right_hand_landmarks:
            rh_wrist_y = results.right_hand_landmarks.landmark[0].y
            if rh_wrist_y < hip_level:
                hands_down = False

        if hands_down:
            return False

    return True


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
        if num >= len(actions): break
        if np.isnan(prob): continue

        color = (100, 100, 100) if actions[num] == IDLE_LABEL else colors[num % len(colors)]

        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


sequence = []
sentence = []
predictions = []
threshold = 0.85

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] * 5
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.6, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-max_length:]

        hands_present = results.left_hand_landmarks or results.right_hand_landmarks
        is_active = check_hands_active(results)

        if len(sequence) == max_length and (hands_present or is_active):
            input_data = np.expand_dims(sequence, axis=0)
            res = model.predict(input_data, verbose=0)[0]

            best_idx = np.argmax(res)
            confidence = res[best_idx]
            current_action_name = actions[best_idx]

            image = prob_viz(res, actions, image, colors)

            if confidence > threshold:
                predictions.append(best_idx)

                if np.unique(predictions[-10:])[0] == best_idx:
                    # afisare actiune doar daca nu e IDLE
                    if current_action_name != IDLE_LABEL:

                        if len(sentence) > 0:
                            if current_action_name != sentence[-1]:
                                sentence.append(current_action_name)
                        else:
                            sentence.append(current_action_name)


            if len(sentence) > 5:
                sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "Press 'c' to Clear", (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Transformer Live Detector', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            sentence = []
            predictions = []
            sequence = []

cap.release()
cv2.destroyAllWindows()