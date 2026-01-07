# --- Suppress Warnings ---
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from absl import logging

    logging.set_verbosity(logging.ERROR)
except ImportError:
    pass
# -------------------------

import cv2
import numpy as np
import os
import joblib
import pandas as pd
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# ==========================================
#        1. LOAD MODELS & CONFIG
# ==========================================

print("\n--- LOADING RSL SYSTEMS ---")

# --- A. Static Model (LSR Letters) ---
static_model = None
static_features = None
static_scaler = None

try:
    # We need ALL three files for the RSL letter detector to work
    if os.path.exists('rsl_model.pkl') and os.path.exists('rsl_scaler.pkl') and os.path.exists('rsl_features.joblib'):
        static_model = joblib.load('rsl_model.pkl')
        static_features = joblib.load('rsl_features.joblib')
        static_scaler = joblib.load('rsl_scaler.pkl')
        print("[OK] Static Mode (LSR Letters) Ready.")
    else:
        print("[WARNING] Static files missing. Run '2_train_model.py' to generate them.")
except Exception as e:
    print(f"[ERROR] Failed to load Static RSL model: {e}")

# --- B. Dynamic Model (Words/Sentences) ---
dynamic_model = None
actions = []
sequence_length = 30  # Default, will be updated by model input shape

try:
    if os.path.exists('rsl_sequence_model.h5'):
        dynamic_model = load_model('rsl_sequence_model.h5')

        # Auto-detect sequence length (usually 30 or 60)
        sequence_length = dynamic_model.input_shape[1]
        print(f"[OK] Dynamic Mode (Words) Ready. Sequence length: {sequence_length}")

        # Load Labels from MP_Data folder
        DATA_PATH = os.path.join('MP_Data')
        if os.path.exists(DATA_PATH):
            actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
            print(f"     Classes: {actions}")
        else:
            print("[WARNING] MP_Data folder not found. Dynamic labels might be wrong.")
    else:
        print("[WARNING] 'rsl_sequence_model.h5' not found. Dynamic mode disabled.")
except Exception as e:
    print(f"[ERROR] Failed to load Dynamic model: {e}")

# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def draw_styled_landmarks(image, results):
    # Draw Face Mesh
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    # Draw Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Draw Left Hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw Right Hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


# --- Feature Extraction for Static (Letters) ---
# Normalizes relative to wrist
def get_static_features(results):
    def get_normalized(hand_landmarks):
        if not hand_landmarks:
            return [0.0] * (21 * 3)
        landmarks_list = hand_landmarks.landmark
        wrist = [landmarks_list[0].x, landmarks_list[0].y, landmarks_list[0].z]
        normalized = []
        for lm in landmarks_list:
            normalized.append(lm.x - wrist[0])
            normalized.append(lm.y - wrist[1])
            normalized.append(lm.z - wrist[2])
        return normalized

    right = get_normalized(results.right_hand_landmarks)
    left = get_normalized(results.left_hand_landmarks)
    return right + left


# --- Feature Extraction for Dynamic (Words) ---
# Raw flattened coordinates including face/pose
def extract_dynamic_keypoints(results):
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
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
    return np.concatenate([pose, face, lh, rh])


# --- Visualization for Words ---
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


# ==========================================
#        3. MAIN APPLICATION LOOP
# ==========================================

cap = cv2.VideoCapture(0)

# Modes
MODE_STATIC = 0
MODE_DYNAMIC = 1
current_mode = MODE_STATIC

# Toggles
show_landmarks = True
show_fps = True

# Static Variables
static_pred = ""
static_conf = 0.0
BOX_TOP_LEFT = (20, 150)
BOX_BOTTOM_RIGHT = (120, 250)
FONT_STATIC = cv2.FONT_HERSHEY_SIMPLEX

# Dynamic Variables
sequence = []
sentence = []
predictions = []
threshold = 0.8
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 100, 200), (100, 200, 200)] * 5

pTime = 0

print("\n--- CONTROLS ---")
print("  m : Switch Mode (Static <-> Dynamic)")
print("  h : Toggle Landmarks")
print("  f : Toggle FPS")
print("  q : Quit")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- Handle Key Inputs ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'): break
        if key == ord('h'): show_landmarks = not show_landmarks
        if key == ord('f'): show_fps = not show_fps
        if key == ord('m'):
            current_mode = 1 - current_mode  # Switch between 0 and 1
            sequence = []  # Reset sequence buffer on switch
            print(f"Switched to mode: {'DYNAMIC (Words)' if current_mode == MODE_DYNAMIC else 'STATIC (Letters)'}")

        # --- Process Image ---
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Draw Landmarks ---
        if show_landmarks:
            draw_styled_landmarks(image, results)

        # =================================================
        #        MODE 0: STATIC (RSL LETTERS)
        # =================================================
        if current_mode == MODE_STATIC:
            if static_model and static_scaler:
                try:
                    # 1. Extract Features (Wrist-Relative)
                    features = get_static_features(results)

                    # 2. Check Compatibility
                    if len(features) != len(static_features):
                        # Shape mismatch (e.g., using old model with new logic)
                        pass
                    else:
                        # 3. Scale & Predict
                        input_df = pd.DataFrame([features], columns=static_features)
                        input_scaled = static_scaler.transform(input_df)

                        if not input_df.empty and np.any(input_df.values != 0):
                            pred = static_model.predict(input_scaled)[0]
                            prob = np.max(static_model.predict_proba(input_scaled))

                            static_pred = pred
                            static_conf = prob
                        else:
                            static_pred = ""
                            static_conf = 0.0

                        # 4. Draw UI
                        if static_conf > 0.65:
                            # Box
                            cv2.rectangle(image, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 255, 255), cv2.FILLED)
                            cv2.rectangle(image, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 0, 0), 5)
                            # Letter
                            text_size = cv2.getTextSize(static_pred, FONT_STATIC, 3, 5)[0]
                            text_x = BOX_TOP_LEFT[0] + (BOX_BOTTOM_RIGHT[0] - BOX_TOP_LEFT[0] - text_size[0]) // 2
                            text_y = BOX_BOTTOM_RIGHT[1] - (BOX_BOTTOM_RIGHT[1] - BOX_TOP_LEFT[1] - text_size[1]) // 2
                            cv2.putText(image, static_pred, (text_x, text_y), FONT_STATIC, 3, (255, 0, 0), 5)

                        # Conf Text
                        cv2.putText(image, f'{int(static_conf * 100)}%', (BOX_TOP_LEFT[0], BOX_TOP_LEFT[1] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                except Exception as e:
                    # e.g., if hands are not visible, scaler might throw minor warning
                    pass
            else:
                cv2.putText(image, "Static Model Missing", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        # =================================================
        #        MODE 1: DYNAMIC (WORDS)
        # =================================================
        elif current_mode == MODE_DYNAMIC:
            if dynamic_model:
                # 1. Extract Features (Raw Flattened)
                keypoints = extract_dynamic_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-sequence_length:]  # Keep buffer at correct length

                if len(sequence) == sequence_length:
                    try:
                        # 2. Predict
                        res = dynamic_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        best_idx = np.argmax(res)
                        best_label = actions[best_idx]
                        confidence = res[best_idx]

                        # 3. Stabilization
                        predictions.append(best_idx)
                        if np.unique(predictions[-10:])[0] == best_idx:
                            if confidence > threshold:
                                if len(sentence) > 0:
                                    if best_label != sentence[-1]:
                                        sentence.append(best_label)
                                else:
                                    sentence.append(best_label)

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # 4. Viz
                        image = prob_viz(res, actions, image, colors)
                    except Exception as e:
                        pass

                # Sentence Bar
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Dynamic Model Missing", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # --- Top UI (Mode Indicator) ---
        mode_text = "MODE: DYNAMIC (Words)" if current_mode == MODE_DYNAMIC else "MODE: STATIC (LSR)"
        color = (245, 117, 16) if current_mode == MODE_DYNAMIC else (255, 0, 0)
        cv2.putText(image, mode_text, (320, 30 if current_mode == MODE_STATIC else 70), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    color, 2)

        # --- FPS Counter ---
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        if show_fps:
            cv2.putText(image, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('RSL Combined Detector', image)

cap.release()
cv2.destroyAllWindows()