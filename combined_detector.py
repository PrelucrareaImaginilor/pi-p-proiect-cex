import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import os
from tensorflow.keras.models import load_model

# --- ENVIRONMENT CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# --- SETUP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =============================================================================
# 1. LOAD STATIC MODEL (Alphabet Detector)
# =============================================================================
print("Loading Static Model (Alphabet)...")
static_model = None
static_features = None
static_scaler = None

try:
    if os.path.exists('rsl_model.pkl') and os.path.exists('rsl_features.joblib'):
        static_model = joblib.load('rsl_model.pkl')
        static_features = joblib.load('rsl_features.joblib')
        if os.path.exists('rsl_scaler.pkl'):
            static_scaler = joblib.load('rsl_scaler.pkl')
        print("Success: Static model loaded.")
    else:
        print("Warning: Static model files not found.")
except Exception as e:
    print(f"Error loading static model: {e}")

# =============================================================================
# 2. LOAD DYNAMIC MODEL (Sequence/Word Detector)
# =============================================================================
print("Loading Dynamic Model (Sequence)...")
dynamic_model = None
actions = []
max_length = 30  # Default value

DATA_PATH = os.path.join('MP_Data_Improved')
MODEL_PATH = 'rsl_improved_model.h5'
META_PATH = 'model_meta.npy'

try:
    if os.path.exists(DATA_PATH):
        actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        dynamic_model = load_model(MODEL_PATH)
        max_length = int(np.load(META_PATH)[0])
        print(f"Success: Dynamic model loaded. Sequence length: {max_length}")
        dynamic_model.predict(np.zeros((1, max_length, 258)), verbose=0)  # Warmup
    else:
        print(f"Warning: Dynamic model files not found.")
except Exception as e:
    print(f"Error loading dynamic model: {e}")


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def get_static_features(hand_landmarks):
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


def extract_dynamic_keypoints(results):
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


# =============================================================================
# 4. MAIN APPLICATION
# =============================================================================

# Config
MODE_STATIC = 0
MODE_DYNAMIC = 1
current_mode = MODE_STATIC

# Visuals
BOX_TOP_LEFT = (20, 150)
BOX_BOTTOM_RIGHT = (120, 250)
STATIC_COLOR = (255, 0, 0)  # Blue
DYNAMIC_COLOR = (245, 117, 16)  # Orange
BAR_COLOR = (50, 50, 50)  # Dark Grey

# Buffers
sequence = []  # Dynamic model input
sentence_history = ""  # Main history string (Raw Text)
last_dynamic_token = ""  # To prevent "Hello Hello" spam in dynamic mode

dynamic_predictions = []  # Stabilization
static_predictions = []  # Stabilization
threshold = 0.8

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- CONTROLS ---")
print(" [SPACEBAR] : Toggle Mode (Adds space if entering Dynamic)")
print(" [c]        : Clear History")
print(" [q]        : Quit")
print("----------------")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- INPUT HANDLING ---
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        if key == 32:  # Spacebar
            current_mode = 1 - current_mode
            sequence = []  # Reset dynamic buffer
            static_predictions = []  # Reset static buffer

            # REQUIREMENT: "when i move to dynamic put a space next"
            #if current_mode == MODE_DYNAMIC:
            sentence_history += " "

            print(f"Switched to {'DYNAMIC' if current_mode == MODE_DYNAMIC else 'STATIC'} mode")

        if key == ord('c'):
            sentence_history = ""
            last_dynamic_token = ""
            print("History Cleared")

        # --- PROCESS IMAGE ---
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        # --- DRAW LANDMARKS ---
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if current_mode == MODE_DYNAMIC:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # ==========================
        # STATIC MODE LOGIC
        # ==========================
        if current_mode == MODE_STATIC:
            current_letter = ""
            conf = 0.0

            if static_model:
                try:
                    rh_data = get_static_features(results.right_hand_landmarks)
                    lh_data = get_static_features(results.left_hand_landmarks)
                    input_data = rh_data + lh_data

                    input_df = pd.DataFrame([input_data], columns=static_features)
                    if static_scaler:
                        input_df = static_scaler.transform(input_df)

                    if not pd.DataFrame([input_data]).empty and np.any(np.array(input_data) != 0):
                        pred = static_model.predict(input_df)[0]
                        conf = np.max(static_model.predict_proba(input_df))
                        current_letter = pred

                        # --- STATIC STABILIZATION ---
                        if conf > 0.75:
                            static_predictions.append(current_letter)
                            static_predictions = static_predictions[-8:]  # Keep last 8

                            # If stable for 8 frames
                            if len(static_predictions) == 8 and len(set(static_predictions)) == 1:
                                stable_char = static_predictions[0]

                                # REQUIREMENT: "letters with no space"
                                # We treat the alphabet as a typewriter.
                                # Simple Debounce: Don't add if it was the *immediately* previous addition
                                # (unless user clears buffer by moving hand, which the buffer reset handles below)

                                sentence_history += stable_char

                                # Clear buffer so we don't add 'A' 100 times a second
                                # User must hold pose -> Add -> Move/Shake/Wait -> Add again
                                static_predictions = []
                    else:
                        static_predictions = []

                except Exception as e:
                    pass

            # Static UI
            cv2.rectangle(image, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, (255, 255, 255), cv2.FILLED)
            cv2.rectangle(image, BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, STATIC_COLOR, 5)
            if conf > 0.65:
                text_size = cv2.getTextSize(current_letter, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = BOX_TOP_LEFT[0] + (BOX_BOTTOM_RIGHT[0] - BOX_TOP_LEFT[0] - text_size[0]) // 2
                text_y = BOX_BOTTOM_RIGHT[1] - (BOX_BOTTOM_RIGHT[1] - BOX_TOP_LEFT[1] - text_size[1]) // 2
                cv2.putText(image, current_letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, STATIC_COLOR, 5)

            cv2.putText(image, "MODE: STATIC", (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, STATIC_COLOR, 2)

        # ==========================
        # DYNAMIC MODE LOGIC
        # ==========================
        elif current_mode == MODE_DYNAMIC:
            if dynamic_model:
                keypoints = extract_dynamic_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-max_length:]

                hands_present = results.left_hand_landmarks or results.right_hand_landmarks
                if hands_present and len(sequence) >= 15:
                    input_pad = np.expand_dims(sequence, axis=0)
                    pad_amt = max_length - len(sequence)
                    if pad_amt > 0:
                        input_pad = np.pad(input_pad, ((0, 0), (0, pad_amt), (0, 0)), mode='constant')

                    res = dynamic_model.predict(input_pad, verbose=0)[0]
                    best_idx = np.argmax(res)
                    conf = res[best_idx]

                    if conf > threshold:
                        dynamic_predictions.append(best_idx)
                        if len(dynamic_predictions) > 10:
                            dynamic_predictions = dynamic_predictions[-10:]

                        if len(dynamic_predictions) >= 10 and np.unique(dynamic_predictions)[0] == best_idx:
                            current_word = actions[best_idx]

                            # Prevent immediate repetition (standard dynamic behavior)
                            if current_word != last_dynamic_token:
                                # REQUIREMENT: "between words in the dynamic mode i want spaces"
                                # Check if we need to insert a space before this word
                                if len(sentence_history) > 0 and not sentence_history.endswith(" "):
                                    sentence_history += " "

                                sentence_history += current_word
                                last_dynamic_token = current_word

            cv2.putText(image, "MODE: DYNAMIC", (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, DYNAMIC_COLOR, 2)

        # ==========================
        # TOP PREDICTION BAR
        # ==========================
        cv2.rectangle(image, (0, 0), (640, 40), BAR_COLOR, -1)

        # Display tail of history (approx last 30 chars to fit screen)
        text_to_show = sentence_history[-30:]

        cv2.putText(image, text_to_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'c' to Clear", (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Unified Detector', image)

cap.release()
cv2.destroyAllWindows()