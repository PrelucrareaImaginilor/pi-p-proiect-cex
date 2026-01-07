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
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuration ---
DATA_PATH = os.path.join('MP_Data')

# 1. Load Labels
# We re-read the folders to ensure we have the exact same classes as the trainer
# Note: This assumes you haven't deleted the folders since training.
try:
    actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
    # Just in case the OS sort order is different, we sort them to match standard behavior
    # (Ideally, we would load a 'labels.json', but this works if folders exist)
    # Note: If your trainer used a specific order, make sure this matches.
    # Usually, os.listdir order + to_categorical mapping aligns if consistent.
    # If your predictions are "scrambled" (saying 'A' when you mean 'B'), we might need to sort this.
    # For now, let's assume os.listdir is consistent.
    print(f"Loaded classes: {actions}")
except FileNotFoundError:
    print("Error: Could not find MP_Data folder to determine labels.")
    exit()

# 2. Load Model
model_path = 'rsl_sequence_model.h5'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Did you run the trainer?")
    exit()

print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")


# Helper Functions (Same as Collector)
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


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


def extract_keypoints(results):
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


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # Dynamic bar width based on probability
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# --- Main Variables ---
sequence = []
sentence = []
predictions = []
threshold = 0.65  # Confidence threshold (80%)

# Define some colors for the visualization bars
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 100, 200), (100, 200, 200)] * 5

cap = cv2.VideoCapture(0)

# Set up MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        # 1. Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Keep only the last 60 frames
        # IMPORTANT: This 60 must match the length you trained on.
        # If your training data varied, this might need adjustment,
        # but usually we truncate to the last 60.
        sequence = sequence[-60:]

        if len(sequence) == 60:
            # CHECK: Are hands actually present?
            # If both hands are missing, skip prediction or assume "Neutral"
            hands_present = results.left_hand_landmarks or results.right_hand_landmarks

            if hands_present:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                best_class_index = np.argmax(res)
                best_class_label = actions[best_class_index]
                confidence = res[best_class_index]

                print(f"Pred: {best_class_label} ({confidence:.2f})")

                predictions.append(best_class_index)

                # Visualization logic
                if np.unique(predictions[-10:])[0] == best_class_index:
                    if confidence > threshold:
                        if len(sentence) > 0:
                            if best_class_label != sentence[-1]:
                                sentence.append(best_class_label)
                        else:
                            sentence.append(best_class_label)
            else:
                # OPTIONAL: If no hands, clear predictions or print "Waiting for hands..."
                print("No hands detected - Skipping prediction")
                pass

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Only visualize probabilities if hands were present to generate 'res'
            if hands_present:
                image = prob_viz(res, actions, image, colors)

        # 4. Display Sentence at Top
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()