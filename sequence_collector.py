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
import time

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuration ---
DATA_PATH = os.path.join('MP_Data')  # Folder to save data
# We assume 30 FPS. If you type '2' seconds, we capture 60 frames.
FPS = 30


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
    # 1. Face (468 landmarks * 3 coords = 1404 values)
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    # 2. Pose (33 landmarks * 4 coords (x,y,z,visibility) = 132 values)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # 3. Left Hand (21 landmarks * 3 coords = 63 values)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # 4. Right Hand (21 landmarks * 3 coords = 63 values)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # Concatenate all into one big array
    return np.concatenate([pose, face, lh, rh])


# --- Main Logic ---
def main():
    cap = cv2.VideoCapture(0)

    # 1. Ask for Inputs
    action_name = input("Enter the word/sign name (e.g., 'hello'): ").strip()
    try:
        duration_sec = float(input("Enter recording duration in seconds (e.g., 2): "))
    except ValueError:
        print("Invalid number. Defaulting to 1 second.")
        duration_sec = 1.0

    sequence_length = int(FPS * duration_sec)

    # Create folder for this word
    action_path = os.path.join(DATA_PATH, action_name)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

    # Find the next sequence number (to avoid overwriting)
    dir_files = os.listdir(action_path)
    # Filter only .npy files and get their numbers
    existing_nums = []
    for f in dir_files:
        if f.endswith('.npy'):
            try:
                num = int(f.split('.')[0])
                existing_nums.append(num)
            except:
                pass

    sequence_count = max(existing_nums) + 1 if existing_nums else 0

    print(f"\n--- READY TO RECORD: '{action_name}' ---")
    print(f"Duration: {duration_sec}s ({sequence_length} frames)")
    print("Press SPACEBAR to start a recording.")
    print("Press 'q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Display Instructions
            cv2.putText(image, f"Word: {action_name}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Saved Count: {sequence_count}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Press SPACE to record", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            key = cv2.waitKey(10) & 0xFF

            # --- START RECORDING LOGIC ---
            if key == 32:  # Spacebar
                window = []  # List to hold all frames for this one video
                print(f"Recording sequence {sequence_count}...")

                # Loop for the specific duration
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)

                    # 1. Detect
                    image, results = mediapipe_detection(frame, holistic)

                    # 2. Draw (visual feedback)
                    draw_styled_landmarks(image, results)

                    # 3. Export Keypoints
                    keypoints = extract_keypoints(results)
                    window.append(keypoints)

                    # 4. Show "Recording" status
                    cv2.putText(image, "RECORDING...", (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1)  # Wait a tiny bit to allow screen update

                # Save the sequence
                npy_path = os.path.join(action_path, str(sequence_count))
                np.save(npy_path, np.array(window))
                print(f"Saved {npy_path}.npy")
                sequence_count += 1

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()