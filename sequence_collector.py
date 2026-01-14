import cv2
import numpy as np
import os
import mediapipe as mp

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuration ---
DATA_PATH = os.path.join('MP_Data_Improved')  # New folder to avoid mixing with old data
FPS = 30


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
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
    """
    Revised feature extraction:
    - DISCARDS Face Mesh (1404 features of noise)
    - KEEPS Pose (132 features) + Hands (63+63 features)
    - Total features: 258
    """
    # 1. Pose (33 landmarks * 4 coords = 132)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # 2. Left Hand (21 landmarks * 3 coords = 63)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # 3. Right Hand (21 landmarks * 3 coords = 63)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])


def main():
    cap = cv2.VideoCapture(0)

    # Input handling
    action_name = input("Enter the word/sign name (e.g., 'hello'): ").strip()
    try:
        duration_sec = float(input("Enter recording duration in seconds (e.g., 2.0): "))
    except ValueError:
        duration_sec = 1.0

    # We determine frame count, but Trainer will handle variations now!
    sequence_length = int(FPS * duration_sec)

    action_path = os.path.join(DATA_PATH, action_name)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

    # Determine next sequence number
    dir_files = os.listdir(action_path)
    existing_nums = []
    for f in dir_files:
        if f.endswith('.npy'):
            try:
                num = int(f.split('.')[0])
                existing_nums.append(num)
            except:
                pass
    sequence_count = max(existing_nums) + 1 if existing_nums else 0

    print(f"\n--- RECORDING: '{action_name}' ---")
    print(f"Frames per clip: {sequence_length}")
    print("Press SPACE to record a sequence. Press 'q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # UI Text
            cv2.putText(image, f"Action: {action_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Count: {sequence_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Collector', image)
            key = cv2.waitKey(10) & 0xFF

            if key == 32:  # SPACE
                frames = []
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Extract reduced features
                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)

                    cv2.putText(image, "RECORDING...", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    cv2.imshow('Collector', image)
                    cv2.waitKey(1)

                # Save sequence
                npy_path = os.path.join(action_path, str(sequence_count))
                np.save(npy_path, np.array(frames))
                print(f"Saved sequence {sequence_count}")
                sequence_count += 1

            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()