import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

# --- Setup Holistic Model ---
mpHolistic = mp.solutions.holistic
# You can adjust min_detection_confidence and min_tracking_confidence if needed
holistic = mpHolistic.Holistic()
mpDraw = mp.solutions.drawing_utils

# Define drawing specs for a cleaner look (optional)
drawing_spec_hands = mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
drawing_spec_face = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Process the Image ONCE with Holistic ---
    results = holistic.process(imgRGB)

    # Convert image back to BGR for drawing
    # (Holistic processing doesn't change the original 'img' variable)

    # --- 1. Draw Face Landmarks ---
    # This replaces your face detection bounding box
    if results.face_landmarks:
        # Use FACEMESH_TESSELATION for the full mesh, or FACEMESH_CONTOURS for just the outline
        mpDraw.draw_landmarks(
            img,
            results.face_landmarks,
            mpHolistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,  # No individual landmarks
            connection_drawing_spec=drawing_spec_face)  # Draw connections

    # --- 2. Draw Hand Landmarks ---
    # Left Hand
    if results.left_hand_landmarks:
        mpDraw.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_spec_hands,
            connection_drawing_spec=drawing_spec_hands)

    # Right Hand
    if results.right_hand_landmarks:
        mpDraw.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mpHolistic.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_spec_hands,
            connection_drawing_spec=drawing_spec_hands)

    # --- FPS Calculation ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)

    cv2.imshow("Image", img)

    # Add an exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()