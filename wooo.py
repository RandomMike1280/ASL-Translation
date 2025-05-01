import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the hand landmark detection model
model_path = 'hand_landmarker.task'  # Download this from the official MediaPipe model repo

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback (not needed for video mode but must be set)
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    pass  # No-op here, as we're using synchronous mode

# Initialize the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

# Start webcam
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Run inference
        result = landmarker.detect_for_video(mp_image, frame_idx * 1000 // 30)
        frame_idx += 1

        # Draw landmarks
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for lm in hand:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow("MediaPipe Tasks - HandLandmarker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
