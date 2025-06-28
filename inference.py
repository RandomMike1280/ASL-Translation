# main.py
import cv2
from helper import *
import time
import numpy as np
import torch
import string
from network import Model
from train import Tokenizer
import pickle

# Initialize MediaPipe Holistic model for landmark tracking
hands = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tokenizer = Tokenizer()
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load classification model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Labels list must match training order
classes = list(string.ascii_uppercase) + ['None']
# model = Model(vocab_size=5)
# model.load_state_dict(torch.load('models/model.pth', map_location=device, weights_only=True))
model = torch.load('models/model.pth', map_location=device, weights_only=False)
model.to(device)
model.eval()

# Precompute distance index pairs for landmark tracking
indices1, indices2 = create_distance_indices()

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0) # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

prev_time = 0

while True:
    # Read frame from camera
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check for detected hands (left and/or right)
    hand_landmarks_list = []
    if results.left_hand_landmarks:
        hand_landmarks_list.append(results.left_hand_landmarks)
    if results.right_hand_landmarks:
        hand_landmarks_list.append(results.right_hand_landmarks)
    for hand_landmarks in hand_landmarks_list:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                small_dots,
                small_lines)

            # Compute wrist position for label placement
            wrist_landmark = hand_landmarks.landmark[0]
            h, w, _ = image.shape
            text_x = int(wrist_landmark.x * w)
            text_y = int(wrist_landmark.y * h) - 20

            # Extract and track landmarks using helper functions
            # landmarks_xy = extract_landmarks(results)
            # distances = calculate_distances_vectorized(landmarks_xy, indices1, indices2)
            # x_tensor = torch.from_numpy(distances).float().to(device)
            # with torch.no_grad():
            #     logits = model(x_tensor.unsqueeze(0))
            #     pred_idx = logits.argmax(dim=1).item()
            # label = classes[pred_idx]
            # cv2.putText(image, label, (text_x, text_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Tracking - Left/Right', image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Hand tracking stopped.")