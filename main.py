# main.py
import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import string
from train_mlp import MLP

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Limit detection to 1 hand, set confidence levels
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load classification model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Labels list must match training order
classes = list(string.ascii_uppercase) + ['None']
model = MLP(input_dim=210, layers=[128, 64, 32], output_dim=len(classes))
model.load_state_dict(torch.load('mlp_model.pth', map_location=device))
model.to(device)
model.eval()

# Precompute index pairs for distances
n_kp = 21
pairs = [(i, j) for i in range(n_kp) for j in range(i+1, n_kp)]

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

    if results.multi_hand_landmarks:
        # Iterate through detected hands (already limited to 1 by initialization)
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Compute wrist position for label placement
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = image.shape
            text_x = int(wrist_landmark.x * w)
            text_y = int(wrist_landmark.y * h) - 20

            # Prepare feature vector and predict gesture
            coords = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
            dist_vec = [np.hypot(coords[i][0] - coords[j][0],
                                 coords[i][1] - coords[j][1]) for i, j in pairs]
            x_tensor = torch.from_numpy(np.array(dist_vec)).float().to(device)
            with torch.no_grad():
                logits = model(x_tensor.unsqueeze(0))
                pred_idx = logits.argmax(dim=1).item()
            label = classes[pred_idx]

            cv2.putText(image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


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