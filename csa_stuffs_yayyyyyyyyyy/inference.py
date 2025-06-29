import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import time
from collections import deque
import mediapipe as mp # Import mediapipe

# Assuming your translator_model.py and train.py are in the same directory
# We need the model architecture and the vocabulary
from translator_model import Encoder, Decoder, Seq2Seq
from csatrain import ASLTranslationDataset, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, NONE_TOKEN, MAX_TARGET_LEN # Import necessary components from train.py
# Import functions and drawing utilities from helper.py
from helper import extract_landmarks, create_distance_indices, calculate_distances_vectorized, mp_holistic, mp_drawing, small_dots, small_lines # Import drawing utilities

# --- Configuration --- #

# Model parameters (must match training configuration)
INPUT_DIM = 1106 # Dimension of your input vectors
EMBED_DIM = 256 # Dimension for token embeddings
HIDDEN_DIM = 256
N_LAYERS = 4
N_HEADS = 4
DROPOUT = 0.0 # Dropout is typically set to 0 for inference

# Inference parameters
MODEL_PATH = r'csa_stuffs_yayyyyyyyyyy\\checkpoints\\20.pth' # Path to your trained model checkpoint
DATA_DIRECTORY = 'dataset' # Path to your dataset folder (to load vocabulary)
WINDOW_SIZE = 30 # Number of frames in the sliding window (adjust as needed)
SLIDE_STEP = 10 # How many frames to slide the window by each step (adjust as needed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Vocabulary --- #
# Re-instantiate the dataset to load the vocabulary.
# In a production setting, you might save and load the vocabulary separately.
dataset = ASLTranslationDataset(DATA_DIRECTORY)
vocab = dataset.vocab
idx_to_token = dataset.idx_to_token

# --- Load Model --- #

# Instantiate the model with the same parameters as training
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(vocab), EMBED_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load the trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- Real-time Inference --- #

# Initialize video capture
cap = cv2.VideoCapture(0) # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# Initialize MediaPipe Holistic model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pre-compute distance indices
indices1, indices2 = create_distance_indices()

# Initialize a deque (double-ended queue) to store vectors for the sliding window
vector_buffer = deque(maxlen=WINDOW_SIZE)

print("Starting real-time inference. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(frame_rgb)

    # --- Hand Landmarking and Vector Extraction ---
    # Extract landmarks
    landmarks_xy = extract_landmarks(results)

    # Calculate distances
    distance_vector = calculate_distances_vectorized(landmarks_xy, indices1, indices2)

    # Convert to torch tensor and move to device
    input_vector = torch.FloatTensor(distance_vector).to(device)
    # input_vector shape: (INPUT_DIM,)

    # Add the current vector to the buffer
    vector_buffer.append(input_vector)

    # --- Sliding Window and Inference ---
    # Perform inference when the buffer is full enough for a window
    if len(vector_buffer) == WINDOW_SIZE:
        # Create a batch of one sequence from the buffer
        # The model expects input shape (batch_size, seq_len, input_dim)
        input_sequence_window = torch.stack(list(vector_buffer), dim=0).unsqueeze(0) # shape: (1, WINDOW_SIZE, INPUT_DIM)

        # Perform translation
        with torch.no_grad(): # No need to calculate gradients during inference
            # The translate method returns token indices
            predicted_token_indices = model.translate(input_sequence_window, max_len=5) # max_len is the desired output tokens (excluding SOS/EOS)
            # predicted_token_indices shape: (batch_size, predicted_seq_len) -> (1, <=5+2)

        # Process and display the predicted tokens
        # Remove SOS and EOS tokens for display
        predicted_tokens = [idx_to_token[idx.item()] for idx in predicted_token_indices[0] if idx.item() not in [dataset.sos_idx, dataset.eos_idx, dataset.pad_idx]]

        # Display the predicted tokens (you can customize this)
        print("Predicted Tokens:", predicted_tokens)

        # --- Slide the window ---
        # Remove the oldest 'SLIDE_STEP' vectors from the buffer
        for _ in range(SLIDE_STEP):
            if len(vector_buffer) > 0:
                vector_buffer.popleft()

    # --- Display Camera Feed with Landmarks ---
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # Draw face landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, small_dots, small_lines)
    # Draw pose landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, small_dots, small_lines)
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, small_dots, small_lines)
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, small_dots, small_lines)

    # Display the frame
    cv2.imshow('ASL Translation', frame)


    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
holistic.close() # Close the MediaPipe Holistic model
cap.release()
cv2.destroyAllWindows()
print("Inference stopped.")
