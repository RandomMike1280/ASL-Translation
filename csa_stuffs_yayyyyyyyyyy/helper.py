import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Constants and Initialization ---
NUM_FACE_LANDMARKS = 478
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + 2 * NUM_HAND_LANDMARKS

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

small_dots = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
small_lines = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

def extract_landmarks(results) -> np.ndarray:
    """Extracts landmark coordinates (x, y) into a single NumPy array."""
    landmarks_array = np.zeros((TOTAL_LANDMARKS, 2), dtype=np.float32)
    sources = [
        (results.pose_landmarks, 0, NUM_POSE_LANDMARKS),
        (results.face_landmarks, NUM_POSE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS),
        (results.left_hand_landmarks, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS),
        (results.right_hand_landmarks, NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS, TOTAL_LANDMARKS)
    ]
    for landmarks, start, end in sources:
        if landmarks:
            coords = [(lm.x, lm.y) for lm in landmarks.landmark]
            coords += [(0, 0)] * (end - start - len(coords))  # Fill with zeros if fewer landmarks
            landmarks_array[start:end] = np.array(coords, dtype=np.float32)
    return landmarks_array

def create_distance_indices():
    """Pre-computes the pairs of indices needed for the distance calculation."""
    indices1, indices2 = [], []
    indices1.extend([0, 1, 2])
    indices2.extend([1, 2, 0])
    for i in range(2, TOTAL_LANDMARKS):
        indices1.extend([i, i])
        indices2.extend([i - 2, i - 1])
    indices1.append(0)
    indices2.append(TOTAL_LANDMARKS - 1)
    return np.array(indices1), np.array(indices2)

def calculate_distances_vectorized(landmarks_xy: np.ndarray, indices1: np.ndarray, indices2: np.ndarray) -> np.ndarray:
    """Calculates distances between specified landmark pairs in a vectorized manner."""
    points1 = landmarks_xy[indices1]
    points2 = landmarks_xy[indices2]
    distances = np.linalg.norm(points1 - points2, axis=1)
    zero_mask = np.all(points1 == 0, axis=1) | np.all(points2 == 0, axis=1)
    distances[zero_mask] = 0
    return distances