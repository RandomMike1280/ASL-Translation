import os
import math
import json
import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe Setup (Initialize outside the processing function for efficiency) ---
# Use context managers for proper resource cleanup
# Add max_num_hands if you expect more than one hand per frame
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

# --- Data Loading Generator ---

def load_raw_dataset_folders(root_dir="labelling/dataset_raw"):
    """
    Generator function to iterate through numbered subfolders in the root directory,
    load the JSON file, and open the MP4 file with OpenCV.

    Args:
        root_dir (str): The root directory containing the numbered subfolders.

    Yields:
        tuple: A tuple containing (cv2.VideoCapture object, dictionary of JSON data, path of folder)
               for each valid folder.
               The caller is responsible for releasing the cv2.VideoCapture object.
    """
    print(f"Scanning directory: {root_dir}")

    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at {root_dir}")
        return # Exit generator

    # Sort folder names numerically
    folder_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()], key=int)

    if not folder_names:
         print(f"No numbered subdirectories found in {root_dir}")
         return # Exit generator


    for folder_name in folder_names:
        folder_path = os.path.join(root_dir, folder_name)
        print(f"Attempting to load data from folder: {folder_name}")

        try:
            # Find the .json and .mp4 files (assuming there's only one of each)
            json_file_path = None
            mp4_file_path = None
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".json"):
                    json_file_path = os.path.join(folder_path, filename)
                elif filename.lower().endswith(".mp4"):
                    mp4_file_path = os.path.join(folder_path, filename)

            if json_file_path is None or mp4_file_path is None:
                print(f"Warning: Skipping folder '{folder_name}'. Missing .json or .mp4 file.")
                continue # Skip to the next folder

            # Load JSON data
            json_data = None
            try:
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON from {json_file_path}: {e}")
                continue # Skip to the next folder
            except Exception as e:
                 print(f"Error opening/reading JSON from {json_file_path}: {e}")
                 continue # Skip to the next folder


            # Load video with OpenCV
            video_capture = cv2.VideoCapture(mp4_file_path)

            if not video_capture.isOpened():
                print(f"Error: Could not open video file: {mp4_file_path}")
                continue # Skip to the next folder

            # Yield the loaded data for this folder
            print(f"Successfully loaded data for folder: {folder_name}")
            yield video_capture, json_data, folder_name

            # Note: The caller is responsible for calling video_capture.release()
            # after processing frames from this video.

        except Exception as e:
            print(f"An unexpected error occurred processing folder {folder_name}: {e}")
            # Do not yield if an error occurred
            continue # Move on to the next folder

# --- MediaPipe Landmark Detection Function ---

def get_landmarks(frame, hands_detector, face_detector):
    """
    Processes a single video frame to detect hand and face landmarks using MediaPipe.

    Args:
        frame (np.ndarray): The input video frame (BGR format from OpenCV).
        hands_detector: An initialized mediapipe.solutions.hands.Hands object.
        face_detector: An initialized mediapipe.solutions.face_detection.FaceDetection object.

    Returns:
        dict: A dictionary containing the detected landmarks:
              {
                  'hands': list of dictionaries, where each dictionary represents a hand:
                           {'type': 'Left' or 'Right', 'landmarks': list of (x, y) normalized coords}.
                           Returns [] if no hands.
                  'face_position': tuple (x, y) of the normalized center coordinates
                                   of the first detected face's bounding box, or None
                                   if no face is detected.
              }
              Coordinates are normalized [0, 1].
    """
    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    # Note: multi_hand_landmarks and multi_handedness are parallel lists
    hands_results = hands_detector.process(frame_rgb)

    # Process with MediaPipe Face Detection
    face_results = face_detector.process(frame_rgb)

    landmarks_data = {
        'hands': [], # This will now be a list of dicts {'type': '...', 'landmarks': [...]}
        'face_position': None
    }

    # Extract hand landmarks and handedness (Left/Right classification)
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        # Ensure the lists are the same length (they should be)
        if len(hands_results.multi_hand_landmarks) == len(hands_results.multi_handedness):
            for i in range(len(hands_results.multi_hand_landmarks)):
                hand_landmarks = hands_results.multi_hand_landmarks[i]
                handedness = hands_results.multi_handedness[i]

                # Extract hand type ('Left' or 'Right')
                # handedness.classification is a list of classifications, take the first one
                hand_type = handedness.classification[0].label

                # Extract landmark coordinates (lm.z is ommited)
                hand_coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Add to the results list
                landmarks_data['hands'].append({
                    'type': hand_type,
                    'landmarks': hand_coords
                })
        else:
             print("Warning: Mismatch between multi_hand_landmarks and multi_handedness lengths.")


    # Extract face position (center of bounding box)
    if face_results.detections:
        # We only need the position, take the first detected face
        detection = face_results.detections[0]
        bbox_c = detection.location_data.relative_bounding_box
        # Calculate center of the normalized bounding box
        center_x = bbox_c.xmin + bbox_c.width / 2
        center_y = bbox_c.ymin + bbox_c.height / 2
        landmarks_data['face_position'] = (center_x, center_y)

    return landmarks_data

# --- Ermm stacking Landmark function ---

def stack_landmarks(landmarks):
    """
    Stacks Landmarks outputted by get_landmarks(). Returns None if either left, right hands and head is not detected.

    Args:
        landmarks: Landmarks returned by get_landmarks()

    Returns:
        list: A list containing landmarks stacked
    """
    face_position = landmarks['face_position']
    left_hand_landmarks = None
    right_hand_landmarks = None
    for h in landmarks['hands']:
        if h['type'] == 'Left':
            left_hand_landmarks = h['landmarks']
        elif h['type'] == 'Right':
            right_hand_landmarks = h['landmarks']

    if not (face_position and left_hand_landmarks and right_hand_landmarks):
        return None
    
    stacked_landmarks = [face_position, *left_hand_landmarks, *right_hand_landmarks]
    return stacked_landmarks

# --- Distances encoding yaey ---

def encode_landmarks(landmarks):
    """
    Encodes Landmarks outputted by get_landmarks() into a pair-wise distances representation

    Args:
        landmarks: Landmarks returned by get_landmarks()

    Returns:
        list: Pair-wise distances
    """
    def _dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    
    stacked_landmarks = stack_landmarks(landmarks)

    # Distances are set to all 0 if not enough information
    if not stacked_landmarks:
        return [0 for _ in range(120)]

    p0 = stacked_landmarks[0] # the head lol
    distances = []
    for i in range(3, len(stacked_landmarks)):
        p1 = stacked_landmarks[i-1]
        p2 = stacked_landmarks[i-2]
        p = stacked_landmarks[i]
        distances.extend((
            _dist(p, p0),
            _dist(p, p1),
            _dist(p, p2)
        ))
    return distances
    
# --- PRO tokenizer. ---

class Tokenizer:
    '''
    Tokenizer Ultra Pro Max
    '''
    def __init__(self):
        vocab_list = [
            '<none>',
            'hello',
            'how',
            'you',
            'nice',
            'meet'
        ]

        self.vocab_dict = {}
        for i, word in enumerate(vocab_list):
            self.vocab_dict[word] = i

    def tokenize(self, word):
        token = self.vocab_dict.get(word, None)
        if token is None:
            raise Exception(f'Unknown token for word {word}')
        return token

# --- Main Execution Block ---

if __name__ == "__main__":
    tokenizer = Tokenizer()

    # Initialize MediaPipe detectors using context managers
    # Added max_num_hands=2 to allow detecting both hands simultaneously
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_detector, \
         mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:

        # Use the generator to iterate through the dataset
        for video_capture, labels, folder_path in load_raw_dataset_folders():
            # 'video_capture' is a cv2.VideoCapture instance
            # 'labels' is the dictionary from the JSON file

            # Optional: Get video properties if needed (e.g., frame rate, total frames)
            # fps = video_capture.get(cv2.CAP_PROP_FPS)
            # frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"\nProcessing video frames (FPS: {fps}, Total frames: {frame_count}) from loaded folder.")

            npz_distances = []
            npz_labels = []

            frame_idx = 0
            while video_capture.isOpened():
                ret, frame = video_capture.read()

                if not ret:
                    # End of video or error reading frame
                    # print(f"End of video or error reading frame {frame_idx}.")
                    break

                # Process the frame to get landmarks
                landmarks = get_landmarks(frame, hands_detector, face_detector)
                encoded = encode_landmarks(landmarks)
                encoded = np.array(encoded)
                npz_distances.append(encoded)

                label = labels.get(str(frame_idx), '<none>')
                label = tokenizer.tokenize(label)
                npz_labels.append(label)

                frame_idx += 1

            npz_distances = np.array(npz_distances)
            npz_labels = np.array(npz_labels)
            npz_file_name = os.path.splitext(os.path.basename(folder_path))[0]
            np.savez(fr'dataset/more_datasets/{npz_file_name}',
                distances=npz_distances,
                labels=npz_labels)

            # Release the video capture object for the current video
            video_capture.release()
            # print(f"Finished processing frames for current video.")

    print("\nFinished processing all videos.")
    # Although we didn't show windows, calling this is good practice
    cv2.destroyAllWindows()