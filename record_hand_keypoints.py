"""
record_hand_keypoints.py

Record camera input at 7 FPS, detect hand landmarks using MediaPipe, and compute pairwise distances between all 21 keypoints per frame.
Press 'q' or ESC to stop recording. After stopping, saves:
  - output.mp4       : recorded video at 7 FPS
  - ./frames/        : all recorded frames as PNG images
  - distances.npy    : numpy array of shape (num_frames, C(21,2)=210) with pairwise distances; zero vectors if no full hand detected
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import os


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 7
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"record_{timestamp}"
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Precompute index pairs for C(21,2)
    n_kp = 21
    pairs = [(i, j) for i in range(n_kp) for j in range(i + 1, n_kp)]

    frames = []
    distances_list = []
    interval = 1.0 / fps
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        frame = cv2.flip(frame, 1)
        curr = time.time()
        if curr - prev_time >= interval:
            prev_time = curr
            frames.append(frame.copy())
            out.write(frame)

            # Hand detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                coords = [(l.x * width, l.y * height) for l in lm]
                dist_vec = []
                for i, j in pairs:
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    dist_vec.append(np.hypot(xi - xj, yi - yj))
                distances_list.append(dist_vec)
                mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            else:
                distances_list.append([0.0] * len(pairs))

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save frames and distances per run
    frame_files = []
    for idx, img in enumerate(frames):
        filename = f"{idx:06d}.png"
        path = os.path.join(frames_dir, filename)
        cv2.imwrite(path, img)
        frame_files.append(filename)
    distances_arr = np.array(distances_list)
    np.savez(os.path.join(output_dir, "data.npz"),
             distances=distances_arr,
             frames=np.array(frame_files))
    print(f"Saved recording in {output_dir}/")


if __name__ == '__main__':
    main()
