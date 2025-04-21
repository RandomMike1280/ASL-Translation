import cv2
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO('best (1).pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Render results
    rendered_frame = results[0].plot()
    
    # Display output
    cv2.imshow('YOLO Pose Estimation', rendered_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()