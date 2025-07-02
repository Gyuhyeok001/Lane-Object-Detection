import sys
import os

# Add yolov5 directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from lane_tracking.lane import detect_lane  # Custom lane detection function

# Select device (GPU if available, otherwise CPU)
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = DetectMultiBackend('yolov5s.pt', device=device)
names = model.names
model.eval()

# Open input video
video_path = os.path.abspath('input/input.video.mp4')  # Adjust filename as needed
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)                 # Frames per second of input video
frame_limit = int(fps * 10)                      # Process up to 10 seconds
frame_count = 0

# Get frame size from the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    exit()
frame_height, frame_width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)             # Reset to first frame

# Prepare output video writer with original frame size and FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/fusion_output.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >= frame_limit:
        break
    frame_count += 1
    print(f"[INFO] Processing frame {frame_count}")

    # Resize frame for YOLOv5 input (640x640)
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    # Run YOLOv5 inference
    with torch.no_grad():
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.4)

    # Calculate scaling factors to map YOLO boxes back to original frame size
    scale_h = frame_height / 640
    scale_w = frame_width / 640

    # Draw YOLO bounding boxes on original frame
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
                x1 = int(x1 * scale_w)
                x2 = int(x2 * scale_w)
                y1 = int(y1 * scale_h)
                y2 = int(y2 * scale_h)
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Apply lane detection on the frame with bounding boxes
    lane_frame = detect_lane(frame.copy())

    # Write processed frame to output video
    out.write(lane_frame)

    # Display the fusion result window
    cv2.imshow("Fusion Result", lane_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
