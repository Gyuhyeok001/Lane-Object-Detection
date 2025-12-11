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

# --- ROS2 & math-related imports ---
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

# Select device (GPU if available, otherwise CPU)
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = DetectMultiBackend('yolov5s.pt', device=device)
names = model.names
model.eval()

# Open input video (captured from virtual camera setup)
video_path = os.path.abspath('input/input.video.mp4')  # Adjust filename as needed
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)           # Frames per second of input video
frame_limit = int(fps * 10)               # Process up to 10 seconds
frame_count = 0

# Get frame size from the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    exit()
frame_height, frame_width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)       # Reset to first frame

# Prepare output video writer with original frame size and FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/fusion_output.mp4', fourcc, fps, (frame_width, frame_height))

# ---------- ROS2 initialization & Twist publisher creation ----------
rclpy.init()
node = rclpy.create_node('lane_fusion_node')
cmd_pub = node.create_publisher(Twist, 'cmd_vel', 10)
print("[INFO] ROS2 node 'lane_fusion_node' started, publishing to /cmd_vel")
# -------------------------------------------------------------------


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
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1
                )

    # Apply lane detection on the frame with bounding boxes
    lane_frame = detect_lane(frame.copy())

    # ---------- Compute Twist command from lane_frame & publish ----------
    # Simple example controller:
    # - Use the bottom 40% of the image as ROI
    # - Treat the average x-position of bright pixels as "lane center"
    # - Compute how far it is from the image center (left/right offset)
    gray = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2GRAY)

    # ROI: use only the bottom 40% of the image
    roi = gray[int(frame_height * 0.6):, :]
    ys, xs = np.where(roi > 150)  # pixels brighter than threshold 150

    twist = Twist()

    if len(xs) > 0:
        lane_center = float(np.mean(xs))
        image_center = frame_width / 2.0

        # Normalized error in range approximately [-1, 1]
        error = (lane_center - image_center) / (frame_width / 2.0)

        # Simple proportional control gains (tunable)
        k_angular = 0.8
        k_linear = 0.2

        # Higher speed near center, reduce speed if far from center
        twist.linear.x = k_linear * (1.0 - min(abs(error), 1.0))
        # If lane is on the right (error > 0), turn left (negative angular.z), and vice versa
        twist.angular.z = -k_angular * error
    else:
        # If no lane is detected, stop for safety
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    cmd_pub.publish(twist)
    # Non-blocking spin to process ROS2 callbacks (if any)
    rclpy.spin_once(node, timeout_sec=0.0)
    # -------------------------------------------------------------------

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

# Clean up ROS2
node.destroy_node()
rclpy.shutdown()
print("[INFO] Finished. ROS2 node shut down.")