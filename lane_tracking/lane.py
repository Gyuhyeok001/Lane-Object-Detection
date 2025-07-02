def detect_lane(frame):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define Region of Interest (ROI) - focus on bottom 60% of the image
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height),
        (0, int(height * 0.4)),  # Top edge of ROI set to 40% of image height
        (width, int(height * 0.4)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=40, maxLineGap=150)

    # Create an empty image to draw lines on
    line_image = np.zeros_like(frame)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Overlay the detected lines on the original frame
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result

