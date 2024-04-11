import cv2
import numpy as np
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640,480)}))
picam2.start()

def detect_obstacle(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for obstacle information
    obstacle_info = []

    # Iterate through detected contours
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)

        # Filter out small contours (noise)
        if area > 1000:
            # Get bounding box coordinates and dimensions
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate center of the obstacle
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate confidence (based on area or other metrics)
            confidence = area / (frame.shape[0] * frame.shape[1])

            # Add obstacle information to list
            obstacle_info.append((center_x, center_y, w, h, confidence))

            # Draw rectangle around obstacle on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display distance and confidence in terminal
            print(f"Obstacle detected at position ({center_x}, {center_y}) with distance {w} pixels and confidence {confidence}")

    return frame, obstacle_info

if __name__ == "__main__":
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from video capture
        ret, frame = picam2.capture_array()   #cap.read()

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect obstacles in frame
        processed_frame, obstacle_info = detect_obstacle(frame)

        # Display frame with detected obstacles
        cv2.imshow('Obstacle Detection', processed_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
