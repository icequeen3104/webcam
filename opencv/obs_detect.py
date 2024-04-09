import cv2
import numpy as np
import time
from picamera2 import Picamera2
picam2 = Picamer2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640,480)}))
picam2.start()

def detect_obstacles(frame):
    # Function to detect obstacles and return bounding boxes

    # Placeholder function, replace with your actual obstacle detection algorithm
    # This could involve any number of computer vision techniques such as contour detection, object detection, etc.
    # For demonstration purposes, let's assume we're just detecting a single obstacle here
    obstacle_detected = False
    obstacle_bbox = None

    # Example: detect an object of any color (just for demonstration)
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the color of obstacles
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])

    # Create a mask using the color bounds
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, assume an obstacle is detected
    if contours:
        obstacle_detected = True
        # Get the bounding box of the first contour
        obstacle_bbox = cv2.boundingRect(contours[0])

    return obstacle_detected, obstacle_bbox

def calculate_fps(prev_time, curr_time):
    # Function to calculate frames per second (FPS)
    return 1.0 / (curr_time - prev_time)

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = picam2.capture_array()
        if not ret:
            break

        # Detect obstacles
        obstacle_detected, obstacle_bbox = detect_obstacles(frame)

        # Display bounding boxes
        if obstacle_detected:
            x, y, w, h = obstacle_bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = calculate_fps(prev_time, curr_time)
        prev_time = curr_time

        # Display FPS on frame
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display direction (dummy example, replace with your logic)
        if obstacle_detected:
            print("Obstacle detected! Move in a different direction.") # Output to terminal
            cv2.putText(frame, "Move in a different direction", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
