import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera)

# Load pre-trained model for obstacle detection (e.g., using HOG or Haar cascades)
# You can use a pre-trained model or train your own depending on your requirements

# Initialize object detector (example: HOGDescriptor for pedestrian detection)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to detect obstacles and display information
def detect_obstacles(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect obstacles in the frame
    obstacles, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)

    # Display information about detected obstacles
    for (x, y, w, h) in obstacles:
        # Calculate obstacle's distance (example: based on size or depth estimation)
        obstacle_distance = w  # Example: Distance based on width of the bounding box

        # Display obstacle's position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        obstacle_position = (x + w // 2, y + h // 2)
        cv2.circle(frame, obstacle_position, 5, (0, 255, 0), -1)

        # Display confidence level (for demonstration, you can set it to a constant value)
        confidence_level = 0.8  # Example: Set confidence level to 0.8

        # Display information on the terminal
        print("Obstacle Detected - Distance:", obstacle_distance, "Position:", obstacle_position, "Confidence:", confidence_level)

    # Display frame with detected obstacles
    cv2.imshow("Obstacle Detection", frame)
    cv2.waitKey(1)

# Capture frames from PiCamera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Detect obstacles in the current frame
    detect_obstacles(image)

    # Clear the stream for the next frame
    rawCapture.truncate(0)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release PiCamera and close OpenCV windows
camera.close()
cv2.destroyAllWindows()
