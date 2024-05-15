import cv2
import numpy as np

def detect_red_object(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    return mask, res

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_rotation(contour):
    # Fit a bounding rectangle to the contour
    rect = cv2.minAreaRect(contour)
    _, _, angle = rect

    return angle

def main():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect red object
        mask, res = detect_red_object(frame)

        # Find contours
        contours = find_contours(mask)

        # Draw contours
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)

        # Find rotation of detected object
        for contour in contours:
            angle = find_rotation(contour)
            print("Rotation Angle:", angle)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
