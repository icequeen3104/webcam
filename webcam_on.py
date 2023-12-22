import cv2 as cv
capture = cv.VideoCapture(0)               #captures your PC's webcam
capture.set(3,640)                         #sets the height of the image
capture.set(4,480)                         #sets the width of the image
capture.set(10,100)                        #for brightening the image
true = 1                                   
while true:
    istrue, frame = capture.read()         #this reads the captured video frame by frame
    cv.imshow('Video', frame)              #displays each frame of the video
    if cv.waitKey(0) & 0xff('d'):           #on pressing the letter "d" the webcam recording would stop immediately
        break
capture.release()                            #optional
