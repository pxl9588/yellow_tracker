from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
yellowLower = (31, 0, 147)
yellowUpper = (98, 141, 255)

redLower = (124, 89, 37)
redUpper = (194, 203, 173)

vs = VideoStream(src=0).start()
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color yellow nad red
    yellowMask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yellowMask = cv2.erode(yellowMask, None, iterations=2)
    yellowMask = cv2.dilate(yellowMask, None, iterations=2)
    
    redMask = cv2.inRange(hsv, redLower, redUpper)
    redMask = cv2.erode(redMask, None, iterations=2)
    redMask = cv2.dilate(redMask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    yellowCounts = cv2.findContours(yellowMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redCounts = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    yellowCounts = yellowCounts[0] if imutils.is_cv2() else yellowCounts[1]
    redCounts = redCounts[0] if imutils.is_cv2() else redCounts[1]

    yellowCenter = None
    redCenter = None

    yellowX = deque(maxlen=32)
    yellowY = deque(maxlen=32)
    yellowRadius = deque(maxlen=32)
    yellowPts = deque(maxlen=32)

    # only proceed if at least one contour was found
    if len(yellowCounts) > 0:

        for i in range(0, len(yellowCounts)):
            ((x, y), radius) = cv2.minEnclosingCircle(yellowCounts[i])
            if 20 < radius < 50:
                yellowX.append(x)
                yellowY.append(y)
                yellowRadius.append(radius)
                M = cv2.moments(yellowCounts[i])
                yellowPts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        # only proceed if the radius meets a minimum size
        for i in range(0, len(yellowPts)):
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(yellowX[i]), int(yellowY[i])), int(yellowRadius[i]), (0, 255, 255), 2)
            cv2.circle(frame, yellowPts[i], 5, (0, 255, 255), -1)

    redX = deque(maxlen=32)
    redY = deque(maxlen=32)
    redRadius = deque(maxlen=32)
    redPts = deque(maxlen=32)

    if len(redCounts) > 0:

        for i in range(0, len(redCounts)):
            ((x, y), radius) = cv2.minEnclosingCircle(redCounts[i])
            if 20 < radius < 50:
                redX.append(x)
                redY.append(y)
                redRadius.append(radius)
                M = cv2.moments(redCounts[i])
                redPts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        # only proceed if the radius meets a minimum size
        for i in range(0, len(redPts)):
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(redX[i]), int(redY[i])), int(redRadius[i]), (0, 0, 255), 2)
            cv2.circle(frame, redPts[i], 5, (0, 0, 255), -1)

    for i in range(1, len(redPts)):
        cv2.line(frame, redPts[i - 1], redPts[i], (0, 0, 255), 5)

    for i in range(1, len(yellowPts)):
        cv2.line(frame, yellowPts[i - 1], yellowPts[i], (0, 255, 255), 5)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
vs.stop()

# close all windows
cv2.destroyAllWindows()