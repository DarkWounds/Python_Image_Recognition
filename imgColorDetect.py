import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # nr of the device if you have multiple camera, take hold of the camera

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))  # the number represents a property
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red = np.array([50, 20, 20])
    upper_red = np.array([150, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)

    """
    1 1 = 1
    0 1 = 0
    1 0 = 0
    0 0 = 0
    
    Takes two images, merges them together and returns the result, applying the mask
    """

    cv2.imshow('result', result2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()