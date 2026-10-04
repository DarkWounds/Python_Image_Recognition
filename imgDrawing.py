import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # nr of the device if you have multiple camera, take hold of the camera

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))  # the number represents a property

    # Drawing:
    img = cv2.line(frame, (0, 0), (width, height), (255, 0 ,0), 10) # frame,point1 to point2, color BGR, thickness
    cv2.line(img, (0, height), (width, 0), (0, 0 ,255), 5)

    img = cv2.rotate(img, cv2.ROTATE_180) #rotate img (camera is upside down)

    # Drawing after rotation so it's easier
    cv2.rectangle(img, (80, 80), (200, 200), (255, 0, 255), 3)
    cv2.circle(img, (100, 100), 30, (255, 0, 255), -1) #image, the center point, radius, color, fill or no fill
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'Testing text', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    """
    base image, the text, center position, the font, scale of the font (how much you magnify it),
    the color, the thickness, line type
    """

    """
    Representation of axes, coordonates in images
    (0,0) ─────────────→ X
  │
  │
  │
  ↓
  Y
    """

    cv2.imshow('rotatedImg', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()  # release the camera import
cv2.destroyAllWindows()