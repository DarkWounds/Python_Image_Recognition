import cv2
import numpy as np

cap = cv2.VideoCapture(0) # nr of the device if you have multiple camera, take hold of the camera

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4)) # the number represents a property

    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # half od width and half of the height

    image[:height // 2, :width // 2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # top left
    image[height // 2:, :width // 2] = cv2.rotate(smaller_frame, cv2.ROTATE_180) # bottom left
    image[:height // 2, width // 2:] = smaller_frame # top right
    image[height // 2:, width // 2:] = smaller_frame # bottom right\

    # image[height // 2:, width // 2:] = cv2.rotate(smaller_frame, cv2.ROTATE_90_CLOCKWISE)
    # can't do this because the height and width don't match and rotating it swaps the height and width

    """
    
                 :height//2       height//2:
                  ↓                 ↓
           ┌─────────────┬─────────────┐
     :     │      1      │      2      │
           ├─────────────┼─────────────┤
     :     │      3      │      4      │
           └─────────────┴─────────────┘
                  ↑                 ↑
               :width//2        width//2:
    
    """

    # cv2.imshow('frame', frame)
    cv2.imshow('img', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() # release the camera import
cv2.destroyAllWindows()