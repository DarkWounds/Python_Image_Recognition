import cv2
import numpy as np

"""
Shi-Tomasi Corner detector & Good feature to track
"""

img = cv2.imread("assets/chess.jpg", 1)


if img is None:
    print("Nu a fost gasita")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 50)
# Source img, number of corners, minimum quality, minimum euclidian distance
"""
(x1, y1), (x2, y2) = corners[0][0]
sqrt((x2 - x1)^2 + (y2 - y1)^2) distance between two points
"""
corners = np.int_(corners) # Or int0(corners)

for corner in corners:
    x, y = corner.ravel() # Flattens, gives the elements from the: [[x, y]] -> [x, y]
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1) #imgDrawing

for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x) ,np.random.randint(0, 255, size=3)))
        # Generate a random RGB color tuple with 3 integer values (Red, Green, Blue)
        cv2.line(img, corner1, corner2, color, 1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()