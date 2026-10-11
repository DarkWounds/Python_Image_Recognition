import numpy as np
import cv2

img = cv2.imread("assets/football.jpg", 0) # 0 -> greyscale
template = cv2.imread("assets/ball.jpg", 0)
h, w = template.shape

if template is None:
   print("No template found")
   exit()

if img is None:
    print("No image found")
    exit()

#print(img)

"""
OpenCV template matching methods:
 TM_CCOEFF: Matches based on correlation after subtracting the mean.
 TM_CCOEFF_NORMED: Normalized correlation coefficient (scores range from -1 to 1).
 TM_CCORR: Matches based on cross-correlation between image and template.
 TM_CCORR_NORMED: Normalized cross-correlation (scores range from 0 to 1).
 TM_SQDIFF: Matches based on the sum of squared differences (lower is better).
 TM_SQDIFF_NORMED: Normalized squared differences (lower is better).

 For CCOEFF and CCORR methods, higher scores indicate better matches.
 For SQDIFF methods, lower scores indicate better matches.
"""

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img,template,method)
    # cv2.matchTemplate() searches for a smaller image (template) inside a larger image (img), comparing them at different positions.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else :
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    cv2.imshow("img", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
