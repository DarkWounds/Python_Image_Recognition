import cv2
import random

img = cv2.imread('assets/puiu.png')

if img is None:
    print("Dosne't exist")
    exit(0)

tag = img[300:500, 500:700] # copy a square from this and paste in img on other rows, must be same height and width
img[100:300, 350:550] = tag

"""
ORIGINAL IMAGE

             500        700
              ↓          ↓
row 300   ┌───────────────┐
          │               │
          │   COPY THIS   │
          │               │
row 500   └───────────────┘
               │
               │
               │ copy
               ↓
row 100   ┌───────────────┐
          │               │
          │   PUT IT HERE │
          │               │
row 300   └───────────────┘
          350            550

"""

print(img)
print(type(img))
print(img.shape) # height, width, channels
print(img[30][45:400]) # between 45 and 400 pixels, BGR

# modify the first 100 lines with random pixels
for i in range(100):
    for j in range(img.shape[1]): # rows, collums, channels
        img[i][j] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

cv2.imshow('puiu_image.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()