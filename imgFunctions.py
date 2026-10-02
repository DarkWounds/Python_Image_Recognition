import cv2
import os

# img = cv2.imread('assets/puiu.png', cv2.IMREAD_UNCHANGED)  # Load an image from file
img = cv2.imread('assets/puiu.png', 1)  # Same thing as above

if img is None:
    print("Imaginea nu a fost găsită!")
    exit(0)
else:
    print("Imagine încărcată cu succes.")

img = cv2.resize(img, (1000, 1000))  # Resize the image to 400x400 pixels
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5) #
img = cv2.rotate(img, cv2.ROTATE_180)

cv2.imwrite('assets/new puiu.png', img)

# cv2.IMREAD_COLOR // Transparency  -1
# cv2.IMREAD_GRAYSCALE 0
# cv2.IMREAD_UNCHANGED 1

print(os.getcwd())
print(os.path.exists('assets/puiu.png'))

cv2.imshow('puiu_image.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()