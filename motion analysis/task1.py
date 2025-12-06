import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

# ------------------------------
# Helper: resize image before display
# ------------------------------
def show(title, img, scale=2):   # scale=2 makes it double size
    h, w = img.shape[:2]
    resized = cv2.resize(img, (w*scale, h*scale))
    cv2.imshow(title, resized)
    cv2.waitKey(0)

trafficObj = cv2.VideoCapture(r"D:\computer vision tasks\motion analysis\traffic.avi")

fn = 73  # Set frame position
trafficObj.set(cv2.CAP_PROP_POS_FRAMES, fn)

ret, frame = trafficObj.read()
if not ret:
    print("Could not read frame")
    exit()

# Show frame
show("Frame", frame)

fGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
show("Gray Frame", fGray)

# Multi-threshold
t = threshold_multiotsu(fGray, classes=4)
print("Threshold values:", t)
print(t[2])

# Binary mask
fBW = np.where(fGray > t[2], 255, 0).astype(np.uint8)
show("Binary Mask", fBW)

# Structuring elements
se5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
se10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

# Morphology
fNoNoise = cv2.morphologyEx(fBW, cv2.MORPH_OPEN, se5)
show("After Opening", fNoNoise)

f1Obj = cv2.morphologyEx(fNoNoise, cv2.MORPH_CLOSE, se10)
show("After Closing", f1Obj)

# Tagging
taggedCars = frame.copy()
width = 2

if np.any(f1Obj == 255):
    r, c = np.where(f1Obj == 255)
    rbar = int(np.mean(r))
    cbar = int(np.mean(c))

    row = slice(rbar - width, rbar + width + 1)
    col = slice(cbar - width, cbar + width + 1)

    taggedCars[row, col] = [0, 0, 255]  # Red marking

show("Tagged Frame", taggedCars)

cv2.destroyAllWindows()
