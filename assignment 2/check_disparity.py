import cv2
import urllib.request
import numpy as np

B = 0.10          # Baseline (meters)
f = 2912          # Focal length in pixels (iPhone 13 Pro Max)
Z_ground = 0.50   # Ground truth nose depth (meters)

left_image_path = "left.jpg"
right_image_path = "right.jpg"


left = cv2.imread(left_image_path)
right = cv2.imread(right_image_path)

if left is None or right is None:
    print("Error: Could not load images. Check the file paths!")
    exit()

points = {"left": None, "right": None}

def click_left(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points["left"] = (x, y)
        print("Left nose point:", points["left"])

def click_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points["right"] = (x, y)
        print("Right nose point:", points["right"])

cv2.namedWindow("LEFT")
cv2.setMouseCallback("LEFT", click_left)

cv2.namedWindow("RIGHT")
cv2.setMouseCallback("RIGHT", click_right)

print("Click nose on LEFT image, then RIGHT image. Press 'q' to calculate.")

while True:
    ldisp = left.copy()
    rdisp = right.copy()

    if points["left"]:
        cv2.circle(ldisp, points["left"], 5, (0,0,255), -1)

    if points["right"]:
        cv2.circle(rdisp, points["right"], 5, (0,0,255), -1)

    cv2.imshow("LEFT", ldisp)
    cv2.imshow("RIGHT", rdisp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

xl = points["left"][0]
xr = points["right"][0]
d = abs(xl - xr)

Z1 = (B * f) / d

print("\n===============================")
print("DISPARITY d =", d, "pixels")
print("Z1 (Stereo Depth) =", round(Z1, 4), "meters")
print("Error1 =", round(abs(Z_ground - Z1) / Z_ground * 100, 2), "%")
print("===============================\n")



print("Downloading MiDaS model ...")
midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas_model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Use left image for ML depth
img = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
input_batch = transform(img).unsqueeze(0)

with torch.no_grad():
    prediction = midas_model(input_batch)
    depth_map = prediction.squeeze().cpu().numpy()

# Normalize depth map for display
depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_norm = (depth_norm * 255).astype("uint8")
cv2.imwrite("depth_map_midas.png", depth_norm)

# Nose ML depth value
xn, yn = points["left"]
depth_rel = depth_map[yn, xn]

# Scale relative depth to metric using ground truth
scale = Z_ground / depth_rel
Z2 = depth_rel * scale

print("Z2 (ML Depth) =", round(Z2, 4), "meters")
print("Error2 =", round(abs(Z_ground - Z2) / Z_ground * 100, 2), "%")
print("Depth map saved as depth_map_midas.png")
print("===============================\n")
