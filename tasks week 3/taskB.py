import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Read image ---
image_path = r"D:\computer vision tasks\tasks week 3\kid.jpg"  # Change this
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # work in grayscale

if image is None:
    raise ValueError("Image not found!")

# --- Step 2: Calculate histogram ---
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
hist = hist.ravel()  # flatten

# Show histogram
plt.plot(hist, color='black')
plt.title("Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.show()

# --- Step 3: Analyze histogram ---
mean_intensity = np.mean(image)
std_intensity = np.std(image)  # spread of intensities

# Rules
if mean_intensity < 60:  
    img_condition = "Over Dark"
elif mean_intensity > 190:  
    img_condition = "Over Bright"
elif std_intensity < 40:  
    img_condition = "Low Contrast"
else:
    img_condition = "Normal"

print(f"Image Condition: {img_condition}")

# --- Step 4: Show image ---
plt.imshow(image, cmap="gray")
plt.title(f"Detected Condition: {img_condition}")
plt.axis("off")
plt.show()
