import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Read image ---
image_path = r"D:\computer vision tasks\tasks week 3\kid.jpg"  # Change this
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found!")

# --- Step 2: Histogram ---
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
hist = hist.ravel()

plt.plot(hist, color='black')
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.show()

# --- Step 3: Analyze brightness/contrast ---
mean_intensity = np.mean(image)
std_intensity = np.std(image)

if mean_intensity < 60:
    img_condition = "Over Dark"
elif mean_intensity > 190:
    img_condition = "Over Bright"
elif std_intensity < 40:
    img_condition = "Low Contrast"
else:
    img_condition = "Normal"

print(f"Detected Condition: {img_condition}")

# --- Step 4: Enhancement ---
enhanced = image.copy()

if img_condition == "Over Dark":
    # Brighten → Histogram Equalization
    enhanced = cv2.equalizeHist(image)

elif img_condition == "Over Bright":
    # Darken → Gamma Correction (<1 darkens, >1 brightens)
    gamma = 2.0  
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(image, table)

elif img_condition == "Low Contrast":
    # Contrast Enhancement → CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

else:
    enhanced = image  # No change if normal

# --- Step 5: Show before & after ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title(f"Original ({img_condition})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced, cmap="gray")
plt.title("Enhanced Image")
plt.axis("off")

plt.show()
