import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Read image ---
image_path = r"D:\computer vision tasks\tasks week 3\kid.jpg"  # Change this to your image path
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    raise ValueError("Image not found. Check the path!")

# Show original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

# --- Step 2: Detect image type ---
if len(image.shape) == 2:
    # Already single channel (Gray or Binary)
    unique_vals = np.unique(image)
    if set(unique_vals).issubset({0, 255}):
        img_type = "Binary"
    else:
        img_type = "Grayscale"

elif len(image.shape) == 3 and image.shape[2] == 3:
    # 3 channels → RGB or actually Gray disguised as RGB
    b, g, r = cv2.split(image)
    if np.array_equal(b, g) and np.array_equal(g, r):
        unique_vals = np.unique(b)
        if set(unique_vals).issubset({0, 255}):
            img_type = "Binary"
        else:
            img_type = "Grayscale"
    else:
        img_type = "RGB"
else:
    img_type = "Unknown"

print(f"Detected Image Type: {img_type}")

# --- Step 3: Show analysis results ---
if img_type == "Binary":
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    plt.imshow(gray, cmap="gray")
    plt.title("Binary Image")
    plt.axis("off")
    plt.show()

elif img_type == "Grayscale":
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    plt.show()

elif img_type == "RGB":
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("RGB Image")
    plt.axis("off")
    plt.show()
