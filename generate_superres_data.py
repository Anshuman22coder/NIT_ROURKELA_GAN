import cv2
import os

# Define paths
hr_folder = "data/superres/HR"
lr_folder = "data/superres/LR"

# Create directories if they don't exist
os.makedirs(hr_folder, exist_ok=True)
os.makedirs(lr_folder, exist_ok=True)

# Check if HR folder has images
if not os.listdir(hr_folder):
    print("⚠️ No images found in data/superres/HR. Please add high-resolution images first!")
    exit()

# Loop through each high-resolution image
for img_name in os.listdir(hr_folder):
    img_path = os.path.join(hr_folder, img_name)
    
    # Read the high-resolution image
    hr_image = cv2.imread(img_path)
    if hr_image is None:
        print(f"Skipping {img_name}: Unable to read image.")
        continue

    # Resize to 50% of original size (downscale)
    lr_image = cv2.resize(hr_image, (hr_image.shape[1] // 2, hr_image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    
    # Save the low-resolution image
    cv2.imwrite(os.path.join(lr_folder, img_name), lr_image)

print("✅ Low-resolution images saved in data/superres/LR")
