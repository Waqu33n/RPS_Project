import cv2
import os
import numpy as np

# Input and output directories
input_dir = "training_set"  # Replace with your dataset folder
output_dir = "augmented_images"
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to rotate an image and keep its original size with mirror padding
def rotate_image_with_padding(image, angle):
    (h, w) = image.shape[:2]  # Original dimensions
    diagonal = int(np.sqrt(h**2 + w**2))  # Compute the diagonal

    # Create a larger canvas to avoid cropping
    expanded_canvas = cv2.copyMakeBorder(image, 
                                         (diagonal - h) // 2, (diagonal - h) // 2, 
                                         (diagonal - w) // 2, (diagonal - w) // 2, 
                                         cv2.BORDER_REFLECT)

    # Compute new center
    center = (expanded_canvas.shape[1] // 2, expanded_canvas.shape[0] // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    rotated = cv2.warpAffine(expanded_canvas, M, (expanded_canvas.shape[1], expanded_canvas.shape[0]))

    # Crop the center to match the original size
    x_start = (rotated.shape[1] - w) // 2
    y_start = (rotated.shape[0] - h) // 2
    cropped = rotated[y_start:y_start + h, x_start:x_start + w]

    return cropped

# Process each image
for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)  # Load image
    
    if img is None:
        print(f"Failed to load {img_name}")
        continue

    # Rotate and save 8 times (every 45°)
    for angle in range(0, 360, 45):  # 0°, 45°, 90°, ... 315°
        rotated_img = rotate_image_with_padding(img, angle)
        
        # Save with a new filename
        new_filename = f"{os.path.splitext(img_name)[0]}_rot{angle}.jpg"
        cv2.imwrite(os.path.join(output_dir, new_filename), rotated_img)

print("Augmentation complete! Rotated images saved in:", os.path.abspath(output_dir))
