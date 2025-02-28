import cv2
import os
import numpy as np
import random

# Input and output directories
input_dir = "augmented_images"  # Replace with your dataset folder
train_dir = "training_set"
test_dir = "testing_set"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

test_number = 0

# Get all image files
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Process each image
for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)  # Load image
    
    if img is None:
        print(f"Failed to load {img_name}")
        continue
    if (random.random() > .833 and test_number < 100):
        cv2.imwrite(os.path.join(test_dir, img_name), img)
        test_number += 1
    else:
        cv2.imwrite(os.path.join(train_dir, img_name), img)


print("Augmentation complete! Rotated images saved in:", os.path.abspath(test_dir))
