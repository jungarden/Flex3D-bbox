import cv2
import os
import numpy as np
import random

# Function for image augmentation with horizontal distortion
def augment_image_and_label(image, keypoints):
    h, w = image.shape[:2]

    # 1. Rotation
    angle = random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    keypoints = [(np.dot(M, np.array([kp[0] * w, kp[1] * h, 1])) / [w, h])[:2] for kp in keypoints]

    # 2. Translation
    tx, ty = random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h))
    keypoints = [(kp[0] + tx / w, kp[1] + ty / h) for kp in keypoints]

    # 3. Scaling
    scale = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    image = cv2.warpAffine(image, M, (w, h))
    keypoints = [(np.dot(M, np.array([kp[0] * w, kp[1] * h, 1])) / [w, h])[:2] for kp in keypoints]

    # 4. Reflection
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        keypoints = [(1 - kp[0], kp[1]) for kp in keypoints]

    return image, keypoints

# Set base directory and paths for images and labels
base_dir = 'path/to/base_directory/'  # Replace with your base directory path
image_folder = os.path.join(base_dir, 'Images/')
label_folder = os.path.join(base_dir, 'labels/')
augmented_image_folder = os.path.join(base_dir, 'augmented_images/')
augmented_label_folder = os.path.join(base_dir, 'augmented_labels/')

if not os.path.exists(augmented_image_folder):
    os.makedirs(augmented_image_folder)

if not os.path.exists(augmented_label_folder):
    os.makedirs(augmented_label_folder)

# Perform augmentation on each image
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Read image
        image = cv2.imread(image_path)

        # Read label
        with open(label_path, 'r') as file:
            data = file.readline().strip().split(' ')
            class_id = data[0]
            keypoints = [(float(data[i]), float(data[i + 1])) for i in range(1, len(data), 2)]

        # Augment 10 times
        for i in range(10):
            # Apply augmentation techniques and horizontal distortion combination
            augmented_image, augmented_keypoints = augment_image_and_label(image, keypoints)

            # Add unique identifier to augmented file name
            base_name, ext = os.path.splitext(image_file)
            augmented_image_name = f'{base_name}_aug_{i+1}{ext}'
            augmented_label_name = f'{base_name}_aug_{i+1}.txt'

            # Save augmented image
            cv2.imwrite(os.path.join(augmented_image_folder, augmented_image_name), augmented_image)

            # Save augmented label
            with open(os.path.join(augmented_label_folder, augmented_label_name), 'w') as file:
                keypoints_flat = [str(coord) for kp in augmented_keypoints for coord in kp]
                file.write(f'{class_id} ' + ' '.join(keypoints_flat) + '\n')
