import cv2
import torch
import torchvision.transforms as T
import numpy as np

# Load images
# Load four images
image1 = cv2.imread('/path/to/image1')
image2 = cv2.imread('/path/to/image2')
image3 = cv2.imread('/path/to/image3')
image4 = cv2.imread('/path/to/image4')

# Convert OpenCV BGR image to PIL RGB image
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

# Define transformations for image augmentation
transform = T.Compose([
    T.ToPILImage(),
    T.RandomCrop((640,840)),
    T.ToTensor(),
    T.ToPILImage()
])

# Apply augmentations
augmented1 = transform(image1)
augmented2 = transform(image2)
augmented3 = transform(image3)
augmented4 = transform(image4)

# Convert images back to numpy arrays
augmented1 = np.array(augmented1).astype(np.uint8)
augmented2 = np.array(augmented2).astype(np.uint8)
augmented3 = np.array(augmented3).astype(np.uint8)
augmented4 = np.array(augmented4).astype(np.uint8)

# Concatenate images horizontally
top_row = np.hstack((augmented1, augmented2))
bottom_row = np.hstack((augmented3, augmented4))

# Concatenate images vertically
combined_image = np.vstack((top_row, bottom_row))

# Save and display the result
cv2.imwrite('torchvision_combined_image.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
