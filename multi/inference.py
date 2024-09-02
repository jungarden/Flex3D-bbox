from __future__ import print_function
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import argparse
import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from darknet_multi import Darknet
from utils_multi import *    
from cfg import parse_cfg
from region_loss_multi import RegionLoss
import dataset_multi
import torchvision.ops as ops

# Hyperparameters
im_width , im_height = 1280,720

save = True
gpus = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
torch.cuda.manual_seed(int(time.time()))

def inference(modelcfg, weightfile, imagefile, save_path):
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    test_width = model.width
    test_height = model.height

    num_keypoints = model.num_keypoints
    conf_thresh = model.conf_thresh
    num_labels = num_keypoints * 2 + 1  # +2 for width, height, +1 for class label
    num_classes = model.num_classes
    anchors = model.anchors
    num_anchors = model.num_anchors

    # Read image
    img = Image.open(imagefile).convert('RGB')
    img_resized = img.resize((test_width, test_height))
    data = transforms.ToTensor()(img_resized).unsqueeze(0)

    # Pass data to GPU
    data = data.cuda()
    with torch.no_grad():
        data = data

    # Convert image to numpy array
    img = data[0, :, :, :].cpu().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))  # Move channels to last dimension (H, W, C)

    # Forward pass
    output = model(data).data

    all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, only_objectness=0)

    # Prepare for visualization
    fig, ax = plt.subplots(figsize=(im_width / 100, im_height / 100))
    ax.set_xlim((0, im_width))
    ax.set_ylim((0, im_height))
    ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))

    # If no predicted boxes, save the image only
    if all(all_box is None for all_box in all_boxes):
        ax.set_xlim((0, im_width))
        ax.set_ylim((0, im_height))
        ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))
        ax.invert_yaxis()
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    # List of class names (depends on the model)
    class_names = ['glove', 'shoes', 'tin', 'box']  # Example class names

    for c in range(num_classes):
        best_conf_est = -sys.maxsize
        box_pr = None
        for j in range(len(all_boxes)):
            boxes = all_boxes[j]  # Boxes for each image
            for box in boxes:
                if (box[2 * num_keypoints] > best_conf_est) and (int(box[-1]) == c):
                    best_conf_est = box[2 * num_keypoints]
                    box_pr = box
                    print(box_pr[16:])

            if box_pr is not None:
                print('box_pr : ')
                print(box_pr[16:])
                corners2D_pr = np.array(np.reshape(box_pr[:2 * num_keypoints], [-1, 2]), dtype='float32')
                # Convert coordinates from model input size to actual image size
                corners2D_pr[:, 0] *= im_width
                corners2D_pr[:, 1] *= im_height

                edges_corners = [
                    [0, 2], [2, 1], [1, 3], [3, 0],  # Lower rectangle (front)
                    [4, 6], [6, 5], [5, 7], [7, 4],  # Upper rectangle (back)
                    [2, 6], [0, 4], [3, 7], [1, 5]   # Edges connecting lower and upper rectangles
                ]

                # Draw lines
                for edge in edges_corners:
                    ax.plot(corners2D_pr[edge, 0], corners2D_pr[edge, 1], color='b', linewidth=1.5)

                # Draw points - only use the first 8 corners
                ax.scatter(corners2D_pr[:, 0], corners2D_pr[:, 1], color='r', s=15)  # Point size can be adjusted with 's'

                # Calculate label position based on the top-most corner of the box
                label_x = corners2D_pr[:, 0].min()
                label_y = corners2D_pr[:, 1].min() - 10  # Adjust to position the label above the box

                # Get predicted class name
                predicted_class_name = class_names[c]  # Class name mapping

                # Display class name above the 3D bounding box
                ax.text(label_x, label_y, predicted_class_name, color='white', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))

    # Invert y-axis to match image coordinate system
    ax.invert_yaxis()
    ax.axis('off')  # Hide axes

    # Save the image only (remove borders)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)

    plt.close(fig)  


def initialize_model(modelcfg, weightfile):
    # Initialize the model and load weights
    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    return model

def inference_from_image(model, img_bgr):
    # Image is already provided in BGR format
    test_width = model.width
    test_height = model.height
    num_keypoints = model.num_keypoints
    conf_thresh = model.conf_thresh
    num_labels = num_keypoints * 2 + 1  # +2 for width, height, +1 for class label
    num_classes = model.num_classes
    anchors = model.anchors
    num_anchors = model.num_anchors

    # Directly use BGR image, convert to Tensor without PIL conversion
    img_resized = cv2.resize(img_bgr, (test_width, test_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    data = transforms.ToTensor()(img_rgb).unsqueeze(0)

    # Pass data to GPU
    data = data.cuda()
    with torch.no_grad():
        output = model(data).data

    # Eliminate low-confidence predictions
    box_pr = get_region_boxes(output, 1, num_keypoints, conf_thresh)
    if box_pr is None or len(box_pr) == 0:
        return  

    box_pr = np.array([tensor_item.detach().numpy() for tensor_item in box_pr])
    
    # Calculate 2D corner points
    corners2D_pr = np.array(np.reshape(box_pr[:16], [8, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * img_bgr.shape[1]  # im_width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * img_bgr.shape[0]  # im_height

    # Draw lines
    edges_corners = [
        [0, 2], [2, 1], [1, 3], [3, 0],  # Lower rectangle (front)
        [4, 6], [6, 5], [5, 7], [7, 4],  # Upper rectangle (back)
        [2, 6], [0, 4], [3, 7], [1, 5]   # Edges connecting lower and upper rectangles
    ]
    
    for edge in edges_corners:
        pt1 = (int(corners2D_pr[edge[0], 0]), int(corners2D_pr[edge[0], 1]))
        pt2 = (int(corners2D_pr[edge[1], 0]), int(corners2D_pr[edge[1], 1]))
        cv2.line(img_bgr, pt1, pt2, (255, 0, 0), 2)  # Blue line

    # Draw points - only use the first 8 corners
    for i in range(8):
        pt = (int(corners2D_pr[i, 0]), int(corners
