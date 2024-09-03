import os
import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from darknet import Darknet
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

# Hyperparameters
im_width, im_height = 1920, 1080

def inference(modelcfg, weightfile, imagefile, save_path):
    # Parameters
    save = True
    gpus = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    test_width = model.test_width
    test_height = model.test_height
    num_keypoints = model.num_keypoints
    conf_thresh = model.conf_thresh
    num_labels = num_keypoints * 2 + 1  # +2 for width, height, +1 for class label
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

    # Record inference start time
    start_time = time.time()

    # Convert image to numpy array
    img = data[0, :, :, :].cpu().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))  # Move channels to the last dimension (H, W, C)

    # Forward pass
    output = model(data).data

    # Record inference end time
    inference_time = time.time() - start_time
    print(f"Image: {imagefile}, Inference Time: {inference_time:.4f} seconds")

    # Eliminate low-confidence predictions
    box_pr = get_region_boxes_conf(output, 1, num_keypoints, conf_thresh, only_objectness=1, validation=False)

    # If there are no objects above the threshold, save the image only
    if box_pr is None or len(box_pr) == 0:
        fig, ax = plt.subplots()
        ax.set_xlim((0, im_width))
        ax.set_ylim((0, im_height))
        ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))
        ax.invert_yaxis()
        ax.axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        plt.close(fig)
        return

    box_pr = np.array([tensor_item.detach().numpy() for tensor_item in box_pr])

    # Calculate 2D corner points
    corners2D_pr = np.array(np.reshape(box_pr[:16], [8, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width  # im_width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height  # im_height

    # Visualization
    fig, ax = plt.subplots()
    ax.set_xlim((0, im_width))
    ax.set_ylim((0, im_height))
    ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))

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

    ax.invert_yaxis()  # Invert y-axis to match image coordinate system
    ax.axis('off')  # Hide axes

    # Save the image only (remove borders)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

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
    test_width = model.test_width
    test_height = model.test_height
    num_keypoints = model.num_keypoints
    conf_thresh = model.conf_thresh
    num_labels = num_keypoints * 2 + 1  # +2 for width, height, +1 for class label
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
    box_pr = get_region_boxes_conf(output, 1, num_keypoints, conf_thresh, only_objectness=1, validation=False)
    if box_pr is None or len(box_pr) == 0:
        return

    box_pr = np.array([tensor_item.detach().numpy() for tensor_item in box_pr])

    # Calculate 2D corner points
    corners2D_pr = np.array(np.reshape(box_pr[:16], [8, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width  # im_width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height  # im_height

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
        pt = (int(corners2D_pr[i, 0]), int(corners2D_pr[i, 1]))
        cv2.circle(img_bgr, pt, 5, (0, 0, 255), -1)  # Red point
    return img_bgr

from tqdm import tqdm
import time

def video_inference(modelcfg, weightfile, videofile):
    # Read video file
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    # Initialize model (only performed once)
    model = initialize_model(modelcfg, weightfile)

    # Set up VideoWriter for saving output video
    output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0

    # Add progress bar using tqdm
    with tqdm(total=total_frames // 2) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only odd frames (15 frames)
            if frame_idx % 2 == 0:
                # Already in BGR format, use without conversion
                processed_frame = inference_from_image(model, frame)

                # If inference_from_image returns None, use the original image
                if processed_frame is None:
                    processed_frame = frame
                else:
                    # Match original resolution
                    if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                        processed_frame = cv2.resize(processed_frame, (width, height))

                # Add frame to video
                output_video.write(processed_frame)
                pbar.update(1)  # Update progress bar

            frame_idx += 1

    cap.release()
    output_video.release()

def webcam_inference(modelcfg, weightfile):
    cap = cv2.VideoCapture(0)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    model = initialize_model(modelcfg, weightfile)
    
    # VideoWriter for save webcam video (optional)
    # output_video = cv2.VideoWriter('output_webcam.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = inference_from_image(model, frame)
        
        if processed_frame is None:
            processed_frame = frame
        else:
            if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                processed_frame = cv2.resize(processed_frame, (width, height))

        # (optional)
        # output_video.write(processed_frame)
        cv2.imshow('Webcam Feed', processed_frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    # output_video.release()
    cv2.destroyAllWindows()

def process_folder(modelcfg, weightfile, folder):
    # Initialize model (only performed once)
    model = initialize_model(modelcfg, weightfile)

    # Process all image files in the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            save_path = os.path.join(folder, f'processed_{filename}')

            # Open image file and convert to PIL.Image object
            img = Image.open(file_path).convert('RGB')

            # Call inference_from_image function to process the image
            processed_img = inference_from_image(model, img)

            # Save the resulting image
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
            cv2.imwrite(save_path, processed_img)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SingleShotPose Inference')
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg')  # Network config
    parser.add_argument('--weightfile', type=str, default='backup/trainbox/model.weights')  # Pretrained weights
    parser.add_argument('--file', type=str, required=True, help='image or video file(image,folder,video,webcam(0)) for inference')
    args = parser.parse_args()

    # Select image, video(webcam), or folder based on file extension
    file_ext = os.path.splitext(args.file)[-1].lower()
    if os.path.isdir(args.file):
        process_folder(args.modelcfg, args.weightfile, args.file)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        inference(args.modelcfg, args.weightfile, args.file, 'output_image.png')
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_inference(args.modelcfg, args.weightfile, args.file)
    elif args.file.lower() == '0':
        webcam_inference(args.modelcfg, args.weightfile)
    else:
        print("Unsupported file format. Please use an image or video/webcam.")