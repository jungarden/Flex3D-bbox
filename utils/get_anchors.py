import numpy as np
from sklearn.cluster import KMeans
import os
import json

def kmeans_anchors(boxes, k=2):
    """Perform k-means clustering to find k anchor boxes."""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(boxes)
    return kmeans.cluster_centers_

def extract_width_height_from_json(json_file_path):
    """Extract width and height from a given JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        widths = []
        heights = []
        if "labelingInfo" in data:
            for item in data["labelingInfo"]:
                if "3DBox" in item:
                    box = item["3DBox"]["location"][0]
                    widths.append(int(box["x-range"]))
                    heights.append(int(box["y-range"]))
        return widths, heights

def calculate_kmeans_anchors(json_dir,image_width, image_height, num_anchors=3):
    """Calculate anchor boxes using k-means clustering and scale them to the new image size."""
    boxes = []

    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_file_path = os.path.join(json_dir, json_file)
            widths, heights = extract_width_height_from_json(json_file_path)
            for w, h in zip(widths, heights):
                boxes.append([w, h])

    # Perform k-means clustering
    boxes = np.array(boxes)
    anchors = kmeans_anchors(boxes, k=num_anchors)

    # Convert to 13x13 feature map size
    scale_w = image_width / 13.0
    scale_h = image_height / 13.0

    anchors[:, 0] /= scale_w
    anchors[:, 1] /= scale_h

    return anchors

# example
base_dir = "../dataset/glove00"
json_dir = os.path.join(base_dir,'json')
image_width = 1280
image_height = 720
num_anchors = 3 

anchors = calculate_kmeans_anchors(json_dir, image_width, image_height, num_anchors)
print(f"Calculated Anchors: {anchors}")