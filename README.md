---

# Flexible 3D Object Detection          (Flex3D-bbox)

### Introduction

Building on the work by Bugra Tekin, Sudipta N. Sinha, and Pascal Fua in "Real-Time Seamless Single Shot 6D Object Pose Prediction" (CVPR 2018), this implementation enhances the original framework to better address everyday object detection tasks.   [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tekin_Real-Time_Seamless_Single_CVPR_2018_paper.pdf)

We introduce fine-tuning across diverse datasets, from custom-labeled data to standard benchmarks, with seamless parsing of both into a unified labeling format. The system supports multiple input types, including images and videos, and is optimized for robust multi-object and multi-class inference. These improvements make the method highly adaptable and effective for a wide range of real-world applications.



![video2](https://github.com/user-attachments/assets/bf39fea3-3afb-4b3a-aca9-0a6a4da62ab0)

### Key Features

- **Utilizing Various Datasets**: Includes `parcel3d`, `AIHUB`, and other "manually labeled" custom datasets.
- **Omitting Reprojection Process**: Streamlined pipeline by removing unnecessary reprojection.
- **Generating Inference Code**: Easy-to-use inference code generation.
- **Adding Multi-Object Inference**: Enhanced capabilities for detecting multiple objects simultaneously.
- **Introducing Anchors**: Improved detection accuracy through the use of anchors.

---

### 1. Download the Repository

Download the repository including the necessary datasets:

```sh
git clone https://github.com/jungarden/Flex3D-bbox.git
```

### 2. System Environment

Ensure your environment meets the following requirements:

- **Python**: 3.6
- **CUDA**: 11.1
- **Cudnn**: 8
- **Docker**: Image: `nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04`

Install the required libraries as follows:

- **PyTorch**:

    ```sh
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

- **OpenCV**:

    ```sh
    pip install opencv-python
    # Alternatively, install a specific version:
    # pip install opencv-contrib-python==4.1.0.25
    ```

- **Scipy**:

    ```sh
    pip install scipy==1.2.0
    ```

- **Pillow**:

    ```sh
    pip install pillow==8.2.0
    ```

- **tqdm**:

    ```sh
    pip install tqdm==4.64.1
    ```

### 3. Parsing

Before training, ensure your dataset labels are correctly formatted using the txt_labels.py script: This script parses and converts your dataset's labeling information into the format required for training. Make sure to select the appropriate labeling method for your dataset, whether it is manually labeled or follows the AIHUB dataset format.

```sh
python3 making_txt_labels.py
```
- **glove00 folder structure**

  
![image](https://github.com/user-attachments/assets/f66df63c-ecd4-43d1-9d33-7c676b6d6eb6)

- **glove00.data**

  
![image](https://github.com/user-attachments/assets/723c7cde-43e7-442a-8b76-dd37cba33793) 


### 4-1. Training (Multi-Object)

To train the model on multiple objects across datasets, use the following command:

```sh
python3 train_multi.py \
--datacfg data/occlusion.data \
--modelcfg
cfg/yolo-pose-multi.cfg \
--initweightfile cfg/darknet19_448.conv.23 \ 
--pretrain_num_epochs 15
```
* darknet19_448.cov.23 is not included in this repo.

### 4-2. Training (Finetuning)

For finetuning on a custom dataset, run:

```sh
python3 train.py \
--datacfg data/box3d.data \
--modelcfg cfg/yolo-pose.cfg \
--initweightfile backup/parcel3d/model.weights \
--pretrain_num_epochs 5
```

### 5. Inference

To perform inference on a video file, execute:

```sh
python3 inference.py \
--datacfg data/occlusion.data \
--modelcfg cfg/yolo-pose-multi.cfg \
--initweightfile backup_multi/model.weights \
--file video.mp4
```

### 6. Results

Below is an example of the detection results:
*multi classes
![image](https://github.com/user-attachments/assets/80527fda-cfbc-41ae-b5b3-88779a124084)
![video](https://github.com/user-attachments/assets/43aae97d-c3c4-428c-a886-c2a883a1bf1d)

### 7. References

- **Original Source**: [Microsoft SingleShotPose](https://github.com/microsoft/singleshotpose)
- **Other Source**: [MISOChallenge-3Dobject](https://github.com/DatathonInfo/MISOChallenge-3Dobject)

---

### Additional Information

#### System Architecture
- **Repository Structure**:
  - `baseline`: Single object detection
  - `multi`: Multi-object detection
  - `dataset`: Contains various datasets
  - `utils`: Contains utility functions (e.g get_anchors.py)

#### Code Modifications

- **train.py**: Removed internal parameters, rotation matrices, and reprojection variables.
- **utils.py**: Created `build_target_anchors` to consider anchors in single detection('baseline'), modified 'get_region_boxes' to consider anchors.
- **image.py** & **dataset.py**: Updated paths for custom datasets.
- **yolo-pose.cfg**: Adjusted the number of filters for anchors and classes.
- **inference.py**: Added visualization for bounding boxes and classes.
