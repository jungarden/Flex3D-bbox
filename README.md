##  3d bounding box object detection(3dbbox) 

- Utilizing various datasets (parcel3d, AIHUB, other custom datasets...)
- Omitting reprojection process
- Generating inference code
- Adding multi-object inference
- Introducing anchors

### 1. Download the repository including the necessary dataset

### 2. System environment

* Python 3.6
* CUDA 11.1 Cudnn 8

* pytorch

      pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

 
* OpenCV

      pip install opencv-python (*pip install opencv-contrib-python==4.1.0.25로 설치)

      
* Scipy

      pip install scipy==1.2.0
      
* Pillow 

      pip install pillow==8.2.0

* tqdm
      pip install tqdm==4.64.1

### 3. training (multi)
      python3 train.py 
      --datacfg data/occlusion.data 
      --modelcfg cfg/yolo-pose-multi.cfg 
      --initweightfile cfg/darknet19_448.conv.23
      --pretrain_num_epochs 15


### 4. training (finetuning)
      python3 train.py 
      --datacfg data/trainbox.data
      --modelcfg cfg/yolo-pose.cfg 
      --initweightfile backup/*custom_dataset*/model.weights
      --pretrain_num_epochs 5


      
### 5. inference
      python3 img_inference.py 
      --datacfg data/occlusion.data 
      --modelcfg cfg/yolo-pose-multi.cfg 
      --initweightfile backup/parcel3d.weights
      --file video.mp4


### 6. results
![image](https://github.com/user-attachments/assets/80527fda-cfbc-41ae-b5b3-88779a124084)



### 7. reference
* Original src.: https://github.com/microsoft/singleshotpose
* other src : https://github.com/DatathonInfo/MISOChallenge-3Dobject


--> train.py <br>
  309. 백그라운드 영상 경로 변경 <br>
**# bg_file_names = get_all_files('VOCdevkit/VOC2012/JPEGImages')** <br>
**bg_file_names = get_all_files('BG/JPEGImages')** <br>
