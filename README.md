![image](https://github.com/user-attachments/assets/2a644774-3afe-468a-846c-729346ef9237)##  3d bounding box object detection(3dbbox) 

### 1. 다운로드 및 코드 경로에서 압축 해제

### 2. 시스템 환경

## Python 3.6
## CUDA 11.1 Cudnn 8

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

### 3. 학습된 모델 테스트 (ape 예시)

* 공개된 학습 모델 테스트
      
      python valid.py --datacfg data/ape.data --modelcfg cfg/yolo-pose.cfg --weightfile data/ape/model/model_backup.weights
      
* 큐브 시각화 
      
      python visualize.py --datacfg data/ape.data --modelcfg cfg/yolo-pose.cfg --weightfile data/ape/model/model_backup.weights

****

### 4. LINEMOD 학습 (ape 예시)

* 학습할 카테고리별로 실행
 
      python train.py --datacfg data/ape.data --modelcfg cfg/yolo-pose.cfg --initweightfile cfg/darknet19_448.conv.23 --pretrain_num_epochs 15

****

### 5. 참고

* Original src.: https://github.com/microsoft/singleshotpose
      
      https://www.dropbox.com/s/lvmr4ssdyo2ham3/singleshotpose-master.zip?dl=0
      
<br>

* 00d3119 (on 21 Oct 2019)에서 수정한 부분 <br>
 
--> region_loss.py <br>
  134. 버전 업데이트에 따른 문법 변경 <br>
**# nProposals = int((conf > 0.25).sum().data[0])** <br>
**nProposals = int((conf > 0.25).sum().data)** <br>
  173. 버전 업데이트에 따른 문법 변경 <br>
**#print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_conf.data[0], loss.data[0]))** <br>
**print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, conf %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data, loss_y.data, loss_conf.data, loss.data))** <br>

--> train.py <br>
  309. 백그라운드 영상 경로 변경 <br>
**# bg_file_names = get_all_files('VOCdevkit/VOC2012/JPEGImages')** <br>
**bg_file_names = get_all_files('BG/JPEGImages')** <br>
