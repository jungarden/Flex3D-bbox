## NIA 5. 객체 3D 데이터 (2021.07.22)

### 1. 다운로드 및 코드 경로에서 압축 해제

* 학습/평가 코드 + LINEMOD 정리 (nia_sample.7z: 5GB)
           
      https://www.dropbox.com/s/upat6b6nyij47dt/nia_sample.7z?dl=0

  코드 경로: ./ <br>
  문서 공유: ./doc <br>
  모델 구조 정의: ./cfg <br>
  데이터 및 학습된 모델 경로: ./data <br>
  LINEMOD 정리 (원본 + 세그멘테이션/큐브 라벨 포맷): ./LINEMOD <br>
  배경 영상 (학습 시 다양한 배경 증강을 위해 활용): ./BG <br>

****

### 2. 3090 GPU 기준 설치 방법

* PyCharm, Anaconda --> 3.6 Python 가상환경으로 선택
* PyTorch v1.8.0 + CUDA 11.1 설치

      conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge <br>

* OpenCV 설치

      pip3 install opencv-python
      
* Scipy 1.2.0 설치

      pip3 install scipy==1.2.0
      
* Pillow 8.2.2 설치

      pip3 install pillow==8.2.0
      
****

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
