import json
import os
import random
import math
import shutil
from utils import *

'''#############################################################################################
<OPTIONS>
dataset_base_dir : 객체 3D 데이터 경로(라벨링데이터, 원천데이터 폴더를 포함)]
data : 객체 3D 대상명
train_ratio : train/test dataset의 비율


<OUTPUT>
LINEMOD 데이터 구성을 위해, 다음 파일들을 생성합니다.
- test.txt
- train,txt
- [폴더명].data
##############################################################################################'''

############# 1. AIHUB 데이터셋
 
dataset_base_dir="/data/yolo3d/YOLO3D/3dhub/MISO/dataset"
data = "tin02"

# ratio
train_ratio = 0.8  # test_ratio = 1 -train_ratio

# 원본 이미지 사이즈
img_width,img_height = '1280','720' #예) AIHUB 데이터셋

# 원천데이터
origin_data_dir = dataset_base_dir + "/" + data + "/" + data + ".원천데이터"
origin_image_dir = origin_data_dir + "/" + data + '.Images'

# 라벨링데이터
labeling_dir = dataset_base_dir + "/" + data + "/" + data + ".라벨링데이터"
labeling_threed_json_dir = labeling_dir + "/" + data + ".3D_json"

# ############# 2. Custom 데이터셋
# data = "custom_dataset"

# # ratio
# train_ratio = 0.8 # test_ratio = 1 -train_ratio

# # 원천데이터
# origin_dir = dataset_base_dir + "/" + data
# origin_image_dir = origin_dir
# labeling_threed_json_dir = origin_dir

'''##############################################################################################'''
class DataProperty:
    def setTrain(self, train):
        self.train = train

    def setValid(self, valid):
        self.valid = valid

    def setBackup(self, backup):
        self.backup = backup

    def setTrRange(self, tr_range):
        self.tr_range = tr_range

    def setName(self, name):
        self.name = name

    def setDiam(self, diam):
        self.diam = diam

    def setGpus(self, gpus):
        self.gpus = gpus

    def setWidth(self, width):
        self.width = width

    def getWidth(self):
        return self.width

    def setHeight(self, height):
        self.height = height

    def getHeight(self):
        return self.height

    def toString(self):
        ret = 'train = ' + self.train + '\n'
        ret += 'valid = ' + self.valid + '\n'
        ret += 'backup = ' + self.backup + '\n'
        ret += 'name = ' + self.name + '\n'
        ret += 'gpus = ' + str(self.gpus) + '\n'
        ret += 'width = ' + self.width + '\n'
        ret += 'height = ' + self.height
        return ret


if not (os.path.isdir("data")):
    os.makedirs(os.path.join("data"))

if not (os.path.isdir(os.path.join("data",data))):
    os.makedirs(os.path.join("data",data))

dataProperty = DataProperty();
dataProperty.setTrain('data/' + data + '/train.txt')
dataProperty.setValid('data/' + data + '/test.txt')
dataProperty.setBackup('backup/' + data )
dataProperty.setName('tin')

dataProperty.setGpus("0,1")

name_list = os.listdir(labeling_threed_json_dir)

with open(labeling_threed_json_dir + '/' + name_list[0], 'r') as f:
    data2 = json.load(f)
    dataProperty.setWidth(img_width)
    dataProperty.setHeight(img_height)

with open(os.path.join('data', data + ".data"), 'w') as file:
    file.write(dataProperty.toString())

##################################

name_list = os.listdir(labeling_threed_json_dir)

labels_txt_data_dir = os.path.join('data', data, 'labels');
if not (os.path.isdir(labels_txt_data_dir)):
    os.makedirs(os.path.join(labels_txt_data_dir))

width = float(dataProperty.getWidth())
height = float(dataProperty.getHeight())

for file_name in name_list:
    # 파일 확장자가 .json인 파일만 처리
    if file_name.endswith('.json'):
        file_path = os.path.join(labeling_threed_json_dir, file_name)
        with open(file_path, 'r') as f:
            data2 = json.load(f)
        # 'labelingInfo'와 관련된 예외 처리 추가
            if 'labelingInfo' in data2 and '3DBox' in data2['labelingInfo'][0] and 'location' in data2['labelingInfo'][0]['3DBox']:
                location = data2['labelingInfo'][0]['3DBox']['location'][0]
                k = open(labels_txt_data_dir + '/' + file_name[-18:-5] + '.txt', 'w')
                k.write('2 ')
                k.write(str(float(location['x2'])/width))
                k.write(' ')
                k.write(str(float(location['y2'])/height))
                k.write(' ')
                k.write(str(float(location['x4'])/width))
                k.write(' ')
                k.write(str(float(location['y4'])/height))
                k.write(' ')
                k.write(str(float(location['x1'])/width))
                k.write(' ')
                k.write(str(float(location['y1'])/height))
                k.write(' ')
                k.write(str(float(location['x3'])/width))
                k.write(' ')
                k.write(str(float(location['y3'])/height))
                k.write(' ')
                k.write(str(float(location['x6'])/width))
                k.write(' ')
                k.write(str(float(location['y6'])/height))
                k.write(' ')
                k.write(str(float(location['x8'])/width))
                k.write(' ')
                k.write(str(float(location['y8'])/height))
                k.write(' ')
                k.write(str(float(location['x5'])/width))
                k.write(' ')
                k.write(str(float(location['y5'])/height))
                k.write(' ')
                k.write(str(float(location['x7'])/width))
                k.write(' ')
                k.write(str(float(location['y7'])/height))
                k.write(' ')


data_root = os.path.join('data', data, 'labels')
loader_root = os.path.join('data', data)

name_list2 = os.listdir(data_root)
image_list = [name for name in name_list2 if name[-4:] == '.txt']
train_len = math.floor(len(image_list) * train_ratio)

train_name = random.sample(image_list, train_len)
valid_name = [name for name in image_list if name not in train_name]

with open(os.path.join(loader_root, 'train.txt'), 'w') as file:
    for i in range(len(train_name)):
        file.write(os.path.join(origin_image_dir, train_name[i].replace('.txt', '.png') + '\n'))

with open(os.path.join(loader_root, 'test.txt'), 'w') as file:
    for i in range(len(valid_name)):
        file.write(os.path.join(origin_image_dir, valid_name[i].replace('.txt', '.png') + '\n'))