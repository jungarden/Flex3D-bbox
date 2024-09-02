import os
import random

# 경로 설정
image_folder = '/data/yolo3d/YOLO3D/3dhub/MISO/baseline/trainbox/Images'
train_txt_path = '/data/yolo3d/YOLO3D/3dhub/MISO/baseline/trainbox/train.txt'
test_txt_path = '/data/yolo3d/YOLO3D/3dhub/MISO/baseline/trainbox/test.txt'

# 파일 리스트 가져오기
file_list = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 파일 리스트 섞기 (랜덤 셔플)
random.shuffle(file_list)

# 85%와 15%로 나누기
train_size = int(len(file_list) * 0)
train_files = file_list[:train_size]
test_files = file_list[train_size:]

# # train.txt에 저장
with open(train_txt_path, 'w') as train_file:
    for file_name in train_files:
        train_file.write(f"{os.path.join(image_folder, file_name)}\n")

# test.txt에 저장
with open(test_txt_path, 'w') as test_file:
    for file_name in test_files:
        test_file.write(f"{os.path.join(image_folder, file_name)}\n")

print(f"Train set: {len(train_files)} files")
print(f"Test set: {len(test_files)} files")

