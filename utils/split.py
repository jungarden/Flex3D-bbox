import os
import random

image_folder = '../dataset/glove00/Images'
train_txt_path = '../baseline/data/glove00/train.txt'
test_txt_path = '../baseline/data/glove00/test.txt'

file_list = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

random.shuffle(file_list)

train_size = int(len(file_list) * 0)
train_files = file_list[:train_size]
test_files = file_list[train_size:]

with open(train_txt_path, 'w') as train_file:
    for file_name in train_files:
        train_file.write(f"{os.path.join(image_folder, file_name)}\n")

with open(test_txt_path, 'w') as test_file:
    for file_name in test_files:
        test_file.write(f"{os.path.join(image_folder, file_name)}\n")

print(f"Train set: {len(train_files)} files")
print(f"Test set: {len(test_files)} files")

