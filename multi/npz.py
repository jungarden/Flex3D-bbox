import numpy as np
import matplotlib.pyplot as plt
import os 

backupdir = '/data/yolo3d/YOLO3D/3dhub/MISO/singleshotpose/multi_obj_pose_estimation/backup_multi'
# npz 파일 로드
data = np.load(os.path.join(backupdir, "costs12.npz"))

# 데이터 로드
training_iters = data['training_iters']
training_losses = data['training_losses']
testing_iters = data['testing_iters']
testing_losses = data['testing_losses']


# 파일에 저장된 배열 확인

for key in data.files:
    print(f"{key}: {data[key]}")

data.close()

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(training_iters, training_losses, label='Training Loss')
plt.plot(testing_iters, testing_losses, label='Testing Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Testing Loss per Iteration')
plt.legend()
plt.grid(True)

save_path = os.path.join(backupdir, 'loss_graph.png')
plt.savefig(save_path)