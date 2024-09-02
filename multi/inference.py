from __future__ import print_function
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import argparse
import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from darknet_multi import Darknet
from utils_multi import *    
from cfg import parse_cfg
from region_loss_multi import RegionLoss
import dataset_multi
import torchvision.ops as ops

#Hyperparameters
# im_width , im_height = 1680,1080
im_width , im_height = 1267,843

save = True
gpus = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
torch.cuda.manual_seed(int(time.time()))

def inference(modelcfg, weightfile, imagefile, save_path):
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    test_width = model.width
    test_height = model.height

    num_keypoints = model.num_keypoints
    conf_thresh = model.conf_thresh
    num_labels = num_keypoints * 2 + 1  # +2 for width, height,  +1 for class label
    num_classes          = model.num_classes
    anchors              = model.anchors
    num_anchors          = model.num_anchors

    # Read image
    img = Image.open(imagefile).convert('RGB')
    img_resized = img.resize((test_width, test_height))
    data = transforms.ToTensor()(img_resized).unsqueeze(0)

    # Pass data to GPU
    data = data.cuda()
    with torch.no_grad():
        data = data

    # 추론 시작 시간 기록
    start_time = time.time()

    # 이미지를 numpy 배열로 변환
    img = data[0, :, :, :].cpu().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))  # 채널을 마지막으로 이동 (H, W, C)

    # Forward pass
    output = model(data).data

    all_boxes = get_multi_region_boxes(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, only_objectness=0)

    # 시각화 위한 준비
    fig, ax = plt.subplots(figsize=(im_width / 100, im_height / 100))
    ax.set_xlim((0, im_width))
    ax.set_ylim((0, im_height))
    ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))

    # 예측된 박스가 없는 경우, 이미지만 저장
    if all(all_box is None for all_box in all_boxes):
        ax.set_xlim((0, im_width))
        ax.set_ylim((0, im_height))
        ax.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))
        ax.invert_yaxis()
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return

    # 클래스 이름 리스트 (모델에 따라 다름)
    class_names = ['glove', 'shoes', 'tin','box']  # 예시 클래스 이름 설정

    for c in range(num_classes):
        best_conf_est = -sys.maxsize
        box_pr = None
        for j in range(len(all_boxes)):
            boxes = all_boxes[j]  # 각 이미지의 박스들
            for box in boxes:
                if (box[2 * num_keypoints] > best_conf_est) and (int(box[-1]) == c):
                    best_conf_est = box[2 * num_keypoints]
                    box_pr = box
                    print(box_pr[16:])

            if box_pr is not None:
                print('box_pr : ')
                print(box_pr[16:])
                corners2D_pr = np.array(np.reshape(box_pr[:2 * num_keypoints], [-1, 2]), dtype='float32')
                # 좌표를 모델 입력 크기에서 실제 이미지 크기로 변환
                corners2D_pr[:, 0] *= im_width
                corners2D_pr[:, 1] *= im_height

                edges_corners = [
                    [0, 2], [2, 1], [1, 3], [3, 0],  # 아래 사각형 (앞면)
                    [4, 6], [6, 5], [5, 7], [7, 4],  # 위 사각형 (뒷면)
                    [2, 6], [0, 4], [3, 7], [1, 5]   # 아래와 위를 연결하는 엣지들
                ]

                # 선 그리기
                for edge in edges_corners:
                    ax.plot(corners2D_pr[edge, 0], corners2D_pr[edge, 1], color='b', linewidth=1.5)

                # 포인트 그리기 - 첫 8개의 코너만 사용
                ax.scatter(corners2D_pr[:, 0], corners2D_pr[:, 1], color='r', s=15)  # 포인트 크기는 's'로 조절 가능

                # 클래스 레이블 위치 계산 (박스의 꼭짓점 중 가장 높은 위치를 기준으로 레이블을 배치)
                label_x = corners2D_pr[:, 0].min()
                label_y = corners2D_pr[:, 1].min() - 10  # 박스 위에 레이블을 위치시키기 위해 살짝 위로 이동

                # 예측된 클래스의 이름 가져오기
                predicted_class_name = class_names[c]  # 클래스 이름 매핑

                # 3D 바운딩 박스 위에 클래스 이름 표시
                ax.text(label_x, label_y, predicted_class_name, color='white', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.5'))

    # y축을 뒤집어 이미지 좌표계와 일치시킴
    ax.invert_yaxis()
    ax.axis('off')  # 축을 숨김

    # 이미지만 저장 (테두리 제거)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0,dpi=100)

    plt.close(fig)  


def initialize_model(modelcfg, weightfile):
    # 모델 초기화 및 가중치 로드
    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    return model

def inference_from_image(model, img_bgr):
    # 이미지가 이미 BGR 포맷으로 제공됨
    test_width           = model.width
    test_height          = model.height
    num_keypoints        = model.num_keypoints
    conf_thresh          = model.conf_thresh
    num_labels           = num_keypoints * 2 + 1  # +2 for width, height,  +1 for class label
    num_classes          = model.num_classes
    anchors              = model.anchors
    num_anchors          = model.num_anchors

    # BGR 이미지를 바로 사용하여 PIL 변환 없이 Tensor로 변환
    img_resized = cv2.resize(img_bgr, (test_width, test_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    data = transforms.ToTensor()(img_rgb).unsqueeze(0)

    # Pass data to GPU
    data = data.cuda()
    with torch.no_grad():
        output = model(data).data
    # 추론 시작 시간 기록
    start_time = time.time()

    # eliminate low-confidence predictions
    box_pr = get_region_boxes(output, 1, num_keypoints, conf_thresh)
    if box_pr is None or len(box_pr) == 0:
        return  

    box_pr = np.array([tensor_item.detach().numpy() for tensor_item in box_pr])
    
    # 2D 코너 포인트 계산
    corners2D_pr = np.array(np.reshape(box_pr[:16], [8, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * img_bgr.shape[1]  # im_width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * img_bgr.shape[0]  # im_height

    # # 선 그리기
    edges_corners = [
        [0, 2], [2, 1], [1, 3], [3, 0],  # 아래 사각형 (앞면)
        [4, 6], [6, 5], [5, 7], [7, 4],  # 위 사각형 (뒷면)
        [2, 6], [0, 4], [3, 7], [1, 5]   # 아래와 위를 연결하는 엣지들
    ]
    # edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    
    for edge in edges_corners:
        pt1 = (int(corners2D_pr[edge[0], 0]), int(corners2D_pr[edge[0], 1]))
        pt2 = (int(corners2D_pr[edge[1], 0]), int(corners2D_pr[edge[1], 1]))
        cv2.line(img_bgr, pt1, pt2, (255, 0, 0), 2)  # 파란색 라인

    # 포인트 그리기 - 첫 8개의 코너만 사용
    for i in range(8):
        pt = (int(corners2D_pr[i, 0]), int(corners2D_pr[i, 1]))
        cv2.circle(img_bgr, pt, 5, (0, 0, 255), -1)  # 빨간색 포인트
    return img_bgr

from tqdm import tqdm
import time

def video_inference(modelcfg, weightfile, videofile):
    # 동영상 파일 읽기
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수

    # 모델 초기화 (한 번만 수행)
    model = initialize_model(modelcfg, weightfile)

    # 동영상 작성을 위한 VideoWriter 설정
    output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0

    # 전체 처리 시간 측정을 위한 시작 시간 기록
    start_time = time.time()

    # tqdm을 사용하여 프로그레스 바 추가
    with tqdm(total=total_frames // 2) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 30프레임 중 홀수 프레임만 처리 (15프레임)
            if frame_idx % 2 == 0:
                # 이미 BGR 포맷이므로 변환 없이 사용
                processed_frame = inference_from_image(model, frame)

                # inference_from_image가 None을 반환하면 원본 이미지를 사용
                if processed_frame is None:
                    processed_frame = frame
                else:
                    # 원본 해상도와 맞추기
                    if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                        processed_frame = cv2.resize(processed_frame, (width, height))

                # 동영상에 프레임 추가
                output_video.write(processed_frame)
                pbar.update(1)  # 프로그레스 바 업데이트

            frame_idx += 1

    # 전체 처리 시간 측정을 위한 종료 시간 기록
    end_time = time.time()

    # 총 경과 시간
    elapsed_time = end_time - start_time

    # 평균 FPS 계산
    processed_frames = frame_idx // 2  # 실제 처리된 프레임 수 (반만 처리했으므로)
    avg_fps = processed_frames / elapsed_time
    print(f"Total processed time: {elapsed_time:.2f}")
    print(f"Average FPS: {avg_fps:.2f}")
    
    cap.release()
    output_video.release()



def process_folder(modelcfg, weightfile, folder):
    # 모델 초기화 (한 번만 수행)
    model = initialize_model(modelcfg, weightfile)

    # 폴더 내의 모든 이미지 파일 처리
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            save_path = os.path.join(folder, f'processed_{filename}')

            # 이미지 파일을 열어 PIL.Image 객체로 변환
            img = Image.open(file_path).convert('RGB')

            # inference_from_image 함수 호출하여 이미지를 처리
            processed_img = inference_from_image(model, img)

            # 결과 이미지를 저장
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)  # 저장을 위해 BGR로 변환
            cv2.imwrite(save_path, processed_img)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SingleShotPose Inference')
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose-multi.cfg')  # network config
    parser.add_argument('--weightfile', type=str, default='backup_multi/model11.weights')  # pretrained weights
    parser.add_argument('--file', type=str, required=True, help='image or video file or folder for inference')  # file or folder for inference
    args = parser.parse_args()

    # 파일 확장자를 기준으로 이미지, 동영상 또는 폴더 선택
    file_ext = os.path.splitext(args.file)[-1].lower()
    if os.path.isdir(args.file):
        process_folder(args.modelcfg, args.weightfile, args.file)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        inference(args.modelcfg, args.weightfile, args.file, 'output_image.png')
    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_inference(args.modelcfg, args.weightfile, args.file)
    else:
        print("지원되지 않는 파일 형식입니다. 이미지 또는 동영상 파일을 사용하세요.")
