import os
import time
import torch
import argparse
import scipy.io
import warnings
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import json
import dataset
from darknet import Darknet
from utils import *


def valid(datacfg, modelcfg, weightfile):
    def truths_length(truths, max_num_gt=50):
        for i in range(max_num_gt):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    backupdir = data_options['backup']
    name = data_options['name']
    gpus = data_options['gpus']
    im_width = int(data_options['width'])
    im_height = int(data_options['height'])
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    seed = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    save = True
    testtime = True
    num_classes = 1
    testing_samples = 0.0
    edges_corners = [
        [0, 2], [2, 1], [1, 3], [3, 0],  # 아래 사각형 (앞면)
        [4, 6], [6, 5], [5, 7], [7, 4],  # 위 사각형 (뒷면)
        [2, 6], [0, 4], [3, 7], [1, 5]   # 아래와 위를 연결하는 엣지들
    ]
    if save:
        makedirs(backupdir + '/test')
        makedirs(backupdir + '/test/gt2')
        makedirs(backupdir + '/test/pr2')
        makedirs(backupdir + '/test/images2')

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    iou_acc = []
    errs_2d = []
    errs_3d = []
    errs_trans = []
    errs_angle = []
    errs_corner2D = []
    preds_trans = []
    preds_rot = []
    preds_corners2D = []
    gts_trans = []
    gts_rot = []
    gts_corners2D = []
    
    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    test_width = model.test_width
    test_height = model.test_height
    num_keypoints = model.num_keypoints
    num_labels = num_keypoints * 2 + 1  # +2 for width, height,  +1 for class label

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images,
                                        shape=(test_width, test_height),
                                        shuffle=False,
                                        transform=transforms.Compose([transforms.ToTensor(), ]))

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0

    type = ".Images"
    if "투명" in test_loader.dataset.lines[0]:
        type = ".TR"

    for batch_idx, (data, target) in enumerate(test_loader):
        # Images
        img = data[0, :, :, :]
        img = img.numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        t1 = time.time()
        # Pass data to GPU
        data = data.cuda()
        target = target.cuda()
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        with torch.no_grad():
            data = data

        t2 = time.time()
        # Forward pass
        output = model(data).data
        t3 = time.time()
        # eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)
        t4 = time.time()
        # Evaluation
        # Iterate through all batch elements
        for box_pr, target in zip([all_boxes], [target[0]]):
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths = target.view(-1, num_labels)
            # Get how many objects are present in the scene
            num_gts = truths_length(truths)
            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2 * num_keypoints + 1):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])

                # Denormalize the corner predictions
                corners2D_gt = np.array(np.reshape(box_gt[:16], [8, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:16], [8, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height
                preds_corners2D.append(corners2D_pr)
                gts_corners2D.append(corners2D_gt)

                # Compute IOU
                iou = compute_convexhull_iou(corners2D_gt, corners2D_pr)
                iou_acc.append(iou)

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                errs_corner2D.append(corner_dist)

                # save prediction image
                if save:
                    # Visualize
                    plt.xlim((0, im_width))
                    plt.ylim((0, im_height))
                    #plt.imshow(scipy.misc.imresize(img, (im_height, im_width)))
                    plt.imshow(cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC))
                    # # Projections
                    for edge in edges_corners:
                        plt.plot(corners2D_gt[edge, 0], corners2D_gt[edge, 1], color='g', linewidth=1.0)
                        plt.plot(corners2D_pr[edge, 0], corners2D_pr[edge, 1], color='b', linewidth=1.0)
                    plt.gca().invert_yaxis()
                    # plt.show()
                    plt.savefig(backupdir + '/test/images2/' + valid_files[count].split('/')[-1])
                    plt.clf()


                # Sum error
                if save:
                    np.savetxt(backupdir + '/test/gt2/corners_' + valid_files[count][-10:-3] + 'txt',
                               np.array(corners2D_gt, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr2/corners_' + valid_files[count][-10:-3] + 'txt',
                               np.array(corners2D_pr, dtype='float32'))

                count = count + 1

        t5 = time.time()

    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 20  # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    eps = 1e-5
    total_iou = len(np.where(np.array(iou_acc) >= 0.5)[0]) * 100 / (len(iou_acc) + eps)
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d) + eps)
    corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D) + eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    if testtime:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (t2 - t1))
        print('    forward pass : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print(' prediction time : %f' % (t4 - t1))
        print('            eval : %f' % (t5 - t4))
        print('-----------------------------------')

    # Print test statistics
    logging('Results of {}'.format(name))
    logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, corner_acc))
    logging('   Intersection Of Union = {:.2f}%'.format(total_iou))


if __name__ == '__main__':

    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='data/trainbox.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg') # network config
    parser.add_argument('--weightfile', type=str, default='backup/trainbox/mode4.weights') # imagenet initialized weights
    args       = parser.parse_args()
    datacfg    = args.datacfg
    modelcfg   = args.modelcfg
    weightfile = args.weightfile
    valid(datacfg, modelcfg, weightfile)
