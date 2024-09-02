from __future__ import print_function
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import shutil
from torchvision import datasets, transforms
from torch.autograd import Variable
from darknet_multi import Darknet
from utils_multi import *    
from cfg import parse_cfg
from region_loss_multi import RegionLoss
import dataset_multi
import warnings
warnings.filterwarnings("ignore")

# Adjust learning rate during training, learning schedule can be changed in network config file
def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr 

def train(epoch):
    global processed_batches
    # Get the dataloader for training dataset
    train_loader = torch.utils.data.DataLoader(dataset_multi.listDataset(trainlist, shape=(init_width, init_height),
                                                            transform=transforms.Compose([transforms.ToTensor(),]), 
                                                            train=True, 
                                                            seen=model.module.seen,
                                                            batch_size=batch_size,
                                                            num_workers=8),
                                                batch_size=batch_size, shuffle=True, **kwargs)

    # TRAINING
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    # Start training
    model.train()
    niter = 0
    # Iterate through batches
    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches += 1

        if use_cuda:
            data = data.cuda()
            target = target.cuda()  # 이 부분이 빠져 있었습니다.

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        region_loss.seen += data.size(0)

        # Compute loss, grow an array of losses for saving later on
        loss = region_loss(output, target, epoch)
        training_iters.append(epoch * math.ceil(len(train_loader.dataset) / float(batch_size)) + niter)
        training_losses.append(loss.item())  # .data 대신 .item() 사용
        niter += 1

        # Backprop: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update weights
        optimizer.step()
    
    return epoch * math.ceil(len(train_loader.dataset) / float(batch_size)) + niter - 1

def eval(niter, datacfg):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
            
    # Parse configuration files
    options       = read_data_cfg(datacfg)
    valid_images  = options['valid']
    backupdir     = options['backup']
    name          = options['name']
    im_width     = int(options['width'])
    im_height    = int(options['height']) 
    testtime     = True
    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
        
    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model.eval()
    iou_acc = []
    errs_2d = []
    total_loss = 0.0  # 총 손실 초기화

    
    # Get the parser for the test dataset
    valid_dataset = dataset_multi.listDataset(valid_images, shape=(model.module.width, model.module.height),
                                              shuffle=False,
                                              objclass=name,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                              ]))
    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 8, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    # Parameters
    num_classes          = model.module.num_classes
    anchors              = model.module.anchors
    num_anchors          = model.module.num_anchors
    testing_samples      = 0.0

    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test examples 
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():  
            output = model(data)
        
        # 손실 계산
        loss = region_loss(output, target, niter)  # 손실 계산
        total_loss += loss.item()  # 총 손실에 더하기

        # Using confidence threshold, eliminate low-confidence predictions
        trgt = target[0].view(-1, num_labels)
        all_boxes = get_multi_region_boxes_valid(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, int(trgt[0][0]), only_objectness=0)
        # Iterate through all images in the batch
        for i in range(output.size(0)):
            # For each image, get all the predictions
            boxes   = all_boxes[i]
            
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            truths  = target[i].view(-1, num_labels)
            
            # Get how many object are present in the scene
            num_gts = truths_length(truths)

            # Iterate through each ground-truth object
            for k in range(num_gts):
                box_gt = list()
                for j in range(1, num_labels):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])
                
                # If the prediction has the highest confidence, choose it as our prediction
                best_conf_est = -sys.maxsize
                for j in range(len(boxes)):
                    if (boxes[j][2*num_keypoints] > best_conf_est) and (boxes[j][2*num_keypoints+2] == int(truths[k][0])):
                        best_conf_est = boxes[j][2*num_keypoints]
                        box_pr        = boxes[j]
                    
                # Denormalize the corner predictions 
                corners2D_gt = np.array(np.reshape(box_gt[:2*num_keypoints], [-1, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:2*num_keypoints], [-1, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height               
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # Compute pixel error
                pixel_dist      = np.mean(np.linalg.norm(corners2D_gt - corners2D_pr, axis=1))
                errs_2d.append(pixel_dist)
                testing_samples += 1

                # Compute IOU
                iou = compute_convexhull_iou(corners2D_gt, corners2D_pr)
                iou_acc.append(iou)

    # 평균 손실 계산 및 기록
    mean_loss = total_loss / len(test_loader.dataset)
    testing_losses.append(mean_loss)  # 손실 값을 testing_losses 리스트에 추가

    # # Compute 2D reprojection score
    eps = 1e-5
    # for px_threshold in [15, 20, 25, 30, 35, 40]:
    #     acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    #     logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))

    # Compute 2D projection error metrics
    px_threshold = 20
    total_iou = len(np.where(np.array(iou_acc) >= 0.8)[0]) * 100 / (len(iou_acc) + eps)
    acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    mean_corner_err_2d = np.mean(errs_2d)


    if testtime:
        print('-----------------------------------')
        print('   Mean corner error is %f' % (mean_corner_err_2d))
        print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        print('   Intersection Of Union = {:.2f}%'.format(total_iou))
        # print('   Class Accuracy = {:.2f}%'.format(class_accuracy))
        print('-----------------------------------')

    # Register losses and errors for saving later on
    testing_iters.append(niter)
    testing_accuracies.append(acc)


def test(niter):
    modelcfg = 'cfg/yolo-pose-multi.cfg'
    datacfg = 'data/glove00.data'
    logging("Testing glove...")
    eval(niter, datacfg)
    datacfg = 'data/shoes01.data'
    logging("Testing shoes...")
    eval(niter, datacfg)    
    datacfg = 'data/tin02.data'
    logging("Testing tin ...")
    eval(niter, datacfg)    



if __name__ == "__main__":

    # Parse command window input
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='data/occlusion.data') # data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose-multi.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default='cfg/darknet19_448.conv.23') # initialization weights
    parser.add_argument('--pretrain_num_epochs', type=int, default=0) # how many epoch to pretrain
    args                = parser.parse_args()
    datacfg             = args.datacfg
    modelcfg            = args.modelcfg
    initweightfile      = args.initweightfile
    pretrain_num_epochs = args.pretrain_num_epochs

    # Parse data configuration file
    data_options = read_data_cfg(datacfg)
    trainlist    = data_options['train']
    gpus         = data_options['gpus']  
    num_workers  = int(data_options['num_workers'])
    backupdir    = data_options['backup']
    im_width     = int(data_options['im_width'])
    im_height    = int(data_options['im_height']) 

    # Parse network and training configuration parameters
    net_options   = parse_cfg(modelcfg)[0]
    loss_options  = parse_cfg(modelcfg)[-1]
    batch_size    = int(net_options['batch'])
    max_batches   = int(net_options['max_batches'])
    max_epochs    = int(net_options['max_epochs'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    steps         = [float(step) for step in net_options['steps'].split(',')]
    scales        = [float(scale) for scale in net_options['scales'].split(',')]
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    # Further params
    if not os.path.exists(backupdir):
        makedirs(backupdir)
    # bg_file_names = get_all_files('/data/yolo3d/YOLO3D/3dhub/MISO/VOCdevkit/VOC2012/JPEGImages')
    nsamples      = file_lines(trainlist)
    use_cuda      = True
    seed          = int(time.time())
    best_acc      = -sys.maxsize
    num_labels    = num_keypoints*2+1 # + 2 for image width, height, +1 for image class

    # Specify which gpus to use
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Specifiy the model and the loss
    model       = Darknet(modelcfg)
    region_loss = RegionLoss(num_keypoints=num_keypoints, num_classes=num_classes, anchors=anchors, num_anchors=num_anchors, pretrain_num_epochs=pretrain_num_epochs)

    # Model settings
    model.load_weights_until_last(initweightfile) 
    model.print_network()
    model.seen        = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    processed_batches = model.seen/batch_size
    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen//nsamples 

    # Variables to save
    training_iters       = []
    training_losses      = []
    testing_iters        = []
    testing_accuracies   = []
    testing_losses      = []


    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}


    # Pass the model to GPU
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=list(map(int, gpus.split(",")))).cuda() # Multiple GPU parallelism

    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

    evaluate = False
    if evaluate:
        logging('evaluating ...')
        test(0, 0)
    else:
        for epoch in range(init_epoch, max_epochs): 
            # TRAIN
            niter = train(epoch)
            # TEST and SAVE
            if (epoch % 10 == 0) and (epoch is not 0): 
                test(niter)
                logging('save training stats to %s/costs12.npz' % (backupdir))
                np.savez(os.path.join(backupdir, "costs12.npz"),
                    training_iters=training_iters,
                    training_losses=training_losses,
                    testing_iters=testing_iters,
                    testing_losses =testing_losses,
                    testing_accuracies=testing_accuracies) 
                if (np.mean(testing_accuracies[-4:]) > best_acc ): # testing for 4 different objects
                    best_acc = np.mean(testing_accuracies[-4:])
                    logging('best model so far!')
                    logging('save weights to %s/model12.weights' % (backupdir))
                    model.module.save_weights('%s/model12.weights' % (backupdir))
        # shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))