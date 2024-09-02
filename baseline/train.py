from __future__ import print_function
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import random
import math
import shutil
import argparse
from torchvision import datasets, transforms
from torch.autograd import Variable  # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

import dataset
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet

import warnings
warnings.filterwarnings("ignore")
import json

iou_acc = []

# Create new directory
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        param_group['lr'] = lr / batch_size
    return lr

def train(epoch):

    global processed_batches

    # Initialize timer
    t0 = time.time()

    # Get the dataloader for training dataset
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(trainlist,
                                                                   shape=(init_width, init_height),
                                                                   shuffle=True,
                                                                   transform=transforms.Compose([transforms.ToTensor(), ]),
                                                                   train=True,
                                                                   seen=model.module.seen,
                                                                   batch_size=batch_size,
                                                                   num_workers=4),
                                               batch_size=batch_size, shuffle=False, **kwargs)

    # Training
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))

    # Start training
    model.train()
    niter = 0

    # Iterate through batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Adjust learning rate
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        # Pass the data to GPU
        if use_cuda:
            data = data.cuda()

        # Wrap tensors in Variable class for automatic differentiation
        data, target = Variable(data), Variable(target)

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        model.seen = model.module.seen + data.data.size(0)
        region_loss.seen = region_loss.seen + data.data.size(0)

        # Compute loss, grow an array of losses for saving later on
        loss = region_loss(output, target, epoch)
        training_iters.append(epoch * math.ceil(len(train_loader.dataset) / float(batch_size)) + niter)
        training_losses.append(convert2cpu(loss.data))

        niter += 1

        # Backprop: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update weights
        optimizer.step()

    return epoch * math.ceil(len(train_loader.dataset) / float(batch_size)) + niter - 1

def test(epoch, niter):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Set the module in evaluation mode (turn off dropout, batch normalization etc.)
    model.eval()
    total_loss = 0.0

    # Parameters
    num_classes = model.module.num_classes
    anchors = model.module.anchors
    num_anchors = model.module.num_anchors
    testtime = True
    errs_corner2D = []
    logging("   Testing...")
    logging("   Number of test samples: %d" % len(test_loader.dataset))

    # Iterate through test examples
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output = model(data)

        # Compute loss
        loss = region_loss(output, target, niter)
        total_loss += loss.item()

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)

        # Iterate through all batch elements
        for box_pr, target in zip([all_boxes], [target[0]]):
            truths = target.view(-1, num_keypoints * 2 + 1)
            num_gts = truths_length(truths)

            for k in range(num_gts):
                box_gt = list()
                for j in range(1, 2 * num_keypoints + 1):
                    box_gt.append(truths[k][j])
                box_gt.extend([1.0, 1.0])
                box_gt.append(truths[k][0])

                # Denormalize the corner predictions
                corners2D_gt = np.array(np.reshape(box_gt[:num_keypoints * 2], [num_keypoints, 2]), dtype='float32')
                corners2D_pr = np.array(np.reshape(box_pr[:num_keypoints * 2], [num_keypoints, 2]), dtype='float32')
                corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

                # Compute IOU
                iou = compute_convexhull_iou(corners2D_gt, corners2D_pr)
                iou_acc.append(iou)

                # Compute corner prediction error
                corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                corner_dist = np.mean(corner_norm)
                errs_corner2D.append(corner_dist)

    # Compute and log average loss
    mean_loss = total_loss / len(test_loader.dataset)
    testing_losses.append(mean_loss)

    # Compute 2D projection error metrics
    px_threshold = 20
    eps = 1e-5
    total_iou = len(np.where(np.array(iou_acc) >= 0.7)[0]) * 100 / (len(iou_acc) + eps)
    acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D) + eps)
    mean_corner_err_2d = np.mean(errs_corner2D)

    if testtime:
        print('-----------------------------------')
        print('   Mean corner error is %f' % (mean_corner_err_2d))
        print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        print('   Intersection Of Union = {:.2f}%'.format(total_iou))
        print('-----------------------------------')

    # Register losses and errors for saving later on
    testing_iters.append(niter)
    testing_accuracies.append(acc)

if __name__ == "__main__":

    # Parse configuration files
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='data/trainbox.data')  # Data config
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose.cfg')  # Network config
    parser.add_argument('--initweightfile', type=str, default='backup/parcel3d/model.weights')  # ImageNet initialized weights
    parser.add_argument('--pretrain_num_epochs', type=int, default=5)  # How many epoch to pretrain
    args = parser.parse_args()
    datacfg = args.datacfg
    modelcfg = args.modelcfg
    initweightfile = args.initweightfile
    pretrain_num_epochs = args.pretrain_num_epochs

    # Parse configuration files
    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(modelcfg)[0]
    trainlist = data_options['train']
    testlist = data_options['valid']
    gpus = data_options['gpus']
    num_workers = int(data_options['num_workers'])
    backupdir = data_options['backup']
    if not os.path.exists(backupdir):
        makedirs(backupdir)
    batch_size = int(net_options['batch'])
    max_batches = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum = float(net_options['momentum'])
    decay = float(net_options['decay'])
    nsamples = file_lines(trainlist)
    nbatches = nsamples / batch_size
    steps = [float(step) * nbatches for step in net_options['steps'].split(',')]
    scales = [float(scale) for scale in net_options['scales'].split(',')]

    # Train parameters
    max_epochs = int(net_options['max_epochs'])
    num_keypoints = int(net_options['num_keypoints'])

    # Test parameters
    im_width = int(data_options['width'])
    im_height = int(data_options['height'])
    test_width = int(net_options['test_width'])
    test_height = int(net_options['test_height'])

    # Specify which GPUs to use
    use_cuda = True
    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    # Specify the model and the loss
    model = Darknet(modelcfg)
    # Parameters
    num_classes = model.num_classes
    anchors = model.anchors
    num_anchors = model.num_anchors

    region_loss = RegionLoss(num_keypoints=8, num_classes=num_classes, anchors=anchors, num_anchors=num_anchors, pretrain_num_epochs=pretrain_num_epochs)

    # Model settings
    model.load_weights_until_last(initweightfile)
    model.print_network()
    model.seen = 0
    region_loss.iter = model.iter
    region_loss.seen = model.seen
    processed_batches = model.seen // batch_size
    init_width = model.width
    init_height = model.height
    init_epoch = model.seen // nsamples

    # Variable to save
    training_iters = []
    training_losses = []
    testing_iters = []
    testing_losses = []
    testing_accuracies = []

    # Specify the number of workers
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # Get the dataloader for test data
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(testlist,
                                                                  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(), ]),
                                                                  train=False),
                                              batch_size=1, shuffle=False, **kwargs)

    # Pass the model to GPU
    if use_cuda:
        model = torch.nn.DataParallel(model, device_ids=list(map(int, gpus.split(",")))).cuda()  # Multiple GPU parallelism

    # Get the optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay * batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0, weight_decay=decay * batch_size)

    best_acc = -sys.maxsize
    for epoch in range(init_epoch, max_epochs):
        # Train
        niter = train(epoch)
        # Test and save
        if ((epoch + 1) % 10 == 0) & ((epoch + 1) > 15):
            test(epoch, niter)
            logging('save training stats to %s/costs.npz' % (backupdir))
            np.savez(os.path.join(backupdir, "costs.npz"),
                     training_iters=training_iters,
                     training_losses=training_losses,
                     testing_iters=testing_iters,
                     testing_accuracies=testing_accuracies,
                     testing_losses=testing_losses)
            if (testing_accuracies[-1] > best_acc):
                best_acc = testing_accuracies[-1]
                logging('best model so far!')
                logging('save weights to %s/model.weights' % (backupdir))
                model.module.save_weights('%s/model.weights' % (backupdir))
    # shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))
