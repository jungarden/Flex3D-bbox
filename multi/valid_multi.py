import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt
import scipy.misc
import warnings
import sys
import argparse
warnings.filterwarnings("ignore")
from torch.autograd import Variable
from torchvision import datasets, transforms

import dataset_multi
from darknet_multi import Darknet
from utils_multi import *
from cfg import parse_cfg

def valid(datacfg, cfgfile, weightfile):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse data configuration files
    data_options = read_data_cfg(datacfg)
    valid_images = data_options['valid']
    name         = data_options['name']
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height']) 

    # Parse net configuration file
    net_options   = parse_cfg(cfgfile)[0]
    loss_options  = parse_cfg(cfgfile)[-1]
    conf_thresh   = float(net_options['conf_thresh'])
    num_keypoints = int(net_options['num_keypoints'])
    num_classes   = int(loss_options['classes'])
    num_anchors   = int(loss_options['num'])
    anchors       = [float(anchor) for anchor in loss_options['anchors'].split(',')]

    # Network I/O params
    num_labels = 2*num_keypoints+1 # +2 for width, height, +1 for object class
    errs_2d = []  # to save
    with open(valid_images) as fp:     # validation file names
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Compute-related Parameters
    use_cuda = True # whether to use cuda or no
    kwargs = {'num_workers': 4, 'pin_memory': True} # number of workers etc.

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the dataloader for the test dataset
    valid_dataset = dataset_multi.listDataset(valid_images, shape=(model.width, model.height), shuffle=False, objclass=name, transform=transforms.Compose([transforms.ToTensor(),]))
    test_loader   = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs) 

    # Iterate through test batches (Batch size for test data is 1)
    logging('Testing {}...'.format(name))
    for batch_idx, (data, target) in enumerate(test_loader):
        
        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            # target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()
        
        # Forward pass
        output = model(data).data  
        t3 = time.time()
        
        # Using confidence threshold, eliminate low-confidence predictions
        trgt = target[0].view(-1, num_labels)
        all_boxes = get_multi_region_boxes_valid(output, conf_thresh, num_classes, num_keypoints, anchors, num_anchors, int(trgt[0][0]), only_objectness=0)        
        t4 = time.time()
        
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
                
                norm         = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)

        t5 = time.time()

    # Compute 2D projection score
    eps = 1e-5
    for px_threshold in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        # Print test statistics
        logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))

if __name__ == '__main__' and __package__ is None:

    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--modelcfg', type=str, default='cfg/yolo-pose-multi.cfg') # network config
    parser.add_argument('--initweightfile', type=str, default='backup_multi/model11.weights') # initialization weight
    args = parser.parse_args()
    datacfg = 'data/glove00.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'data/shoes01.data'
    valid(datacfg, args.modelcfg, args.initweightfile)
    datacfg = 'data/box03.data'
    valid(datacfg, args.modelcfg, args.initweightfile)