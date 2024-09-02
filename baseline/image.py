#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image, ImageChops, ImageMath
import numpy as np


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res  = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):

    ow, oh = img.size
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy, num_keypoints, max_num_gt):
    num_labels = 2 * num_keypoints + 1  # 8개의 키포인트 * 2 + 1(confidence) = 17
    label = np.zeros((max_num_gt, num_labels))
    
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, num_labels))
        cc = 0
        
        for i in range(bs.shape[0]):
            xs = []
            ys = []
            for j in range(num_keypoints):
                xs.append(bs[i][2*j+1])
                ys.append(bs[i][2*j+2])
            
            # 키포인트들을 정규화하고 변환
            for j in range(num_keypoints):
                xs[j] = min(0.999, max(0, xs[j] * sx - dx)) 
                ys[j] = min(0.999, max(0, ys[j] * sy - dy))
                
            # 업데이트된 xs와 ys를 bs에 다시 저장
            for j in range(num_keypoints):
                bs[i][2*j+1] = xs[j]
                bs[i][2*j+2] = ys[j]
            
            label[cc] = bs[i]
            cc += 1
            if cc >= max_num_gt:
                break

    label = np.reshape(label, (-1))
    return label


def change_background(img, mask, bg):
    # oh = img.height  
    # ow = img.width
    ow, oh = img.size
    bg = bg.resize((ow, oh)).convert('RGB')
    
    imcs = list(img.split())
    bgcs = list(bg.split())
    maskcs = list(mask.split())
    fics = list(Image.new(img.mode, img.size).split())
    
    for c in range(len(imcs)):
        negmask = maskcs[c].point(lambda i: 1 - i / 255)
        posmask = maskcs[c].point(lambda i: i / 255)
        fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
    out = Image.merge(img.mode, tuple(fics))
    
    return out

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, num_keypoints, max_num_gt):

    # 파일 이름 추출 및 확장자 변경
    filename = imgpath.split('/')[-1].replace('.png', '.txt').replace('.jpg', '.txt')

    # 경로에 맞게 labpath 변경
    # labpath = os.path.join('data', imgpath.split('/')[-4], 'labels', filename)
    labpath = imgpath.replace('.jpg', '.txt').replace('.png', '.txt').replace('augmented_images','augmented_labels').replace('Images','labels')

    type = ".Images"
    if "투명" in imgpath:
        type = ".TR"

    # maskpath = imgpath.replace('원천데이터', '라벨링데이터').replace(type, '.Mask').replace('.png', '_b.png')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    # mask = Image.open(maskpath).convert('RGB')
    # bg = Image.open(bgpath).convert('RGB')
    
    # img = change_background(img, mask, bg)
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    ow, oh = img.size
    label = fill_truth_detection(labpath, ow, oh, flip, dx, dy, 1./sx, 1./sy, num_keypoints, max_num_gt)
    return img,label

