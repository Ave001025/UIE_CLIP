# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np

import scipy.io
import random

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    #parser.add_argument('--file_path', default="/data/zengzekai/UIE/Datasets/ChallengeDB_release/Images/", help='path to input image file')
    #parser.add_argument('--csv_path', default="/data/zengzekai/UIE/Datasets/ChallengeDB_release/Data/", help='path to input image file')
    
    parser.add_argument('--file_path', default="/data/zengzekai/UIE/Datasets/SOTA_Dataset", help='path to input image file')
    parser.add_argument('--csv_path', default="/data/zengzekai/UIE/Datasets/", help='path to input image file')
    
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    #image_mat = scipy.io.loadmat(os.path.join(args.csv_path, 'AllImages_release.mat'))['AllImages_release'][7:]
    #mos_mat = scipy.io.loadmat(os.path.join(args.csv_path, 'AllMOS_release.mat'))['AllMOS_release'][0][7:]
#    random.seed(1)
#    index = list(range(0, 8000))
#    random.shuffle(index)
#    train_index = index[0:7200]
#    test_index = index[7200:8000]
    
    with open('my_test_id_best.txt') as f:
        test_index = [int(line.rstrip()) for line in f]
    
    image_mat = scipy.io.loadmat(os.path.join(args.csv_path, 'SOTA_dataset.mat'))['path'][:-1]
    mos_mat = scipy.io.loadmat(os.path.join(args.csv_path, 'SOTA_dataset.mat'))['mos'][:-1]
    
    #print("打印一下")
    #print(args.file_path)
    #print(image_mat[0][0][0])
    #temp = os.path.join(args.file_path, image_mat[0][0][0][0])
    #temp  = args.file_path+image_mat[0][0][0]
    #print("路径：",temp)
    
    pred_score = []
    y_true = []
    #这里也是通过读取这些文件路径进行读取的，然后使用相应允许的MOS值，两者进行一个比较
    #这里的代码估计该有喜爱就可以了，改成图像的路径，遍历一下就可以了，可以使用这个得到一个文件夹图像的评分
    #for i in tqdm(range(len(image_mat))):
    for i in tqdm(test_index):
        #output, attributes = restoration_inference(model, os.path.join(args.file_path, image_mat[i][0][0]), return_attributes=True)
        output, attributes = restoration_inference(model, args.file_path+image_mat[i][0][0], return_attributes=True)
        output = output.float().detach().cpu().numpy()
        pred_score.append(attributes[0])
        y_true.append(mos_mat[i][0])
        

    pred_score = np.squeeze(np.array(pred_score))*100
    #y_true = mos_mat
    y_true = np.squeeze(np.array(y_true))
    #print("打印预测分数和真实分数：")
    #print(pred_score,y_true)
    #print(pred_score.shape,y_true.shape)

    p_srocc = srocc(pred_score, y_true)
    p_plcc = plcc(pred_score, y_true)

    print(args.checkpoint)
    print('SRCC: {} | PLCC: {}'.\
          format(p_srocc, p_plcc))


if __name__ == '__main__':
    main()

