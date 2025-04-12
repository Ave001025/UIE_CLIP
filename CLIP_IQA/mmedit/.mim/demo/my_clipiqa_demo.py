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


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_coop_koniq.py', help='test config file path')
    parser.add_argument('--checkpoint', default="/data/zengzekai/UIE/Codes/CLIP-IQA-2-3.8/work_dirs/clipiqa_coop_koniq_224_Clear_Turbid_random_best/iter_10000.pth", help='checkpoint file')
    parser.add_argument('--file_path', default="/data/zengzekai/UIE/Datasets/SOTA_Dataset", help='path to input image file')
    parser.add_argument('--csv_path', default="/data/zengzekai/UIE/Datasets/", help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))
    
    folder = "/data/zengzekai/UIE/Codes/test01_uranker/examples/ranker_example/"
    #"/data/zengzekai/UIE/Datasets/UIEB/valid/input/" 
    folder_2 = "/data/zengzekai/UIE/test03_Semi/results/Ucolor/test_90/"
    
    
    image_paths = []
    image_paths_2 = []
    
    for filename in os.listdir (folder):
          path = os.path.join (folder, filename)
          image_paths.append (path)
    
    
    for filename_2 in os.listdir (folder_2):
          path_2 = os.path.join (folder_2, filename_2)
          image_paths_2.append (path_2)
    
    
    pred_score = []
    pred_score_2 = []
    count = 0    
    for i in tqdm(range(len(image_paths))):
        #output, attributes = restoration_inference(model, os.path.join(args.file_path, image_mat[i][0][0]), return_attributes=True)
        output, attributes = restoration_inference(model, image_paths[i], return_attributes=True)
        output = output.float().detach().cpu().numpy()
        pred_score.append(attributes[0])
        
#        output_2, attributes_2 = restoration_inference(model, image_paths_2[i], return_attributes=True)
#        output_2 = output_2.float().detach().cpu().numpy()
#        pred_score_2.append(attributes_2[0])
#        if attributes_2[0] < attributes[0]:
#            count = count + 1 
        
        
    pred_score = np.squeeze(np.array(pred_score))*100
#    pred_score_2 = np.squeeze(np.array(pred_score_2))*100
    print("scores:")
    print(pred_score)
    print(np.mean(pred_score))
    
#    print("target_scores:")
#    print(np.mean(pred_score_2))
    
    print("the number of error:")
    print(count)
    #print(np.mean(pred_score))

if __name__ == '__main__':
    main()
