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

import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose
from PIL import ImageEnhance, Image, ImageFilter
from torchvision.transforms import Compose as torchcompose
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_coop_koniq.py', help='test config file path')
    parser.add_argument('--checkpoint', default="/data/zengzekai/UIE/Codes/CLIP-IQA-2-3.8/work_dirs/clipiqa_coop_koniq_224_Clear_Turbid_random_best/iter_10000.pth", help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))
    #t = torch.tensor((3,256,256))
    #data = torch.randn(8,3,256,256)
    to_tensor = transforms.ToTensor()
    img = Image.open("/data/zengzekai/UIE/Codes/test01_uranker/examples/ranker_example/400_img_.png")
    data = to_tensor(img)
    data = data.unsqueeze(0)
    
    device = next(model.parameters()).device  # model device
    #print(device)
    data = data.to(device)
    with torch.no_grad():
        # result = model(test_mode=True, *t)
        result = model.forward_test(lq=data)
    print(result['attributes'])


if __name__ == '__main__':
    main()

