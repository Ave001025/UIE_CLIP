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
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default="/data/zengzekai/UIE/Datasets/koniq10k/1024x768/", help='path to input image file')
    parser.add_argument('--csv_path', default="/data/zengzekai/UIE/Datasets/koniq10k/koniq10k_distributions_sets.csv", help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

#要使用这部分代码，首先需要有标签值（mos值），这是重要的，这个数据集采用的实csv文件来进行读取
#如果使用csv文件来读取也算是比较简单的
#如果使用Ranker排序话，要使用什么数据集么，这个我就有点不太懂了，这里需要好好思考一下
#因为里面计算质量指标的SROCC还有PLCC都是需要参考分数来进行计算的


def main():
    args = parse_args()

    #初始化模型配置
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    #读取csv文件
    csv_list = pd.read_csv(args.csv_path, on_bad_lines='skip')
    #取出csv文件中的测试文件那几行
    img_test = csv_list[csv_list.set=='test'].reset_index()

    txt_path = './koniq_resize.txt'
    #对应的标签值（mos代表图像质量分数，百分制）
    y_true = csv_list[csv_list.set=='test'].MOS.values

    pred_score = []
    for i in tqdm(range(len(img_test))):
        #文件路径，不知道这里返回的是什么，但是attributes[0]就是得到的分数值
        output, attributes = restoration_inference(model, os.path.join(args.file_path, img_test['image_name'][i]), return_attributes=True)
        output = output.float().detach().cpu().numpy()
        pred_score.append(attributes[0])

    #删除维数为1的维度，使用softmax得到的时概率值，需要乘于100才能得到分数值
    pred_score = np.squeeze(np.array(pred_score))*100

    #计算两者的相关系数
    p_srocc = srocc(pred_score, y_true)
    p_plcc = plcc(pred_score, y_true)

    print(args.checkpoint)
    print('SRCC: {} | PLCC: {}'.\
          format(p_srocc, p_plcc))


if __name__ == '__main__':
    main()
