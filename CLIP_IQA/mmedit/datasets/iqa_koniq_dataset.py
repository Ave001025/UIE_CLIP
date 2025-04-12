# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

import pandas as pd
import scipy.io
import numpy as np
import os
import random

#不同数据集读取格式，返回图像和对应的MOS分数
#原本的数据读取格式
#@DATASETS.register_module()
#class IQAKoniqDataset(BaseSRDataset):
#
#    def __init__(self,
#                 img_folder,
#                 pipeline,
#                 ann_file,
#                 scale=1,
#                 test_mode=False):
#        super().__init__(pipeline, scale, test_mode)
#        self.img_folder = str(img_folder)
#        self.ann_file = pd.read_csv(ann_file, error_bad_lines=True)
#        if test_mode:
#            self.data_infos = self.ann_file[self.ann_file.set=='test'].reset_index()
#            self.gt_labels = self.ann_file[self.ann_file.set=='test'].MOS.values
#        else:
#            self.data_infos = self.ann_file[self.ann_file.set=='training'].reset_index()
#            self.gt_labels = self.ann_file[self.ann_file.set=='training'].MOS.values
#
#    def load_annotations(self):
#        return 0
#
#    def __getitem__(self, idx):
#        """Get item at each call.
#        Args:
#            idx (int): Index for getting each item.
#        """
#        results = dict(
#            lq_path=osp.join(self.img_folder, self.data_infos['image_name'][idx]),
#            gt=self.gt_labels[idx]/100)
#        results['scale'] = self.scale
#        return self.pipeline(results)


#---------------------------------------------------------------
@DATASETS.register_module()
class IQAKoniqDataset(BaseSRDataset):

    def __init__(self,
                 img_folder,
                 pipeline,
                 ann_file,
                 scale=1,
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.img_folder = str(img_folder)
        #self.ann_file = pd.read_csv(ann_file, error_bad_lines=True)
        # random.seed(1)
        # index = list(range(0, 8000))
        # random.shuffle(index)
        # train_index = index[0:7200]
        # test_index = index[7200:8000]
        # with open('test_id.txt','w') as f :
        #     for i in test_index:
        #         f.write("%s\n" % i)
        # with open('train_id.txt','w') as f:
        #     for i in train_index:
        #         f.write("%s\n" % i)
        with open('/data/zengzekai/UIE/Codes/CLIP-IQA-2-3.8/my_train_id_best.txt') as f:
            train_index = [int(line.rstrip()) for line in f]
        
        with open('/data/zengzekai/UIE/Codes/CLIP-IQA-2-3.8/my_test_id_best.txt') as f:
            test_index = [int(line.rstrip()) for line in f]
        
        
        if test_mode:
            self.data_infos = scipy.io.loadmat(ann_file)['path'][test_index]
            self.gt_labels = scipy.io.loadmat(ann_file)['mos'][test_index]
        else:
#            self.data_infos = scipy.io.loadmat(ann_file)['path'][0:4200]
#            self.data_infos_1 = scipy.io.loadmat(ann_file)['path'][5000:8000]
#            self.data_infos = np.concatenate((self.data_infos,self.data_infos_1),axis = 0)
#            self.gt_labels = scipy.io.loadmat(ann_file)['mos'][0:4200]
#            self.gt_labels_1 = scipy.io.loadmat(ann_file)['mos'][5000:8000]
#            self.gt_labels = np.concatenate((self.gt_labels,self.gt_labels_1),axis = 0)
            self.data_infos = scipy.io.loadmat(ann_file)['path'][train_index]
            self.gt_labels = scipy.io.loadmat(ann_file)['mos'][train_index]
              
              

    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        results = dict(
            lq_path=self.img_folder + self.data_infos[idx][0][0],
            gt=np.float32(self.gt_labels[idx]/100))

        results['scale'] = self.scale
        return self.pipeline(results)
#------------------------------------------

@DATASETS.register_module()
class IQALIVEITWDataset(BaseSRDataset):

    def __init__(self,
                 img_folder,
                 pipeline,
                 file_path,
                 scale=1,
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)
        self.img_folder = str(img_folder)
        self.data_infos = scipy.io.loadmat(osp.join(file_path, 'AllImages_release.mat'))['AllImages_release'][7:]
        self.gt_labels = scipy.io.loadmat(osp.join(file_path, 'AllMOS_release.mat'))['AllMOS_release'][0][7:]
        # self.std_mat = scipy.io.loadmat(os.path.join(file_path, 'AllStdDev_release.mat'))


    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        results = dict(
            lq_path=osp.join(self.img_folder, self.data_infos[idx][0][0]),
            gt=self.gt_labels[idx]/100)
        results['scale'] = self.scale
        return self.pipeline(results)

@DATASETS.register_module()
class IQAAVADataset(BaseSRDataset):

    def __init__(self,
                 img_folder,
                 pipeline,
                 file_path,
                 scale=1,
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)
        self.img_folder = str(img_folder)
        if test_mode:
            self.data_infos = np.loadtxt(os.path.join(file_path, 'test_ava_name.txt'), dtype=int)
            self.gt_labels = np.loadtxt(os.path.join(file_path, 'test_ava_score.txt'), dtype=float)[:, 0]
        else:
            self.data_infos = np.loadtxt(os.path.join(file_path, 'train_ava_name.txt'), dtype=int)
            self.gt_labels = np.loadtxt(os.path.join(file_path, 'train_ava_score.txt'), dtype=float)[:, 0]
        # self.std_mat = scipy.io.loadmat(os.path.join(file_path, 'AllStdDev_release.mat'))


    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
        results = dict(
            lq_path=osp.join(self.img_folder, str(self.data_infos[idx])+'.jpg'),
            gt=self.gt_labels[idx]/100)
        results['scale'] = self.scale
        return self.pipeline(results)
