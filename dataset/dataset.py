import imghdr
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as transFunc
import torchvision.transforms.functional as F

import os
import yaml
import numpy as np
from PIL import Image
from random import Random
import matplotlib.pyplot as plt
from utils import build_historgram
import os.path as osp


myrandom = Random(567)
class Dataset_UIEB(data.Dataset):
    def __init__(self, opt, type="train"):
        self.raw_path = opt['root'] + 'raw-890/'
        self.gt_path = opt['root'] + 'reference-890/'
        
        self.n1_path = "/data/zengzekai/UIE/Datasets/UIEB/train/GC_res/"
        self.n2_path = "/data/zengzekai/UIE/Datasets/UIEB/train/HE_res/"
        self.n3_path = "/data/zengzekai/UIE/Datasets/UIEB/train/IBLA_res/"
        self.n4_path = "/data/zengzekai/UIE/Datasets/UIEB/train/UDCP_res/"
        self.n5_path = "/data/zengzekai/UIE/Datasets/UIEB/train/FUnIE_res/"
        self.n6_path = "/data/zengzekai/UIE/Datasets/UIEB/train/USUIR_res/"
        self.crop_size = opt['crop_size']
        self.type = type
        if self.type == "train":
            f = open(opt['train_list_path'])
        elif self.type == "test" or self.type == 'valid':
            f = open(opt['test_list_path'])
        self.filenames = f.readlines()

    def __getitem__(self, item):
        raw_item_path = self.raw_path + self.filenames[item].rstrip()
        gt_item_path = self.gt_path + self.filenames[item].rstrip()
              
        raw_img = Image.open(raw_item_path)
        gt_img = Image.open(gt_item_path)        
        
        img_w = raw_img.size[0]
        img_h = raw_img.size[1]

        if self.type == "train":
            # random resize and crop to 256 x 256
            n1_item_path = self.n1_path + self.filenames[item].rstrip()
            n2_item_path = self.n2_path + self.filenames[item].rstrip()
            n3_item_path = self.n3_path + self.filenames[item].rstrip()
            n4_item_path = self.n4_path + self.filenames[item].rstrip()
            n5_item_path = self.n5_path + self.filenames[item].rstrip()
            n6_item_path = self.n6_path + self.filenames[item].rstrip()
            
            n1_img = Image.open(n1_item_path)
            n2_img = Image.open(n2_item_path)
            n3_img = Image.open(n3_item_path)
            n4_img = Image.open(n4_item_path)
            n5_img = Image.open(n5_item_path)
            n6_img = Image.open(n6_item_path)
        
            i, j, h, w = transforms.RandomResizedCrop(self.crop_size).get_params(raw_img, (0.08, 1.0),
                                                                                  (3. / 4., 4. / 3.))
            raw_cropped = F.resized_crop(raw_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            gt_cropped = F.resized_crop(gt_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            raw_cropped = transforms.ToTensor()(raw_cropped)
            gt_cropped = transforms.ToTensor()(gt_cropped)
            
            n1_cropped = F.resized_crop(n1_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            n2_cropped = F.resized_crop(n2_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            n3_cropped = F.resized_crop(n3_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            n4_cropped = F.resized_crop(n4_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            n5_cropped = F.resized_crop(n5_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            n6_cropped = F.resized_crop(n6_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            
            n1_cropped = transforms.ToTensor()(n1_cropped)
            n2_cropped = transforms.ToTensor()(n2_cropped)
            n3_cropped = transforms.ToTensor()(n3_cropped)
            n4_cropped = transforms.ToTensor()(n4_cropped)
            n5_cropped = transforms.ToTensor()(n5_cropped)
            n6_cropped = transforms.ToTensor()(n6_cropped)
            
            
            if np.random.rand(1) < 0.5:  # flip horizonly
                raw_cropped = torch.flip(raw_cropped, [2])
                gt_cropped = torch.flip(gt_cropped, [2])
                
                n1_cropped = torch.flip(n1_cropped, [2])
                n2_cropped = torch.flip(n2_cropped, [2])
                n3_cropped = torch.flip(n3_cropped, [2])
                n4_cropped = torch.flip(n4_cropped, [2])
                n5_cropped = torch.flip(n5_cropped, [2])
                n6_cropped = torch.flip(n6_cropped, [2])
    
            if np.random.rand(1) < 0.5:  # flip vertically
                raw_cropped = torch.flip(raw_cropped, [1])
                gt_cropped = torch.flip(gt_cropped, [1])
                
                n1_cropped = torch.flip(n1_cropped, [1])
                n2_cropped = torch.flip(n2_cropped, [1])
                n3_cropped = torch.flip(n3_cropped, [1])
                n4_cropped = torch.flip(n4_cropped, [1])
                n5_cropped = torch.flip(n5_cropped, [1])
                n6_cropped = torch.flip(n6_cropped, [1])                        

            return {'raw_img':raw_cropped, 'gt_img':gt_cropped, 'n1_img':n1_cropped,'n2_img':n2_cropped,'n3_img':n3_cropped,'n4_img':n4_cropped,'n5_img':n5_cropped,'n6_img':n6_cropped, }
        
        elif self.type == "test":
            raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)
            raw_img = transforms.ToTensor()(raw_img)
            gt_img = transforms.ToTensor()(gt_img)

            return {'raw_img':raw_img, 'gt_img':gt_img}
            
    def __len__(self):
        return len(self.filenames)

