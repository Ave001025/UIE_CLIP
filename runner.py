from asyncio.log import logger
from pickletools import read_uint2
import plistlib
from tabnanny import check
from unittest import result
import os
import time
import random
import argparse
import mmcv
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc
from mmedit.datasets.pipelines import Compose
from PIL import ImageEnhance, Image, ImageFilter
import utils
import loss
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torchvision.transforms import Compose as torchcompose

def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)   

class UIE_Runner():
    def __init__(self, options, type='train'):
        manual_seed(114514)
        self.type = type
        self.dataset_opt = options['dataset']
        self.model_opt = options['model']
        self.training_opt = options['train']
        self.experiments_opt = options['experiments']
        self.test_opt = options['test']

        self.model = utils.build_model(self.model_opt)

        self.train_dataloader = utils.build_dataloader(self.dataset_opt, type='train')
        self.test_dataloader = utils.build_dataloader(self.dataset_opt, type='test')
        
        self.optimizer = utils.build_optimizer(self.training_opt, self.model)
        self.lr_scheduler = utils.build_lr_scheduler(self.training_opt, self.optimizer)
        
        self.tb_writer = SummaryWriter(os.path.join(self.experiments_opt['save_root'], 'tensorboard'))
        self.logger = utils.build_logger(self.experiments_opt)
        
        self.my_config = "./CLIP_IQA/configs/clipiqa/clipiqa_coop_koniq.py"
        self.my_checkpoint = "./CLIP_IQA/CLIP_best.pth"
        self.clip_model = init_model(self.my_config, self.my_checkpoint)
        self.clip_model = self.clip_model.cuda()

    def main_loop(self):
        psnr_list = []
        ssim_list = []
        start_epoch = 0
        weights = self.my_curriculum_weight(0)
        my_max_ssim = 0
        my_max_psnr = 0 
        my_max_clip_score = 0
        if self.model_opt['resume_ckpt_path']:
            ckpt = torch.load(self.model_opt['resume_ckpt_path'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            psnr_list.append(ckpt['max_psnr'])
            ssim_list.append(ckpt['max_ssim'])
            start_epoch = ckpt['epoch'] + 1
            for _ in range(start_epoch * 50):
                self.lr_scheduler.step()

        for epoch in range(start_epoch, self.training_opt['epoch']):
            print('================================ %s %d / %d ================================' % (self.experiments_opt['save_root'].split('/')[-1], epoch, self.training_opt['epoch']))
 
            loss = self.train_loop(epoch,weights,self.clip_model)
            print('train from scratch *** ')
            print(
            f'n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|n6_weight:{weights[5]}|inp_weight:{weights[6]}')
  
            torch.cuda.empty_cache()
            psnr, ssim,weights,my_max_psnr,my_max_ssim,my_max_clip_score = self.test_loop(checkpoint_path=None, epoch_num=epoch, max_psnr=my_max_psnr, max_ssim=my_max_ssim,max_clip_score=my_max_clip_score)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            self.logger.info(
                f"Epoch: {epoch}/{self.training_opt['epoch']}\t"
                f"Loss: {loss}\t"
                f"PSNR: {psnr} (max: {np.max(np.array(psnr_list))})\t"
                f"SSIM: {ssim} (max: {np.max(np.array(ssim_list))})\t"
            )
            if np.max(np.array(psnr_list)) == psnr or np.max(np.array(ssim_list)) == ssim:
                self.logger.warning(f"After {epoch+1} epochs trainingg, model achecieves best performance ==> PSNR: {psnr}, SSIM: {ssim}\n")
                self.save(epoch, psnr, ssim)

    def main_test_loop(self):
        if self.test_opt['start_epoch'] >=0 and self.test_opt['end_epoch'] >=0 and self.test_opt['test_ckpt_path'] is None:
            for i in range(self.test_opt['start_epoch'], self.test_opt['end_epoch']):
                checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], f'checkpoint_{i}.pth')
                self.test_loop(checkpoint_name, i)
        else:
            self.test_loop(self.test_opt['test_ckpt_path'])
                 

    def my_curriculum_weight(self,difficulty):
        diff_list = [0.502,0.398,0.399,0.36,0.428,0.46]
        cl_lambda = 0.25   
        weights = [(1 + cl_lambda) if difficulty > x else (1 - cl_lambda) for x in diff_list]
        
        weights.append(len(diff_list))
        new_weights = [i / sum(weights) for i in weights]
        return new_weights
            
    def train_loop(self, epoch_num,weights,clip_model):
        total_loss = 0
        ranker_model = utils.build_model(self.training_opt['ranker_args']) if self.training_opt['loss_rank'] else None
        with tqdm(total=len(self.train_dataloader)) as t_bar:
            for iter_num, data in enumerate(self.train_dataloader):
                # put data to cuda device
                if self.model_opt['cuda']:
                    data = {key:value.cuda() for key, value in data.items()}
                                  
                result = self.model(**data)
                    
                loss = self.build_loss(result, data['gt_img'],data['n1_img'],data['n2_img'],data['n3_img'],data['n4_img'],data['n5_img'],data['n6_img'],data['raw_img'],ranker_model,weights,clip_model)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                total_loss = (total_loss * iter_num + loss) / (iter_num + 1)
                t_bar.set_description('Epoch:%d/%d, loss:%.6f' % (epoch_num, self.training_opt['epoch'], total_loss))
                t_bar.update(1)
                
        self.tb_writer.add_scalar('train/loss', total_loss, epoch_num + 1)
        if self.training_opt['loss_rank']:
            ranker_model=ranker_model.cpu()
        return total_loss
            
    def test_loop(self, checkpoint_path=None, epoch_num=-1,max_psnr = 0,max_ssim = 0,max_clip_score = 0):
        # if checkpoint_path == None:
        #     raise NotImplementedError('checkpoint_name can not be NoneType!')

        with torch.no_grad():
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            clip_meter = AverageMeter()
            if checkpoint_path:
                ckpt_dict = torch.load(checkpoint_path)['net']
                self.model.load_state_dict(ckpt_dict)
            if self.test_opt['save_img']:
                save_root = os.path.join(self.experiments_opt['save_root'], 'results')
                utils.make_dir(save_root)


            with tqdm(total=len(self.test_dataloader)) as t_bar:
                for iter_num, data in enumerate(self.test_dataloader):
                    _, _, h, w = data['gt_img'].shape
                    gt_img = data['gt_img'][0].permute(1, 2, 0).detach().numpy()
                    if self.model_opt['cuda']:
                        data = {key:value.cuda() for key, value in data.items()}

                    upsample = nn.UpsamplingBilinear2d((h, w))
                    pred_img = upsample(utils.normalize_img(self.model(**data)))
                    
                    tmp_img = pred_img
                    clip_score = self.clip_model.forward_test(lq = pred_img)['attributes']
                    pred_img = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    if self.test_opt['save_img']:
                        cv2.imwrite(os.path.join(self.experiments_opt['save_root'], 'results', str(iter_num)+'.png'), pred_img[:, :, ::-1] * 255.0)

                    psnr = utils.calc_psnr(pred_img, gt_img, is_for_torch=False)
                    ssim = utils.calc_ssim(pred_img, gt_img, is_for_torch=False)


                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)
                    clip_meter.update(clip_score)


                    if checkpoint_path:
                        t_bar.set_description('checkpoint: %s, psnr:%.6f, ssim:%.6f' % (checkpoint_path.split('/')[-1], psnr_meter.avg, ssim_meter.avg))
                    if epoch_num >= 0:
                        t_bar.set_description('Epoch:%d/%d, psnr:%.6f, ssim:%.6f' % (epoch_num, self.training_opt['epoch'], psnr_meter.avg, ssim_meter.avg))

                    t_bar.update(1)
                    
        if epoch_num >= 0:
            self.tb_writer.add_scalar('valid/psnr', psnr_meter.avg, epoch_num + 1)
            self.tb_writer.add_scalar('valid/ssim', ssim_meter.avg, epoch_num + 1)
            
        max_ssim = max(max_ssim, ssim_meter.avg)
        max_psnr = max(max_psnr, psnr_meter.avg)
        max_clip_score = max(clip_meter.avg, max_clip_score)
        weights = self.my_curriculum_weight(max_clip_score)
        
        return psnr_meter.avg, ssim_meter.avg,weights,max_psnr,max_ssim,max_clip_score
            
    def save(self, epoch_num, psnr, ssim):
        # path for saving
        path = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'])
        utils.make_dir(path)
            
        checkpoint = {
        "net": self.model.state_dict(),
        'optimizer':self.optimizer.state_dict(),
        "epoch": epoch_num,
        "max_psnr": psnr,
        "max_ssim": ssim
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_{epoch_num}.pth'))
    
    def build_loss(self, pred, gt,n1,n2,n3,n4,n5,n6,raw, ranker_model,weights,clip_model):
        loss_total = 0
        Loss_L1 = nn.L1Loss().cuda()
        loss_total = loss_total + self.training_opt['loss_coff'][0] * Loss_L1(pred, gt)

        if self.training_opt['loss_vgg']:
            Loss_VGG = loss.make_perception_loss(self.training_opt.get('loss_vgg_args')).cuda()
            loss_total = loss_total + self.training_opt['loss_coff'][1] * Loss_VGG(pred, gt)
            
        if self.training_opt['loss_rank']:
            loss_total = loss_total + self.training_opt['loss_coff'][2] * loss.ranker_loss(ranker_model, pred)
            
        if self.training_opt['loss_contrast']:
            contrastloss = loss.ContrastLoss(ablation=False)
            loss_total = loss_total + self.training_opt['loss_coff'][3] * contrastloss(pred,gt,raw)
            
        if self.training_opt['loss_c2r']:
            c2r_loss = loss.C2R(ablation=False)
            loss1 = self.training_opt['loss_coff'][4] * c2r_loss(pred,gt,n1,n2,n3,n4,n5,n6,raw,weights) 
            loss_total = loss_total + loss1 
            
        if self.training_opt['loss_clip']:
            loss_total = loss_total + self.training_opt['loss_coff'][5] * loss.clip_loss(clip_model, pred,gt)
            
        return loss_total

          
    

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger():
    def __init__(self, experiments_opt):
        self.expreiments_opt = experiments_opt
        self.loss_list = []
        self.srocc_list = []
        self.acc_list = []
        self.path = os.path.join(self.expreiments_opt['logger_root'], self.expreiments_opt['exp_name'])
        
    def write(self, loss, srocc, acc, epoch_num):
        self.loss_list.append(loss)
        self.srocc_list.append(srocc)
        self.acc_list.append(acc)
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            
        # draw loss and score 
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.path, 'loss.png'))
        plt.clf()
        plt.plot(self.srocc_list)
        plt.plot(self.acc_list)
        plt.savefig(os.path.join(self.path, 'scores.png'))
        plt.clf()
        
        # saving log
        if np.max(np.array(self.srocc_list)) == srocc:
            with open(os.path.join(self.path, 'log.txt'), 'a') as f:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write('%s saving best checkpoint: Ecpoh:%d loss:%.6f, SROCC:%.3f, ACC:%.3f\n' % (now, epoch_num, loss, srocc, acc))
            return True
        else:
            return False