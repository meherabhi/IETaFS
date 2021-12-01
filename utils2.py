# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:55:49 2021

@author: eee
"""

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import Generator
from model import Discriminator
from model import audio_network
from model import video_network
from model import AU_Torch

import os
import re


class Utils:

    def build_model(self):
        """Create a generator and a discriminator."""
        self.AUnet= AU_Torch().to(self.device)
        self.AU_optimizer=torch.optim.Adam(self.AUnet.parameters(), lr=0.0001 , betas= (0.9, 0.999), 
                                           eps=1e-08, weight_decay=0.0001 , amsgrad=False)
        # TODO: implement data parallelization for multiple gpus
        # self.gpu_ids = torch.cuda.device_count()
        # print("GPUS AVAILABLE: ", self.gpu_ids)
        # if self.gpu_ids > 1:
        #     torch.nn.DlataParalle(self.D, device_ids=list(range(self.gpu_ids)))
        #     torch.nn.DataParallel(self.G, device_ids=list(range(self.gpu_ids)))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        #TensorBoard, these utilities let you log PyTorch models and metrics into
        # a directory for visualization within the TensorBoard UI.
        from logger import Logger
        #This module defines functions and classes which implement a flexible event logging system
        #for applications and libraries
        self.logger = Logger(self.log_dir)
        #SummaryWriter class is your main entry to log data for consumption and visualization by TensorBoard
        self.writer = SummaryWriter(logdir=self.log_dir)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.AU_optimizer.zero_grad()
    

    def save_models(self, iteration, epoch):
        try:  # To avoid crashing on the first step

            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-AU.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-AU_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))

        except:
            pass


        AU_path= os.path.join(self.model_save_dir,'{}-{}-AU.ckpt'.format(iteration+1, epoch))
        torch.save(self.AUnet.state_dict(),AU_path)
        AU_path_optim=os.path.join(
            self.model_save_dir, '{}-{}-AU_optim.ckpt'.format(iteration+1, epoch))
        torch.save(self.AU_optimizer.state_dict(), AU_path_optim)
        print(f'Saved model checkpoints in {self.model_save_dir}...')
# Loads the pre trained G and D networks and their corresponding optimizers
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}-{}...'.format(resume_iters, self.first_epoch))
        AU_path= os.path.join(self.model_save_dir,'{}-{}-AU.ckpt'.format(resume_iters, self.first_epoch))
        self.AUnet.load_state_dict(torch.load(
            AU_path, map_location=lambda storage, loc: storage))
        AU_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-AU_optim.ckpt'.format(resume_iters, self.first_epoch))
        self.AU_optimizer.load_state_dict(torch.load(AU_optim_path))

    def numericalSort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
