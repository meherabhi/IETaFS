# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:54:17 2021

@author: eee
"""

import torch
# import torch.nn.functional as F
# from torchvision.utils import save_image, make_grid

from utils2 import Utils

import numpy as np

# import os
import time
import datetime
# import random
# import glob


class Solver2(Utils):

    def __init__(self, data_loader, config_dict):
        # NOTE: the following line create new class arguments with the
        # values in config_dict
        self.__dict__.update(**config_dict)
# the data_loader can be enumerated
        self.data_loader = data_loader
        
        self.device = 'cuda:' + \
            str(self.gpu_id) if torch.cuda.is_available() else 'cpu'
        print(f"Model running on {self.device}")
#Build a tensorboard logger
#It maintains a logger of the events and uses the tensorboard UI for display
        if self.use_tensorboard:
            self.build_tensorboard()

        self.loss_visualization = {}
# Create a generator and a discriminator.
# Loads the models to the device and initializes their optimizers
        self.build_model()

    def train(self):
        print('Training...')

        self.global_counter = 0
# If the training has to restart, load the current iteration and the previosuly trained models
# else make the first_iter no=0
        if self.resume_iters:
            self.first_iteration = self.resume_iters
            self.restore_model(self.resume_iters)
        else:
            self.first_iteration = 0
# Set the time 
        self.start_time = time.time()

        for epoch in range(self.first_epoch, self.num_epochs_AU_train):
            print(f"EPOCH {epoch} WITH {len(self.data_loader)} STEPS")

            self.alpha_rec = 1
            self.epoch = epoch
            for iteration in range(self.first_iteration, len(self.data_loader)):
                self.iteration = iteration
                torch.cuda.empty_cache()
                self.get_training_data()
                self.AU_training()
            
# After how many interations is the model saved
                if self.iteration % self.model_save_step == 0:
                    self.save_models(self.iteration, self.epoch)
# After how many steps are the events logged
                if self.iteration % self.log_step == 0:
                    self.update_tensorboard()
                self.global_counter += 1

            # Save the last model
            self.save_models(self.iteration, self.epoch)

            self.first_iteration = 0  # Next epochs start from 0

# This function will load self.x_real with the inout images
# self.label_org with the inout image AU's
# self.label_trg with the randomly generated target AU vectors
# self.label_trg_virtual: labels for virtual cycle consistency
    def get_training_data(self):
        try:
            self.spect_image, self.label_org  = next(self.data_iter)
        except:
            #iter() function returns an iterator for the given object
            self.data_iter = iter(self.data_loader)
            # From the iterator take the input and the corresponding label
            #next() function returns the next item in an iterator.
            self.spect_image, self.label_org = next(self.data_iter)

        
        self.label_org = self.label_org.to(self.device)
        self.spect_image= self.spect_image.to(self.device)


    def std_normalized_l2_loss(self,output, target):
        std_inv = np.array([6.6864805402, 5.2904440280, 3.7165409939, 4.1421640454, 8.1537399389, 7.0312877415, 2.6712380967,
                            2.6372177876, 8.4253649884, 6.7482162880, 9.0849960354, 10.2624412692, 3.1325531319, 3.1091179819,
                            2.7337937590, 2.7336441031, 4.3542467871, 5.4896293687, 6.2003761588, 3.1290341469, 5.7677042738,
                            11.5460919611, 9.9926451700, 5.4259818848, 20.5060642486, 4.7692101480, 3.1681517575, 3.8582905289,
                            3.4222250436, 4.6828286809, 3.0070785113, 2.8936539301, 4.0649030157, 25.3068458731, 6.0030623160,
                            3.1151977458, 7.7773542649, 6.2057372469, 9.9494258692, 4.6865422850, 5.3300697628, 2.7722027974,
                            4.0658663003, 18.1101618617, 3.5390113731, 2.7794520068], dtype=np.float32)
        std_inv = std_inv[[0,1,3,4,5,6,8,9,11,13,14,16,19,22,24,25,44]]
        weights =torch.from_numpy(std_inv).to(target) #.reshape((1, label_dim)))
        dif = output - target
        ret = torch.mean(torch.div(dif, weights)**2)
        return ret
    
    def AU_training(self):
        pred_label=self.AUnet(self.spect_image)
        ls=self.std_normalized_l2_loss(pred_label, self.label_org)
        self.reset_grad()
        ls.backward()
        self.AU_optimizer.step()
        self.loss_visualization['AU/loss'] = ls.item()
        return
        
    def update_tensorboard(self):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}],  [{}/{}], Epoch [{}/{}]".format(
            et, self.iteration+1, len(self.data_loader), self.epoch+1, self.num_epochs_AU_train)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in self.loss_visualization.items():
                self.writer.add_scalar(
                    tag, value, global_step=self.global_counter)

