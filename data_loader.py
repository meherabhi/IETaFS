# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:03:28 2021

@author: eee
"""
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
import torch
import os
import random
import numpy as np
from config import get_config
from collections import Counter

# Defining the custom dataset
class RAVDESS(data.Dataset):
# image_dir: default='data/RAVDESS'
# attr_path: default=''data/RAVDESS/attributes.txt'
# c_dim: No of AU's
# Transform: is a composition of transformations which convert a numpy array to a tensor and then normalizes it
# using its mean and standard deviation
# mode: train or test
    def __init__(self, image_dir, attr_path, transform, mode, c_dim, AU_Train_mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.c_dim = c_dim
        self.AU_Train= AU_Train_mode
        self.train_dataset = []
        self.test_dataset = []

        # Fills train_dataset and test_dataset --> [filename, boolean attribute vector]
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        print("------------------------------------------------")
        print("Training images: ", len(self.train_dataset))
        print("Testing images: ", len(self.test_dataset))

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
# Leave the first 2 lines because it has table specs and column names
        random.seed(1234)
        random.shuffle(lines)
        self.counts = Counter()
        # Extract the info from each line
        for idx, line in enumerate(lines):
# split each AU at spaces
            split = line.split()
# take file name from the first element
            spectrogram_filename = split[0]
# Take the values from element 1 onwards
            label_file_name = split[1] 
            video_frame_filename= split[2]
            # no_frames= len(os.listdir(os.path.join(self.image_dir,video_frame_filename[:-7])))
            self.counts[video_frame_filename[-29:-8]] += 1
            # labels is a Vector representing the presence of each attribute in each image
            # label = []  
            # for n in range(self.c_dim):
            #     label.append(float(label_values[n])/5.)
# Keep the first 100 images in the test dataset
# here test_dataset is an empty list and each of these lines are appended to it
            if idx < 100:
                self.test_dataset.append([spectrogram_filename, label_file_name , video_frame_filename ])
            else:
                self.train_dataset.append([spectrogram_filename, label_file_name , video_frame_filename ])

        print('Dataset ready!...')
# standard __getitem__ function which returns the image along with its label
    def __getitem__(self, index):
        if self.AU_Train:
            dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
            spect_filename, label_filename, videofrm_filename= dataset[index]
            spect_file = torch.from_numpy( np.load(os.path.join(self.image_dir, spect_filename)) ).float()
            label= torch.from_numpy( np.load(os.path.join(self.image_dir,label_filename))).float()
            return  spect_file, torch.FloatTensor(label)
            
        else:
            dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
            spect_filename, label_filename, videofrm_filename= dataset[index]
            spect_file = torch.from_numpy( np.load(os.path.join(self.image_dir, spect_filename)) ).float()
            frame_image = Image.open(os.path.join(self.image_dir, videofrm_filename))
            label= torch.from_numpy( np.load(os.path.join(self.image_dir,label_filename))).float()
            # frame_image= ImageOps.grayscale(frame_image)
            random_frame_no=np.random.randint(0,self.counts[spect_filename[-29:-8]])
            still_image_file_name=os.path.join(videofrm_filename[:-7],"{:03d}".format(random_frame_no)+".jpg")
    
            still_image_label= torch.from_numpy( np.load(os.path.join(self.image_dir,"Labels" + 
                                                                  spect_filename[-29:-8] +"_"+
                                                                  "{:03d}".format(random_frame_no)+".npy"))).float()
    
            still_image= Image.open(os.path.join(self.image_dir,still_image_file_name))
            return  spect_file, self.transform(frame_image) , torch.FloatTensor(label), self.transform(still_image), torch.FloatTensor(still_image_label)

    def __len__(self):
        return self.num_images

# will return a dataloader which can be enumerated
def get_loader(image_dir, attr_path, c_dim, image_size=128,
                batch_size=25, mode='train', num_workers=1, AU_Train_mode='True'):

    transform = []
# Convert a PIL Image or numpy.ndarray to tensor.
    transform.append(T.ToTensor())
# Normalize a tensor image with mean and standard deviation.
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
# Composes several transforms together.
    transform = T.Compose(transform)
    dataset = RAVDESS(image_dir, attr_path, transform, mode, c_dim, AU_Train_mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)

    return data_loader


