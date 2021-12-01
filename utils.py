import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import Generator
from model import Discriminator
from model import audio_network
from model import video_network
from model import AU_Torch
# from model import Color_D

import cv2 , os
import numpy as np
from skimage import transform
import torch
import matplotlib.pyplot as plt
import os
import re
from HRNet.tools import test


class Utils:

    def build_model(self):
        """Create a generator and a discriminator."""
        # G_path= os.path.join(self.model_save_dir,'{}-{}-G.ckpt'.format(7001,37))
        self.G = Generator(self.g_conv_dim, self.c_dim,
                           self.g_repeat_num).to(self.device)
        
        # self.G.load_state_dict(torch.load(
        #     G_path, map_location=lambda storage, loc: storage))
        self.D = Discriminator(
            self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num).to(self.device) 
        
        # self.Color_D = Color_D().to(self.device) 
        
        self.audionet= audio_network().to(self.device)
        self.videonet= video_network().to(self.device)    
        AU_path= os.path.join(self.model_save_dir,'{}-{}-AU.ckpt'.format(990, self.num_epochs_AU_train-1))
        self.AUnet= AU_Torch().to(self.device)
        self.AUnet.load_state_dict(torch.load(
            AU_path, map_location= self.device))
        self.AU_optimizer=torch.optim.Adam(self.AUnet.parameters(), lr=0.00001 , betas= (0.9, 0.999), 
                                           eps=1e-08, weight_decay=0.0001 , amsgrad=False)
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        # self.Color_d_optimizer = torch.optim.Adam(
        #     self.Color_D.parameters(), lr = 2e-04, betas = (0, 0.999)) 
                
        self.audio_optimizer=  torch.optim.Adam(
            self.audionet.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.video_optimizer=  torch.optim.Adam(
            self.videonet.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        # self.HRNet= test.main()
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

    def smooth_loss(self, att):
        return torch.mean(torch.mean(torch.abs(att[:, :, :, :-1] - att[:, :, :, 1:])) +
                          torch.mean(torch.abs(att[:, :, :-1, :] - att[:, :, 1:, :])))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.video_optimizer.zero_grad()
        self.audio_optimizer.zero_grad()
        self.AU_optimizer.zero_grad()
        # self.Color_d_optimizer.zero_grad()
        

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp(0, 1)


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def imFromAttReg(self, att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real

    def create_labels(self, data_iter):
        """Return samples for visualization"""
        x, c = [], []
        x_data, c_data = data_iter.next()

        for i in range(self.num_sample_targets):
            x.append(x_data[i].repeat(
                self.batch_size, 1, 1, 1).to(self.device))
            c.append(c_data[i].repeat(self.batch_size, 1).to(self.device))

        return x, c

    def save_models(self, iteration, epoch):
        try:  # To avoid crashing on the first step
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-G.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-D.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            
            # os.remove(os.path.join(self.model_save_dir,
            #                         '{}-{}-Color_D.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-AU.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-audionet.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-videonet.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-G_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-D_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-AU_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-audionet_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-videonet_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            
            # os.remove(os.path.join(self.model_save_dir,
            #                         '{}-{}-Color_d_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            
        except:
            pass

        G_path = os.path.join(self.model_save_dir,
                              '{}-{}-G.ckpt'.format(iteration+1, epoch))
        D_path = os.path.join(self.model_save_dir,
                              '{}-{}-D.ckpt'.format(iteration+1, epoch))
        
        # Color_D_path = os.path.join(self.model_save_dir,
        #                       '{}-{}-Color_D.ckpt'.format(iteration+1, epoch))
        
        AU_path= os.path.join(self.model_save_dir,'{}-{}-AU.ckpt'.format(iteration+1, epoch))
        audio_path= os.path.join(self.model_save_dir,'{}-{}-audionet.ckpt'.format(iteration+1, epoch))
        video_path= os.path.join(self.model_save_dir,'{}-{}-videonet.ckpt'.format(iteration+1, epoch))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        
        # torch.save(self.Color_D.state_dict(), Color_D_path)
        
        torch.save(self.AUnet.state_dict(),AU_path)
        torch.save(self.audionet.state_dict(),audio_path)
        torch.save(self.videonet.state_dict(),video_path)
        G_path_optim = os.path.join(
            self.model_save_dir, '{}-{}-G_optim.ckpt'.format(iteration+1, epoch))
        D_path_optim = os.path.join(
            self.model_save_dir, '{}-{}-D_optim.ckpt'.format(iteration+1, epoch))
        
        # Color_D_path_optim = os.path.join(
        #     self.model_save_dir, '{}-{}-Color_d_optim.ckpt'.format(iteration+1, epoch))
        
        AU_path_optim=os.path.join(
            self.model_save_dir, '{}-{}-AU_optim.ckpt'.format(iteration+1, epoch))
        audionet_path_optim=os.path.join(
            self.model_save_dir, '{}-{}-audionet_optim.ckpt'.format(iteration+1, epoch))
        videonet_path_optim=os.path.join(
            self.model_save_dir, '{}-{}-videonet_optim.ckpt'.format(iteration+1, epoch))
        torch.save(self.g_optimizer.state_dict(), G_path_optim)
        
        # torch.save(self.Color_d_optimizer.state_dict(), Color_D_path_optim)
        
        torch.save(self.AU_optimizer.state_dict(), AU_path_optim)
        torch.save(self.audio_optimizer.state_dict(), audionet_path_optim)
        torch.save(self.video_optimizer.state_dict(), videonet_path_optim)
        torch.save(self.d_optimizer.state_dict(), D_path_optim)

        print(f'Saved model checkpoints in {self.model_save_dir}...')
# Loads the pre trained G and D networks and their corresponding optimizers
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}-{}...'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        G_path = os.path.join(
            self.model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        D_path = os.path.join(
            self.model_save_dir, '{}-{}-D.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        
        # Color_D_path = os.path.join(
        #     self.model_save_dir, '{}-{}-Color_D.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        
        AU_path= os.path.join(self.model_save_dir,'{}-{}-AU.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        audio_path= os.path.join(self.model_save_dir,'{}-{}-audionet.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        video_path= os.path.join(self.model_save_dir,'{}-{}-videonet.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        self.G.load_state_dict(torch.load(
            G_path, map_location=self.device))
        self.D.load_state_dict(torch.load(
            D_path, map_location=self.device ))
        
        # self.Color_D.load_state_dict(torch.load(
        #     Color_D_path, map_location=self.device ))
        
        self.AUnet.load_state_dict(torch.load(
            AU_path, map_location= self.device ))
        self.audionet.load_state_dict(torch.load(
            audio_path,  map_location= self.device))
        self.videonet.load_state_dict(torch.load(
            video_path,  map_location= self.device))
        
        G_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-G_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        D_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-D_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        
        # Color_D_optim_path = os.path.join(
        #     self.model_save_dir, '{}-{}-Color_d_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        
        AU_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-AU_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        audionet_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-audionet_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        videonet_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-videonet_optim.ckpt'.format(resume_iters, self.first_epoch+self.num_epochs_AU_train))
        self.d_optimizer.load_state_dict(torch.load(D_optim_path))
        
        # self.Color_d_optimizer.load_state_dict(torch.load(Color_D_optim_path))
        
        self.g_optimizer.load_state_dict(torch.load(G_optim_path))
        self.AU_optimizer.load_state_dict(torch.load(AU_optim_path))
        self.audio_optimizer.load_state_dict(torch.load(audionet_optim_path))
        self.video_optimizer.load_state_dict(torch.load(videonet_optim_path))

    def numericalSort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def optimizer_to(self,optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
    
    
    # def getFaceRect(self, im, faceDetector):
    #     faceRects = faceDetector(im, 0)
    #     if len(faceRects)>0:
    #         faceRect = faceRects[0]
    #         newRect = dlib.rectangle(int(faceRect.left()),int(faceRect.top()), 
    #                              int(faceRect.right()),int(faceRect.bottom()))
    #     else:
    #         newRect = dlib.rectangle(0,0,im.shape[1],im.shape[0])
    #     return newRect
    
    # def landmarks2numpy(self, landmarks_init):
    #     landmarks = landmarks_init.parts()
    #     points = []
    #     for ii in range(0, len(landmarks)):
    #         points.append([landmarks[ii].x, landmarks[ii].y])
    #     return np.array(points)
    
    # # Estimate the similarity transformation of landmarks_frame to landmarks_im
    # # Uses 3 points: a tip of the nose and corners of the eyes
    # def getRigidAlignment(self, landmarks_frame, landmarks_im):
    #     # video frame
    #     video_lmk = [[np.int(landmarks_frame[30][0]), np.int(landmarks_frame[30][1])], 
    #                  [np.int(landmarks_frame[36][0]), np.int(landmarks_frame[36][1])],
    #                  [np.int(landmarks_frame[45][0]), np.int(landmarks_frame[45][1])] ]
    
    #     # Corners of the eye in normalized image
    #     img_lmk = [ [np.int(landmarks_im[30][0]), np.int(landmarks_im[30][1])], 
    #                 [np.int(landmarks_im[36][0]), np.int(landmarks_im[36][1])],
    #                 [np.int(landmarks_im[45][0]), np.int(landmarks_im[45][1])] ]
    
    #     # Calculate similarity transform
    #     #tform = cv2.estimateRigidTransform(np.array([video_lmk]), np.array([img_lmk]), False)
    #     tform = transform.estimate_transform('similarity', np.array(video_lmk), np.array(img_lmk)).params[:2,:]
    #     return tform  
    
    # def getTriMaskAndMatrix(self, im,srcPoints,dstPoints,dt_im):
    #     maskIdc = np.zeros((im.shape[0],im.shape[1],3), np.int32)
    #     matrixA = np.zeros((2,3, len(dt_im)), np.float32)
    #     for i in range(0, len(dt_im)):
    #         t_src = []
    #         t_dst = []
    #         for j in range(0, 3):
    #             t_src.append((srcPoints[dt_im[i][j]][0], srcPoints[dt_im[i][j]][1]))
    #             t_dst.append((dstPoints[dt_im[i][j]][0], dstPoints[dt_im[i][j]][1]))        
    #         # get an inverse transformatio: from t_dst to t_src
    #         Ai_temp = cv2.getAffineTransform(np.array(t_dst, np.float32), np.array(t_src, np.float32))
    #         matrixA[:,:,i] = Ai_temp - [[1,0,0],[0,1,0]]
    #         # fill in a mask with triangle number
    #         cv2.fillConvexPoly(img=maskIdc, points=np.int32(t_dst), color=(i,i,i), lineType=8, shift=0) 
    #     return (maskIdc,matrixA)
    
    # # Smoothes the warp field (offsets) depending oon the distance from a face
    # def smoothWarpField(self, dstPoints, warpField):
    #     warpField_blured = warpField.copy()
    
    #     # calculate facial mask
    #     faceHullIndex = cv2.convexHull(np.array(dstPoints[:68,:]), returnPoints=False)
    #     faceHull = []
    #     for i in range(0, len(faceHullIndex)):
    #         faceHull.append((dstPoints[faceHullIndex[i],0], dstPoints[faceHullIndex[i],1]))
    #     maskFace = np.zeros((warpField.shape[0],warpField.shape[1],3), dtype=np.uint8)  
    #     cv2.fillConvexPoly(maskFace, np.int32(faceHull), (255, 255, 255))     
    
    #     # get distance transform
    #     dist_transform = cv2.distanceTransform(~maskFace[:,:,0], cv2.DIST_L2,5)
    #     max_dist = dist_transform.max()
    
    #     # initialize a matrix with distance ranges ang sigmas for smoothing
    #     maxRadius = 0.05*np.linalg.norm([warpField.shape[0], warpField.shape[1]]) # 40 pixels for 640x480 image
    #     thrMatrix = [[0, 0.1*max_dist, 0.1*maxRadius],
    #                  [0.1*max_dist, 0.2*max_dist, 0.2*maxRadius],
    #                  [0.2*max_dist, 0.3*max_dist, 0.3*maxRadius],
    #                  [0.3*max_dist, 0.4*max_dist, 0.4*maxRadius],
    #                  [0.4*max_dist, 0.5*max_dist, 0.5*maxRadius],
    #                  [0.5*max_dist, 0.6*max_dist, 0.6*maxRadius],
    #                  [0.6*max_dist, 0.7*max_dist, 0.7*maxRadius],
    #                  [0.7*max_dist, 0.8*max_dist, 0.8*maxRadius],
    #                  [0.8*max_dist, 0.9*max_dist, 0.9*maxRadius],
    #                  [0.9*max_dist, max_dist + 1, maxRadius]]
    #     for entry in thrMatrix:
    #         # select values in the range (entry[0], entry[1]]
    #         mask_range = np.all(np.stack((dist_transform>entry[0], dist_transform<=entry[1]), axis=2), axis=2)
    #         mask_range = np.stack((mask_range,mask_range), axis=2)
    
    #         warpField_temp = cv2.GaussianBlur(warpField, (0, 0), entry[2])        
    #         warpField_blured[mask_range] = warpField_blured[mask_range]       
    #     return warpField_blured
    
    # def mainWarpField(self, im,srcPoints,dstPoints,dt_im, fake_batch, j):
    #     yy,xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    #     numPixels = im.shape[0] * im.shape[1]
    #     xxyy = np.reshape(np.stack((xx.reshape(numPixels),yy.reshape(numPixels)), axis = 1), (numPixels, 1, 2))           
    
    #     # get a mask with triangle indices
    #     (maskIdc, matrixA) = self.getTriMaskAndMatrix(im,srcPoints,dstPoints,dt_im)
          
    #     # compute the initial warp field (offsets) and smooth it
    #     warpField = self.getWarpInit(matrixA, maskIdc,numPixels,xxyy)  #size: im.shape[0]*im.shape[1]*2
    #     warpField_blured = self.smoothWarpField(dstPoints, warpField)
        
    #     # get the corresponding indices instead of offsets and make sure theu are in the image range
    #     warpField_idc = warpField_blured + np.stack((xx,yy), axis = 2)
    #     warpField_idc[:,:,0] = np.clip(warpField_idc[:,:,0],0,im.shape[1]-1)  #x
    #     warpField_idc[:,:,1] = np.clip(warpField_idc[:,:,1],0,im.shape[0]-1)  #y
        
    #     # fill in the image with corresponding indices
    #     im_new2 = im.copy()
    #     im_new2[yy,xx,:] = im[np.intc(warpField_idc[:,:,1]), np.intc(warpField_idc[:,:,0]),:]
    #     fake_batch[j, :, yy , xx] = fake_batch[j , : , np.intc(warpField_idc[:,:,1]), np.intc(warpField_idc[:,:,0])]
    #     return im_new2
    
    # def getWarpInit(self, matrixA, maskIdc,numPixels,xxyy):
    #     maskIdc_resh = maskIdc[:,:,0].reshape(numPixels)
    #     warpField = np.zeros((maskIdc.shape[0],maskIdc.shape[1],2), np.float32)
    #     warpField = warpField.reshape((numPixels,2))
        
    #     # wrap triangle by triangle
    #     for i in range(0, matrixA.shape[2]):
    #         xxyy_masked = []
    #         xxyy_masked = xxyy[maskIdc_resh==i,:]
    #         # don't process empty array
    #         if xxyy_masked.size == 0:
    #             continue
    #         warpField_temp = cv2.transform(xxyy_masked, np.squeeze(matrixA[:,:,i]))
    #         warpField[maskIdc_resh==i,:] =  np.reshape(warpField_temp, (warpField_temp.shape[0], 2))  
    
    #         # reshape to original image shape 
    #     warpField = warpField.reshape((maskIdc.shape[0],maskIdc.shape[1],2))
    #     return warpField
    
    # # Omit the out-of-face control points in the template which fall out of image range after alignment
    # def omitOOR(self, landmarks_template, shape):
    #     outX = np.logical_or((landmarks_template[:,0] < 0),  (landmarks_template[:,0] >= shape[1]))
    #     outY = np.logical_or((landmarks_template[:,1] < 0), (landmarks_template[:,1] >= shape[0]))
    #     outXY = np.logical_or(outX,outY)
    #     landmarks_templateConstrained = landmarks_template.copy()
    #     landmarks_templateConstrained = landmarks_templateConstrained[~outXY]
    #     return landmarks_templateConstrained
    
    # def insertBoundaryPoints(self, width, height, np_array):
    #     ## Takes as input non-empty numpy array
    #     np_array = np.append(np_array,[[0, 0]],axis=0)
    #     np_array = np.append(np_array,[[0, height//2]],axis=0)
    #     np_array = np.append(np_array,[[0, height-1]],axis=0)    
    #     np_array = np.append(np_array,[[width//2, 0]],axis=0)
    #     np_array = np.append(np_array,[[width//2, height-1]],axis=0)        
    #     np_array = np.append(np_array,[[width-1, 0]],axis=0)
    #     np_array = np.append(np_array,[[width-1, height//2]],axis=0)   
    #     np_array = np.append(np_array,[[width-1, height-1]],axis=0)
    #     return np_array
    
    # # Calculate Delaunay triangles for set of points.
    # # Returns the vector of indices of 3 points for each triangle
    # def calculateDelaunayTriangles(self, subdiv, points):
    #     # Get Delaunay triangulation
    #     triangleList = subdiv.getTriangleList()
    
    #     # Find the indices of triangles in the points array
    #     delaunayTri = []
    
    #     for t in triangleList:
    #         # The triangle returned by getTriangleList is
    #         # a list of 6 coordinates of the 3 points in
    #         # x1, y1, x2, y2, x3, y3 format.
    #         # Store triangle as a list of three points
    #         pt = []
    #         pt.append((t[0], t[1]))
    #         pt.append((t[2], t[3]))
    #         pt.append((t[4], t[5]))
    
    #         pt1 = (t[0], t[1])
    #         pt2 = (t[2], t[3])
    #         pt3 = (t[4], t[5])
    
    #         #if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
    #         # Variable to store a triangle as indices from list of points
    #         ind = []
    #         # Find the index of each vertex in the points list
    #         for j in range(0, 3):
    #             for k in range(0, len(points)):
    #                 if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
    #                     ind.append(k)
    #                     # Store triangulation as a list of indices
    #         if len(ind) == 3:
    #             delaunayTri.append((ind[0], ind[1], ind[2]))
    #     return delaunayTri
    
    # def hallucinateControlPoints(self, landmarks_init, im_shape, INPUT_DIR='E:/Full_Model/Fine_Tune_data/features', performTriangulation = False):
    
    #     # load template control points
    #     templateCP_fn = os.path.join(INPUT_DIR, 'brene_controlPoints.txt')
    #     templateCP_init = np.int32(np.loadtxt(templateCP_fn, delimiter=' '))    
        
    #     # align the template to the frame
    #     tform = self.getRigidAlignment(templateCP_init, landmarks_init)
        
    #     # Apply similarity transform to the frame
    #     templateCP = np.reshape(templateCP_init, (templateCP_init.shape[0], 1, templateCP_init.shape[1]))
    #     templateCP = cv2.transform(templateCP, tform)
    #     templateCP = np.reshape(templateCP, (templateCP_init.shape[0], templateCP_init.shape[1]))
        
    #     # omit the points outside the image range
    #     templateCP_Constrained = self.omitOOR(templateCP, im_shape)
    
    #     # hallucinate additional keypoint on a new image 
    #     landmarks_list = landmarks_init.tolist()
    #     for p in templateCP_Constrained[68:]:
    #         landmarks_list.append([p[0], p[1]])
    #     landmarks_out = np.array(landmarks_list)
            
    #     subdiv_temp = None
    #     dt_temp = None
    #     if performTriangulation:
    #         srcTemplatePoints = templateCP_Constrained.copy()
    #         srcTemplatePoints = self.insertBoundaryPoints(im_shape[1], im_shape[0], srcTemplatePoints)
    #         subdiv_temp = self.createSubdiv2D(im_shape, srcTemplatePoints)  
    #         dt_temp = self.calculateDelaunayTriangles(subdiv_temp, srcTemplatePoints) 
        
    #     return (subdiv_temp, dt_temp, landmarks_out)
    
    # def createSubdiv2D(self, size, landmarks):
    #     '''
    #         Input
    #     size[0] is height
    #     size[1] is width    
    #     landmarks as in dlib-detector output
    #         Output
    #     subdiv -- Delaunay Triangulation        
    #     '''   
    #     # Rectangle to be used with Subdiv2D
    #     rect = (0, 0, size[1], size[0])
        
    #     # Create an instance of Subdiv2D
    #     subdiv = cv2.Subdiv2D(rect)
        
    #     # Insert points into subdiv
    #     for p in landmarks :
    #         subdiv.insert((p[0], p[1])) 
          
    #     return subdiv
