import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
# import torchvision
from utils import Utils
import cv2, os
# from skimage import transform
# import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import random
import glob
# from HRNet.lib.core import function

# import cv2
# import numpy as np
# import math
# import dlib
# import kornia
# from kornia.color.gray import bgr_to_grayscale
# import torch
# from kornia import crop_and_resize
# from PIL import Image
# from torchvision import transforms

class Solver(Utils):

    def __init__(self, data_loader, config_dict):
        # NOTE: the following line create new class arguments with the
        # values in config_dict
        self.__dict__.update(**config_dict)
# the data_loader can be enumerated
        self.data_loader = data_loader
        if self.run_device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device='cpu'
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
        

        
        for epoch in range(self.first_epoch + self.num_epochs_AU_train, self.num_epochs + self.num_epochs_AU_train):
            print(f"EPOCH {epoch} WITH {len(self.data_loader)} STEPS")
                
            self.alpha_rec = 1
            self.epoch = epoch
            for iteration in range(self.first_iteration, len(self.data_loader)):     

                self.iteration = iteration            
                torch.cuda.empty_cache()
                self.get_training_data()
                # self.AU_training()
                # self.Pose_D()
                # self.lmk_loss()
                self.train_discriminator()
                self.joint_training_D()
                # self.Color_disc()
                
                
# 'number of D updates per each G update'
                if (self.iteration+1) % self.n_critic == 0:
                    generation_outputs = self.train_generator()
                    # self.joint_training_G()

                if (self.iteration+1) % self.sample_step == 0:
                    self.print_generations(generation_outputs)
# After how many interations is the model saved
                if self.iteration % self.model_save_step == 0:
                    self.save_models(self.iteration, self.epoch)
# After how many steps are the events logged
                if self.iteration % self.log_step == 0:
                    self.update_tensorboard()
                self.global_counter += 1
                
                # current_time = datetime.datetime.now().hour
                # (current_time >= 10 and current_time < 22) and 
                if self.device=='cpu':
                    self.device= 'cuda'
                    print("Currently running on", self.device)
                    self.G = self.G.to(self.device)
                    self.D =self.D.to(self.device)
                    self.audionet = self.audionet.to(self.device)
                    self.videonet = self.videonet.to(self.device)
                    self.AUnet = self.AUnet.to(self.device)
                    self.HRNet= self.HRNet.to(self.device)
                    # self.Color_D= self.Color_D.to(self.device)
                    self.optimizer_to(self.g_optimizer, self.device)
                    self.optimizer_to(self.d_optimizer, self.device)
                    self.optimizer_to(self.AU_optimizer, self.device)
                    self.optimizer_to(self.audio_optimizer, self.device)
                    self.optimizer_to(self.video_optimizer, self.device)
                    # self.optimizer_to(self.Color_d_optimizer, self.device)
                                        
                # if (current_time >= 22 or current_time < 10)  and self.device=='cuda':
                #     self.device='cpu'
                #     print("Currently running on", self.device)
                #     self.G = self.G.to(self.device)
                #     self.D =self.D.to(self.device)
                #     self.audionet = self.audionet.to(self.device)
                #     self.videonet = self.videonet.to(self.device)
                #     self.AUnet = self.AUnet.to(self.device)
                #     self.HRNet= self.HRNet.to(self.device)
                #     # self.Color_D= self.Color_D.to(self.device)
                #     self.optimizer_to(self.g_optimizer, self.device)
                #     self.optimizer_to(self.d_optimizer, self.device)
                #     self.optimizer_to(self.AU_optimizer, self.device)
                #     self.optimizer_to(self.audio_optimizer, self.device)
                #     self.optimizer_to(self.video_optimizer, self.device)  
                #     # self.optimizer_to(self.Color_d_optimizer, self.device)

            # Decay learning rates.
            if (self.epoch+1) > self.num_epochs_decay:
                # float(self.num_epochs_decay))
                self.g_lr -= (self.g_lr / 10.0)
                # float(self.num_epochs_decay))
                self.d_lr -= (self.d_lr / 10.0)
                self.update_lr(self.g_lr, self.d_lr)
                print('Decayed learning rates, self.g_lr: {}, self.d_lr: {}.'.format(
                    self.g_lr, self.d_lr))

            # Save the last model
            self.save_models(self.iteration, self.epoch)

            self.first_iteration = 0  # Next epochs start from 0
# This function will load self.x_real with the inout images
# self.label_org with the inout image AU's
# self.label_trg with the randomly generated target AU vectors
# self.label_trg_virtual: labels for virtual cycle consistency

    def get_training_data(self):
        try:
            self.spect_image, self.x_real_AV, self.label_org , self.x_real, self.x_real_label = next(self.data_iter)
        except:
            #iter() function returns an iterator for the given object
            self.data_iter = iter(self.data_loader)
            # From the iterator take the input and the corresponding label
            #next() function returns the next item in an iterator.
            self.spect_image, self.x_real_AV, self.label_org, self.x_real, self.x_real_label = next(self.data_iter)

        self.x_real = self.x_real.to(self.device)  # Load the Input images to the device.
        # Load the Labels for computing classification loss.
        self.x_real_AV = self.x_real_AV.to(self.device)
        self.x_real_label = self.x_real_label.to(self.device)
        
        self.label_org = self.label_org.to(self.device)
        self.spect_image= self.spect_image.to(self.device)
        # Get random targets for training
        self.label_trg = self.get_random_labels_list()
        # Convert the values to float and clamp the values
        self.label_trg = torch.FloatTensor(self.label_trg).clamp(0, 1)
        # Load the Labels for computing classification loss.
        self.label_trg = self.label_trg.to(self.device)
# use_virtual: 'Boolean to decide if we should use the virtual cycle concistency loss'
        if self.use_virtual:
            self.label_trg_virtual = self.get_random_labels_list()
            self.label_trg_virtual = torch.FloatTensor(
                self.label_trg_virtual).clamp(0, 1)
            # Labels for computing classification loss.
            self.label_trg_virtual = self.label_trg_virtual.to(self.device)

            assert not torch.equal(
                self.label_trg_virtual, self.label_trg), "Target label and virtual label are the same"
# This function creates a list called trg_list which holds all the randomly generated target AU 
# vectors which, this randomness is created by adding a uniform variance to each element of AU list
# size of trg_list will be [No of AU x size of batch ]
    def get_random_labels_list(self):
        trg_list = []
        for _ in range(self.batch_size):
            random_num = random.randint(
                0, len(self.data_loader)*self.batch_size-1)
            # Select a random AU vector from the dataset
            trg_list_aux = self.data_loader.dataset[random_num][2]
            # Apply a variance of 0.1 to the vector
            trg_list.append(trg_list_aux.numpy() +
                            np.random.uniform(-0.1, 0.1, trg_list_aux.shape))
        return trg_list

    def train_discriminator(self):
        # Compute loss with real images.
        critic_output, classification_output = self.D(self.x_real_AV)
        # Critic decides if the image is real or not, and the mean of the loss is taken as the critic_loss
        d_loss_critic_real = -torch.mean(critic_output)
        # MSE between the original label and the one predicted is taken for the classification loss
        d_loss_classification = torch.nn.functional.mse_loss(
            classification_output, self.label_org)
        
        # Compute loss with fake images.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        # creates the fake image with the target expression using the color and the attention masks
        # and is stored as x_fake
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)
        # by using .detach(), the gradient is not computed for that input
        # hence only the discriminator is trained and the gradient wont flow back to the gen
        # Here we are only calculating the critic loss for real/fake
        critic_output, _ = self.D(x_fake.detach())
        d_loss_critic_fake = torch.mean(critic_output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(self.x_real.size(0), 1, 1, 1).to(self.device)
        # Half of image info from fake and half from real
        x_hat = (alpha * self.x_real.data + (1 - alpha)
                 * x_fake.data).requires_grad_(True)
        critic_output, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(critic_output, x_hat)

        # Backward and optimize.
        d_loss = d_loss_critic_real + d_loss_critic_fake +  self.lambda_cls * \
            d_loss_classification + self.lambda_gp * d_loss_gp

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        self.loss_visualization['D/loss'] = d_loss.item()
        self.loss_visualization['D/loss_real'] = d_loss_critic_real.item()
        self.loss_visualization['D/loss_fake'] = d_loss_critic_fake.item()
        self.loss_visualization['D/loss_cls'] = self.lambda_cls * \
            d_loss_classification.item()
        self.loss_visualization['D/loss_gp'] = self.lambda_gp * \
            d_loss_gp.item()

    def train_generator(self):
        # Original-to-target domain.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        # creates the fake image with the target expression using the color and the attention masks
        # and is stored as x_fake
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)
        # This fake image is passed through  both the discriminators
        # and the corresponding loss are taken
        critic_output, classification_output = self.D(x_fake)
        g_loss_fake = -torch.mean(critic_output)
        g_loss_cls = torch.nn.functional.mse_loss(
            classification_output, self.label_trg)

        # Target-to-original domain.
        if not self.use_virtual:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.x_real_label)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

        else:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.x_real_label)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

            reconstructed_attention_mask_2, reconstructed_color_regression_2 = self.G(
                x_fake, self.label_trg_virtual)
            x_fake_virtual = self.imFromAttReg(
                reconstructed_attention_mask_2, reconstructed_color_regression_2, x_fake)

            reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression = self.G(
                x_fake_virtual, self.label_trg)
            x_rec_virtual = self.imFromAttReg(
                reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression, x_fake_virtual.detach())

        # Compute losses
        g_loss_saturation_1 = attention_mask.mean()
        g_loss_smooth1 = self.smooth_loss(attention_mask)

        if not self.use_virtual:
            g_loss_rec = torch.nn.functional.l1_loss(self.x_real, x_rec)
            g_loss_saturation_2 = reconstructed_attention_mask.mean()
            g_loss_smooth2 = self.smooth_loss(reconstructed_attention_mask)

        else:
            g_loss_rec = (1-self.alpha_rec)*torch.nn.functional.l1_loss(self.x_real, x_rec) + \
                self.alpha_rec * \
                torch.nn.functional.l1_loss(x_fake, x_rec_virtual)

            g_loss_saturation_2 = (1-self.alpha_rec) * reconstructed_attention_mask.mean() + \
                self.alpha_rec * reconstructed_virtual_attention_mask.mean()

            g_loss_smooth2 = (1-self.alpha_rec) * self.smooth_loss(reconstructed_virtual_attention_mask) + \
                self.alpha_rec * self.smooth_loss(reconstructed_attention_mask)

        g_attention_loss = self.lambda_smooth * g_loss_smooth1 + self.lambda_smooth * g_loss_smooth2 \
            + self.lambda_sat * g_loss_saturation_1 + self.lambda_sat * g_loss_saturation_2
            
        pred_label=self.AUnet(self.spect_image)
        attention_mask, color_regression = self.G(self.x_real, pred_label)
        x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)        
        videofeat=self.videonet(x_fake)
        audiofeat=self.audionet(self.spect_image)
        real_label= videofeat.new_ones((self.batch_size,1))
        av_loss = self.contrastive_loss(videofeat, audiofeat, real_label)
        total_loss_2 =av_loss

        # Color_feat_fake = self.Color_D(x_fake)
        # color_loss = self.dis_loss(Color_feat_fake , 1 )

        g_loss =  0.5 * total_loss_2 + g_loss_fake + self.lambda_rec * g_loss_rec + \
            self.lambda_cls * g_loss_cls + g_attention_loss

        self.reset_grad()
        
        g_loss.backward()
        
        self.g_optimizer.step()
        self.AU_optimizer.step()
        
        # Logging.
        # self.loss_visualization['Color_G/loss'] = color_loss.item()
        self.loss_visualization['Joint_G/loss'] = 0.5 * total_loss_2.item()
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/loss_fake'] = g_loss_fake.item()
        self.loss_visualization['G/loss_rec'] = self.lambda_rec * \
            g_loss_rec.item()
        self.loss_visualization['G/loss_cls'] = self.lambda_cls * \
            g_loss_cls.item()
        self.loss_visualization['G/attention_loss'] = g_attention_loss.item()
        self.loss_visualization['G/loss_smooth1'] = self.lambda_smooth * \
            g_loss_smooth1.item()
        self.loss_visualization['G/loss_smooth2'] = self.lambda_smooth * \
            g_loss_smooth2.item()
        self.loss_visualization['G/loss_sat1'] = self.lambda_sat * \
            g_loss_saturation_1.item()
        self.loss_visualization['G/loss_sat2'] = self.lambda_sat * \
            g_loss_saturation_2.item()
        self.loss_visualization['G/alpha'] = self.alpha_rec

        if not self.use_virtual:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
            }

        else:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
                "reconstructed_virtual_attention_mask": reconstructed_virtual_attention_mask,
                "reconstructed_virtual_color_regression": reconstructed_virtual_color_regression,
                "x_rec_virtual": x_rec_virtual,
            }
        
            
    def contrastive_loss(self, videofeat, audiofeat, label):
        euclidean_distance = F.pairwise_distance(videofeat, audiofeat, p=2)
        loss_contrastive = torch.tensor([0.]).to(self.device)
        for i in range(self.batch_size):
            loss_contrastive += 0.5*(
                                (label[i][0])*torch.pow(euclidean_distance[i], 2)
                                +
                                (1-label[i][0])*torch.pow(torch.clamp(1-euclidean_distance[i], min=0.0), 2)
                                )
        loss_contrastive = loss_contrastive / self.batch_size
        return loss_contrastive
    
    # def landmarkDetect(self, input_image):
    #     predictor_path = '/home/mtech/kmeher/lip_dir/shape_predictor_68_face_landmarks.dat'
    #     detector = dlib.get_frontal_face_detector()
    #     predictor = dlib.shape_predictor(predictor_path)
    #     dlib.cuda.set_device(0)
    #     # img = Image.fromarray(frame)
    #     # img.show()
    #     points = []
    #     # self.detector = dlib.get_frontal_face_detector
    #     # rects will detect the face in an image and draw a rectangle around it
    #     boxes= input_image.new_zeros([self.batch_size, 4, 2])
    #     angle_rot: torch.tensor = input_image.new_ones(self.batch_size)
    #     scale: torch.tensor = input_image.new_ones(self.batch_size, 2)
    #     center_rot: torch.tensor = input_image.new_ones(self.batch_size , 2)
    #     for j in range(self.batch_size):
    #         frame=input_image[j,:,:,:].transpose(0,2).transpose(0,1).detach().numpy()*255
    #         frame = frame.astype(np.uint8)
    #         frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         rects = detector(frame, 1)
    #         for i in range(len(rects)):
    #             # landmarks will store the LM points which are predicted using dlib.predictor which takes as arguments 
    #             # the frame and the bounding rectangle for the face
    #             landmarks = np.matrix([
    #                         [p.x, p.y] for p in predictor(frame, rects[i]).parts()
    #             ])
    #             frame = frame.copy()
    #             for idx, point in enumerate(landmarks):
    #                 # Landmarks 48,51,54,57 hold the extremities of the lips
    #                 # if there arent 4 landmarks, return a empty list because the frame corresponding 
    #                 # does'nt have a full face
    #                 if(idx==48 or idx==51 or idx==54 or idx==57):
    #                     points.append([point[0,0], point[0,1]])
    #                     # pos = (point[0,0], point[0,1])
    #                     #cv2.circle(frame, pos, 2, (255,0,0), -1)
    #         if len(points) != 4:
    #             return []
    #         print('key points', points)
    #         # cv2.imshow('facialpoint', frame)
    #         (h, w) = frame.shape[:2]
    #         centerX = int(round((points[0][0]+points[2][0])/2.0))
    #         centerY = int(round((points[0][1]+points[2][1])/2.0))
    #         center_rot[j,:] = torch.tensor([centerX, centerY])
    #         RAangle = math.atan(
    #                   1.0*(points[2][1]-points[0][1])/(points[2][0]-points[0][0]))
    #         #print('RAangle:', RAangle)
    #         angle_rot[j] = torch.tensor([180.0/math.pi*RAangle])
    #         halfplusMouth = int(round(1.15/2.0*math.sqrt(
    #                     math.pow(points[2][1]-points[0][1], 2) + math.pow(points[2][0]-points[0][0], 2)
    #         )))
    #         #print('angle:', angle)
    #         boxes[j,0,:]= torch.tensor([centerX-halfplusMouth, centerY-halfplusMouth])
    #         boxes[j,1,:]= torch.tensor([centerX+halfplusMouth, centerY-halfplusMouth])
    #         boxes[j,2,:]= torch.tensor([centerX+halfplusMouth, centerY+halfplusMouth])
    #         boxes[j,3,:]= torch.tensor([centerX-halfplusMouth, centerY+halfplusMouth])
    #     input_image = bgr_to_grayscale(input_image)   
    #     M: torch.tensor= kornia.get_rotation_matrix2d(center_rot.to(self.device), angle_rot.to(self.device), scale.to(self.device)).to(self.device)
    #     input_image = kornia.warp_affine(input_image , M , (h,w).to(self.device))
            
    #         # M = cv2.getRotationMatrix2D(center_rot, angle_rot, scale)
    #         # rotateImg = cv2.warpAffine(frame, M, (w,h))
    #         # cv2.imshow('rotate', rotateImg)    
    #         # extract lip   
    #         # leftUp = (centerX-halfplusMouth, centerY-halfplusMouth)
    #         # rightDown = (centerX+halfplusMouth, centerY+halfplusMouth)
            
            
    #     input_image = crop_and_resize(input_image, boxes.to(self.device), (h,w).to(self.device)) 
    #     points = []
    #     # print(input_image.shape)
    #     # print(input_image.requires_grad)
    #     # output =input_image.squeeze(0).squeeze(0).detach().numpy()*255
    #     # output = output.astype(np.uint8)
    #     # cv2.imwrite("output_lip.jpg", output)
    #     return input_image

    def joint_training_D(self):
        pred_label=self.AUnet(self.spect_image)
        attention_mask, color_regression = self.G(self.x_real, pred_label)
        x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)
        # x_fake = x_fake.flip(-3)
        
        # r: torch.Tensor = x_fake[..., 0:1, :, :]
        # g: torch.Tensor = x_fake[..., 1:2, :, :]
        # b: torch.Tensor = x_fake[..., 2:3, :, :]
        
        # x_fake = 0.299 * r + 0.587 * g + 0.114 * b
        
        
        # r1: torch.Tensor = self.x_real_AV[..., 0:1, :, :]
        # g1: torch.Tensor = self.x_real_AV[..., 1:2, :, :]
        # b1: torch.Tensor = self.x_real_AV[..., 2:3, :, :]
        
        # x_real_AV = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
        
        
        audiofeat=self.audionet(self.spect_image)
        videofeat=self.videonet(self.x_real_AV)
        real_label= videofeat.new_ones((self.batch_size,1))
        fake_label= videofeat.new_zeros((self.batch_size,1))
        av_loss_real= self.contrastive_loss(videofeat, audiofeat, real_label)
        videofeat=self.videonet(x_fake.detach())
        av_loss_fake= self.contrastive_loss(videofeat, audiofeat, fake_label)
        av_loss=av_loss_real+ av_loss_fake
        #Discrminator Training
        # critic_output, _ = self.D(x_fake.detach())
        # d_loss_fake = torch.mean(critic_output)
        total_loss_1= 0.5 * av_loss
        self.reset_grad()
        total_loss_1.backward()
        # update optmizer
        self.audio_optimizer.step()
        self.video_optimizer.step()
        self.loss_visualization['Joint_D/loss'] = total_loss_1.item()
        # self.d_optimizer.step()
        
        
    # def joint_training_G(self):        
    #     pred_label=self.AUnet(self.spect_image)
    #     attention_mask, color_regression = self.G(self.x_real, pred_label)
    #     x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)
        
    #     # r: torch.Tensor = x_fake[..., 0:1, :, :]
    #     # g: torch.Tensor = x_fake[..., 1:2, :, :]
    #     # b: torch.Tensor = x_fake[..., 2:3, :, :]
        
    #     # x_fake = 0.299 * r + 0.587 * g + 0.114 * b
        
    #     # critic_output, classification_output = self.D(x_fake)
    #     # g_loss_fake = -torch.mean(critic_output)
    #     # g_loss_cls = torch.nn.functional.mse_loss(classification_output, self.label_org)
        
    #     videofeat=self.videonet(x_fake)
    #     audiofeat=self.audionet(self.spect_image)
    #     real_label= videofeat.new_ones((self.batch_size,1))
    #     av_loss = self.contrastive_loss(videofeat, audiofeat, real_label)
    #     total_loss_2 =av_loss
    #     self.reset_grad()
    #     total_loss_2.backward()
    #     self.AU_optimizer.step()
    #     self.g_optimizer.step()
        
    #     self.loss_visualization['Joint_G/loss'] = total_loss_2.item()
        
        
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
 
    
    def lmk_loss(self):        
        pred_label=self.AUnet(self.spect_image)
        attention_mask, color_regression = self.G(self.x_real, pred_label)
        x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)
        
        input_1 = x_fake.clone()
        output_size= (256,256)
        input_1 = torch.nn.functional.interpolate(input_1, output_size)

        input_2 = self.x_real_AV.clone()
        input_2 = torch.nn.functional.interpolate(input_2, output_size)
        
        pred_1 = self.HRNet(input_1)
        pred_2 = self.HRNet(input_2)
        
        loss = torch.nn.functional.mse_loss( pred_1 , pred_2 )
        self.reset_grad()
        loss.backward()
        self.g_optimizer.step()
        
        self.loss_visualization['lmk/loss'] = loss.item()
        
    def dis_loss(self, inp, label):
        if label == 1:
            loss = F.relu(1.0 - inp).mean()
        else:
            loss = F.relu(1.0 + inp).mean()
        return loss
        
        
    # def Color_disc(self):
        
    #     pred_label=self.AUnet(self.spect_image)
    #     attention_mask, color_regression = self.G(self.x_real, pred_label)
    #     x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)

    #     Color_feat_real = self.Color_D(self.x_real_AV)
    #     Color_feat_fake = self.Color_D(x_fake.detach())
        
    #     real_label= 1
    #     fake_label= 0
        
    #     color_loss= self.dis_loss(Color_feat_real, real_label) + self.dis_loss(Color_feat_fake, fake_label) 
        
    #     self.reset_grad()
        
    #     color_loss.backward()
    #     self.Color_d_optimizer.step()
        
    #     self.loss_visualization['Color_D/Loss'] = color_loss.item()
        
        
            
    
    # def Pose_D(self):        
    #     pred_label=self.AUnet(self.spect_image)
    #     attention_mask, color_regression = self.G(self.x_real, pred_label)
    #     x_fake = self.imFromAttReg( attention_mask, color_regression, self.x_real)
        
    #     batch_size, im_channels, im_height, im_width = self.x_real.shape
    #     faceDetector = dlib.get_frontal_face_detector()
    #     landmarkDetector = dlib.shape_predictor(self.predictor_dir) 
        
    #     for j in range(0, self.batch_size):
            
    #         frame = self.x_real_AV[j,:,:,:].transpose(0,2).transpose(0,1).cpu().detach().numpy()*255
    #         frame = frame.astype(np.uint8)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         newRect = self.getFaceRect(frame, faceDetector)
    #         landmarks_frame_init = self.landmarks2numpy(landmarkDetector(frame, newRect))
            
    #         im = x_fake[j,:,:,:].transpose(0,2).transpose(0,1).cpu().detach().numpy()*255
    #         im = im.astype(np.uint8)
    #         cv2.imwrite("Before.jpg", im )
            
    #         # save_image(self.denorm(x_fake[j,:,:,:]), 'Fake_before.jpg' )
            
    #         newRect = self.getFaceRect(im, faceDetector)
    #         landmarks_im = self.landmarks2numpy(landmarkDetector(im, newRect))
            
    #         tform = self.getRigidAlignment(landmarks_frame_init, landmarks_im)
            
    #         frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))
            
    #         image = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2RGB)
            
    #         landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
    #         landmarks_frame = cv2.transform(landmarks_frame, tform)
    #         landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))
            
    #         (subdiv_temp, dt_im, landmarks_frame) = self.hallucinateControlPoints(landmarks_init = landmarks_frame, 
    #                                                                         im_shape = frame_aligned.shape, 
    #                                                                         INPUT_DIR=self.feature_dir , 
    #                                                                         performTriangulation = True)
    #         landmarks_list = landmarks_im.copy().tolist()
    #         for p in landmarks_frame[68:]:
    #             landmarks_list.append([p[0], p[1]])
    #         srcPoints = np.array(landmarks_list)
    #         srcPoints = self.insertBoundaryPoints(im_width, im_height, srcPoints) 
            
            
    #         dstPoints_frame = landmarks_frame
    #         dstPoints_frame = self.insertBoundaryPoints(im_width, im_height, dstPoints_frame)
            
    #         dstPoints = dstPoints_frame - srcPoints + srcPoints 
            
    #         im_new = self.mainWarpField(im,srcPoints,dstPoints,dt_im, x_fake , j) 
            
    #         im = x_fake[j,:,:,:].transpose(0,2).transpose(0,1).cpu().detach().numpy()*255
    #         im = im.astype(np.uint8)
    #         cv2.imwrite("After.jpg", im)
            
    #         # save_image(self.denorm(x_fake[j,:,:,:]), 'Fake_after.jpg' )
        
        
    def print_generations(self, generator_outputs_dict):
        # print_epoch_images = False
        save_image(self.denorm(self.x_real), self.sample_dir +
                  '/{}_4real_.png'.format(self.epoch))
        save_image((generator_outputs_dict["color_regression"]+1)/2,
                   self.sample_dir + '/{}_2reg_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_fake"]), self.sample_dir + '/{}_3res_.png'.format(self.epoch))
        save_image(generator_outputs_dict["attention_mask"],
                   self.sample_dir + '/{}_1attention_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_rec"]), self.sample_dir + '/{}_5rec_.png'.format(self.epoch))
        if not self.use_virtual:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

        else:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention_.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

            save_image(generator_outputs_dict["reconstructed_virtual_attention_mask"],
                       self.sample_dir + '/{}_8rec_virtual_attention.png'.format(self.epoch))
            save_image(self.denorm(generator_outputs_dict["reconstructed_virtual_color_regression"]),
                       self.sample_dir + '/{}_91rec_virtual_reg.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["x_rec_virtual"]), self.sample_dir + '/{}_92rec_epoch_.png'.format(self.epoch))

    def update_tensorboard(self):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}],  [{}/{}], Epoch [{}/{}]".format(
            et, self.iteration+1, len(self.data_loader), self.epoch+1, self.num_epochs+ self.num_epochs_AU_train)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in self.loss_visualization.items():
                self.writer.add_scalar(
                    tag, value, global_step=self.global_counter)

    def animation(self, mode='animate_image'):
        
        
        
        
        from PIL import Image
        from torchvision import transforms as T
# The list regular_image_transform has all the sequence of transformations composed, which 
# resize and normalize it
        # regular_image_transform = []
        # regular_image_transform.append(T.ToTensor())
        # regular_image_transform.append(T.Normalize(
        #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        
        regular_image_transform = T.Compose([T.ToTensor(),
                                             T.Normalize(mean=(0.5, 0.5, 0.5), 
                                                         std=(0.5, 0.5, 0.5))])
# Load the model of the generator
        G_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*G.ckpt')), key=self.numericalSort)[0]
        # self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{self.gpu_id}'))
        self.G.load_state_dict(torch.load(G_path, map_location= 'cpu'))
        # self.G = self.G.cuda(0)
        AU_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*AU.ckpt')), key=self.numericalSort)[0]
        # self.AUnet.load_state_dict(torch.load(AU_path, map_location=f'cuda:{self.gpu_id}'))
        self.AUnet.load_state_dict(torch.load(AU_path, map_location='cpu'))
        # self.AUnet = self.AUnet.cuda(0)
        reference_expression_images = []
        
        attributes=[]
        with torch.no_grad():
            with open(self.animation_attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()
    # targets= all zeros for no of attributes in the text
                targets = torch.zeros(len(csv_lines), self.c_dim)
                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    spectrogram_path = os.path.join(
                        self.animation_spectrograms_dir, splitted_lines[0])
                    # input_spectrogram= (torch.from_numpy( np.load(spectrogram_path) ).float()).unsqueeze(0).cuda()
                    input_spectrogram= (torch.from_numpy( np.load(spectrogram_path) ).float()).unsqueeze(0)
                    reference_expression_images.append(splitted_lines[1])
                    targets[idx, :] = self.AUnet(input_spectrogram)
                    attributes.append(targets[idx,:].tolist())
                        
            with open("/home/mtech/kmeher/Full_Model/animations/eric_andre/attributes1.txt", 'w') as file:
                for i,row in enumerate(attributes):
                    f= "{:03d}.jpg".format(i)
                    s = " ".join(map(str, row))
                    file.write(f+" "+ s+'\n')
# Finally from the code above targets, inout_images will have the AU's and their corresponding images

        # if mode == 'animate_random_batch':
        #     animation_batch_size = 7

        #     self.data_iter = iter(self.data_loader)
        #     images_to_animate, _ = next(self.data_iter)
        #     images_to_animate = images_to_animate[0:animation_batch_size].cuda(
        #     )

        #     for target_idx in range(targets.size(0)):
        #         targets_au = targets[target_idx, :].unsqueeze(
        #             0).repeat(animation_batch_size, 1).cuda()
        #         resulting_images_att, resulting_images_reg = self.G(
        #             images_to_animate, targets_au)

        #         resulting_images = self.imFromAttReg(
        #             resulting_images_att, resulting_images_reg, images_to_animate).cuda()

        #         save_images = - \
        #             torch.ones((animation_batch_size + 1)
        #                         * 2, 3, 128, 128).cuda()

        #         save_images[1:animation_batch_size+1] = images_to_animate
        #         save_images[animation_batch_size+1] = input_images[target_idx]
        #         save_images[animation_batch_size +
        #                     2:(animation_batch_size + 1)*2] = resulting_images

        #         save_image((save_images+1)/2, os.path.join(self.animation_results_dir,
        #                                                     reference_expression_images[target_idx]))
                    
                    
            if mode == 'animate_image':
                images_to_animate_path = glob.glob(
                    self.animation_images_dir + '/*')
    # Transform the images present in the image_to_animate directory to the required size and normalize them
                for image_path in images_to_animate_path:
                    image_to_animate = regular_image_transform(
                        # Image.open(image_path)).unsqueeze(0).cuda()
                        Image.open(image_path)).unsqueeze(0)
    
                    for target_idx in range(targets.size(0)):
                        # targets_au = targets[target_idx, :].unsqueeze(0).cuda()
                        targets_au = targets[target_idx, :].unsqueeze(0)
                        resulting_images_att, resulting_images_reg = self.G(
                            image_to_animate, targets_au)
                        # resulting_image = self.imFromAttReg(
                        #     resulting_images_att, resulting_images_reg, image_to_animate).cuda()
                        resulting_image = self.imFromAttReg(
                            resulting_images_att, resulting_images_reg, image_to_animate)
                        save_image(self.denorm(resulting_image), os.path.join(self.animation_results_dir,
                                                                        image_path.split('/')[-1].split('.')[0]
                                                                        + '_' + reference_expression_images[target_idx].rstrip()))
                    

    def finetune(self):
        from PIL import Image
        from torchvision import transforms as T
        regular_image_transform = T.Compose([T.ToTensor(),
                                             T.Normalize(mean=(0.5, 0.5, 0.5), 
                                                         std=(0.5, 0.5, 0.5))])
        # Load the model of the generator
        G_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*G.ckpt')), key=self.numericalSort)[0]
        self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{self.gpu_id}'))
        self.G = self.G.cuda(0)
        AU_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*AU.ckpt')), key=self.numericalSort)[0]
        self.AUnet.load_state_dict(torch.load(AU_path, map_location=f'cuda:{self.gpu_id}'))
        self.AUnet = self.AUnet.cuda(0)
        reference_expression_images = []
        attributes=[]
        with torch.no_grad():
            with open(self.fine_tune_attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()
                targets = torch.zeros(len(csv_lines), self.c_dim)
                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    spectrogram_path = os.path.join(
                        self.fine_tune_spectrograms_dir, splitted_lines[0])
                    input_spectrogram= (torch.from_numpy( np.load(spectrogram_path) ).float()).unsqueeze(0).cuda()
                    reference_expression_images.append(splitted_lines[1])
                    targets[idx, :] = self.AUnet(input_spectrogram)
                    attributes.append(targets[idx,:].tolist())
            
            image_path = os.path.join (self.fine_tune_images_dir, "{:06d}.jpg".format(1635) )
            
            image_to_animate = regular_image_transform(
                Image.open(image_path)).unsqueeze(0).cuda()

            for target_idx in range(targets.size(0)):
                targets_au = targets[target_idx, :].unsqueeze(0).cuda()
                resulting_images_att, resulting_images_reg = self.G(
                    image_to_animate, targets_au)
                resulting_image = self.imFromAttReg(
                    resulting_images_att, resulting_images_reg, image_to_animate).cuda()
                save_image(self.denorm(resulting_image), os.path.join(self.fine_tune_results_dir,"{:06d}.jpg".format(target_idx)))
            
            
        # """ Code to modify single Action Units """

        # Set data loader.
        # self.data_loader = self.data_loader

        # with torch.no_grad():
        #     for i, (self.x_real, c_org) in enumerate(self.data_loader):

        #         # Prepare input images and target domain labels.
        #         self.x_real = self.x_real.to(self.device)
        #         c_org = c_org.to(self.device)

        #         # c_trg_list = self.create_labels(self.data_loader)

        #         crit, cl_regression = self.D(self.x_real)
        #         # print(crit)
        #         print("ORIGINAL", c_org[0])
        #         print("REGRESSION", cl_regression[0])

        #         for au in range(17):
        #             alpha = np.linspace(-0.3,0.3,10)
        #             for j, a in enumerate(alpha):
        #                 new_emotion = c_org.clone()
        #                 new_emotion[:,au]=torch.clamp(new_emotion[:,au]+a, 0, 1)
        #                 attention, reg = self.G(self.x_real, new_emotion)
        #                 x_fake = self.imFromAttReg(attention, reg, self.x_real)
        #                 save_image((x_fake+1)/2, os.path.join(self.result_dir, '{}-{}-{}-images.jpg'.format(i,au,j)))

        #         if i >= 3:
        #             break
            
            
        
        
        