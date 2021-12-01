# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:04:47 2021

@author: eee
"""


import os 
import cv2  
from PIL import Image  
  

  
# Folder which contains all the images 
# from which video is to be generated 
os.chdir("E:/Full_Model/animations/eric_andre/results")   
  
mean_height = 0
mean_width = 0
  
num_of_images = len(os.listdir('.')) 

  
# Video Generating function 
def generate_video(): 
    image_folder = '.' # make sure to use your folder 
    video_name = 'video-1.mp4'
    
      
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
     
    # Array images should only consider 
    # the image files ignoring others if any 
    print(images)  
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    # setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, 0, 25, (width, height))  
  
    # Appending the images to the video one by one 
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated 
  
  
# Calling the generate_video function 
generate_video() 