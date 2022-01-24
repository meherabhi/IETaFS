# IETaFS: Identity preserved Expressive Talking Faces with Synchrony
Code For IETaFS: Identity preserved Expressive Talking Faces with Synchrony	

The below lines follow the "Description -- Command" format.

1) To create the environment -- conda env create -f environment.yml

2) To use the exisiting env  -- conda activate ganimation


3) To run animation -- python main.py --mode animation																																														
[ Animate a single image with a audio file ]

At the location "animations/eric_andre" populate the following accordingly.

images_to_animate: images that we want to animate.
pretrained_models: pretrained models (Of the Generator and the AUnet)
results: folder where the resulting images will be stored.
attributes.txt: See the sample to understand how to create 
spectrograms: Spectrograms of the driving audio							

4) To train the model -- python main.py --mode train	
( Set the appropriate parameters in config.py )


5) The format of saving the model is <epoch>-<iteration>-<model_name>.ckpt
The models are saved in the exeriments folder.

6) Activate the corresponding environments for running the codes of ATVGnet, DAVS models. -- conda env list


7) Refer to their official github pages for the commands to follow

  
Parameters:
 
You can either modify these parameters in main.py or by calling them as command line arguments.

Lambdas
  
lambda_cls: classification lambda.
  
lambda_rec: lambda for the cycle consistency loss.
  
lambda_gp: gradient penalty lambda.
  
lambda_sat: lambda for attention saturation loss.
  
lambda_smooth: lambda for attention smoothing loss.
  
Training parameters
  
c_dim: number of Action Units to use to train the model.
  
batch_size
  
num_epochs
  
num_epochs_decay: number of epochs to start decaying the learning rate.
  
g_lr: generator's learning rate.
  
d_lr: discriminator's learning rate.
  
Pretrained models parameters
  
The weights are stored in the following format: <epoch>-<iteration>-<G/D>.ckpt where G and D represent the Generator and the Discriminator respectively. We save the state of thoptimizers in the same format and extension but add '_optim'.
  
resume_iters: iteration numbre from which we want to start the training. Note that we will need to have a saved model corresponding to that exact iteration number.
  
first_epoch: initial epoch for when we train from pretrained models.
  
Miscellaneous:
  
mode: train/test.
  
image_dir: path to your image folder.
  
attr_path: path to your attributes txt folder.
  
outputs_dir: name for the output folder.
  
Virtual
  
use_virtual: this flag activates the use of cycle consistency loss during the training.

<p align="center">
  <img width="170" height="170" src="https://github.com/meherabhi/IETaFS-Identity-preserved-Expressive-Talking-Faces-with-Synchrony/blob/main/results/face_comp.png">
</p>

<p align="center">
  <img width="600" height="150" src="https://github.com/meherabhi/IETaFS-Identity-preserved-Expressive-Talking-Faces-with-Synchrony/blob/main/results/model_comp.jpg">
</p>

#### Use your own dataset
If you want to use other datasets you will need to detect and crop bounding boxes around the face of each image, compute their corresponding Action Unit vectors and resize them to 128x128px.

You can perform all these steps using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). First you will need to setup the project. They provide guides for [linux](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) and [windows](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). Once the models are compiled, read their [Action Unit wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and their [documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments) on these models to find out which is the command that you need to execute.

After computing these Action Units, depending on the command that you have used, you will obtain different output formats. With the command that I used, I obtained a _csv_ file for each image containing its corresponding Action Units vector among extra information, a folder for each image containing the resized and cropped image and a _txt_ file with extra details about each image. You can find in _openface_utils_ folder the code that I used to extract all the Action Unit information in a _txt_ file and to group all the images into a single folder.

After having the Action Unit _txt_ file and the image folder you can move them to the directory of this project. By default, this code assumes that you have these two elements in _`./data/celeba/`_.

This code is an extension to the implementation in https://github.com/vipermu/ganimation.




