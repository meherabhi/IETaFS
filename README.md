# Photo-Realistic-Face-Generation
Code For Photo Realistic Talking Faces

		
														                                                                         Description   											Command


1) To create the environment									conda env create -f environment.yml

2) To use the exisiting env in the							        conda activate ganimation
EEE dept GPU

3) To run animation 										python main.py --mode animation																																														
[ Animate a single image with a audio file ]

At the location "animations/eric_andre" populate the following accordingly.

images_to_animate: images that we want to animate.
pretrained_models: pretrained models (Of the Generator and the AUnet)
results: folder where the resulting images will be stored.
attributes.txt: See the sample to understand how to create 
spectrograms: Spectrograms of the driving audio							

4) To train the model								  		 python main.py --mode train	
( Set the appropriate parameters in config.py )


5) The format of saving the model is <epoch>-<iteration>-<model_name>.ckpt
The models are saved in the exeriments folder.

6) Activate the corresponding environments 						          conda env list
for running the codes of ATVGnet, DAVS models.

7) Refer to their official github pages for the commands to follow
