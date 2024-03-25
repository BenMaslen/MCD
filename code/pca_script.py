#importlibraries

#libraries to extract frames
import os
import pandas as pd
import cv2
import numpy as np
import shutil

#libraries to undergo image augmentation
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import pickle as pk
import random as random

#argument library
import argparse
parser = argparse.ArgumentParser(description='This script estimates principal components to dimension reduce an image with 3 colour layers to 2, \
    saving the results as a pca.pkl file in the specified differenced directory.')

parser.add_argument("-l","--location_dat", help="Path to location dat csv",required=True)
parser.add_argument("-t","--training_loc", help="training data folder where yaml file and folders for images & labels are assumed to be within",required=True)
parser.add_argument("-c","--center_image_surround", help="whether to read in center frame from surrounding video or images", action="store_true")
parser.add_argument("-i","--iterate", help="whether to iterate through the pca algo per image or run the pca in one go", action="store_true")
parser.add_argument("-di","--differenced_directory", help="Directory where differenced images will be saved",required=True)
parser.add_argument("-n","--number_images", help="Number of images from training set to base sample on.",required=True,type=int)

args = parser.parse_args()

#create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

#set directories
#Imgae_to_video_locator_csv
image_to_vid_file = args.location_dat
#training data location
#folder with images, labels and yaml file are assumed to be within training data folder
training_dat_loc = args.training_loc

#whether to read in center frame from surrounding video/images
center_image_surround = args.center_image_surround

#directory where differenced images will be saved
diff_folder = args.differenced_directory

#iterate
iterate = args.iterate

#number of images to base sample on
n = args.number_images

#images
image_loc = training_dat_loc + 'images/'

#images
image_loc = training_dat_loc + 'images/'

#import image location data
val_frame_dat = pd.read_csv(image_to_vid_file)

im_num_list_full = list(range(0,val_frame_dat.shape[0]))

im_num_list_train = [im_num for im_num in im_num_list_full if val_frame_dat['split'][im_num] == "train"]
#im_num_list = [im_num for im_num in im_num_list_full if val_frame_dat['split'][im_num] == "train"]

im_num_list = random.sample(im_num_list_train, n)
#im_num_list = list(range(0,10))

if(iterate==True):
    expl_var = []

pca = PCA(n_components=2,whiten=True)

for im_num in im_num_list:
    #read in center image
    if(center_image_surround==True):
            #read in video
        video = val_frame_dat['vid_location'][im_num]
        vidcap = cv2.VideoCapture(video)
        frame3 = val_frame_dat['frame'][im_num]
        vidcap.set(1, frame3)
        success,img_center = vidcap.read()
    else:
        img_center = Image.open(image_loc + val_frame_dat['split'][im_num] + "/" + val_frame_dat['image_name'][im_num])

    #convert images to tensors
    img_center_tens = convert_tensor(img_center)

    #fix video image before differencing
    if(center_image_surround==True):
        img_center_tens = torch.stack((img_center_tens[2,:,:],img_center_tens[1,:,:],img_center_tens[0,:,:]),dim=0)
    
    
    im_depth, im_height, im_width,  = img_center_tens.shape    
    # Reshape the image tensor into a 2D matrix
    im_matrix = np.reshape(convert_image(img_center_tens), (im_height * im_width, im_depth))
    
    if(iterate==True):
        pca.fit(im_matrix)
        expl_var.append([pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1]])
    else:    
        if(im_num==im_num_list[0]):
            #start off with the first image matrix
            im_matrix_all = im_matrix
        else:
            im_matrix_all = np.concatenate((im_matrix_all,im_matrix))

if(iterate==False):
    pca.fit(im_matrix_all)
    pd.DataFrame(pca.explained_variance_ratio_).to_csv(diff_folder+"explained_var.csv")
else:
    pd.DataFrame(expl_var).to_csv(diff_folder+"explained_var_norm_1340.csv")
            
pk.dump(pca, open(diff_folder + "pca.pkl","wb"))
