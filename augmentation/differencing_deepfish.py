#importlibraries

#libraries to extract frame
import os
import pandas as pd
import cv2
import numpy as np
import shutil

#libraries to undergo differencing
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.decomposition import PCA

#argument library
import argparse
parser = argparse.ArgumentParser(description='This script performs three frame differencing, using absalute value, range or \
variance measures of spread. Users can adjust the differencing distance, as well as a scaling factor alpha.')

parser.add_argument("-l","--location_dat", help="Path to location dat csv",required=True)
parser.add_argument("-t","--training_loc", help="training data folder where yaml file and folders for images & labels are assumed to be within",required=True)
parser.add_argument("-rs","--resize", help="image size (if resizing needed for surrounding images).\
If you do not want to resize surrounding images replace with False (default)",nargs="*",default=False,type=int)
parser.add_argument("-s","--time_stamp", help="Pixel location of time_stamp in video if wanted to be removed. \
This blacks out top left corner of video to x1 to x2 pixels across and y1 to y2 down. if you do not want to \
remove timestamp, replace with False (default)",nargs="*",default=False,type=int)
parser.add_argument("-c","--center_image_surround", help="whether to read in center frame from surrounding video or images", action="store_true")
parser.add_argument("-di","--differenced_directory", help="Directory where differenced images will be saved",required=True)
parser.add_argument("-n","--name", help="Differenced image folder name",required=True)
parser.add_argument("-d","--diff", help="Distance in frames between differenced frames and center frame (defaults to 1).",default=1,type=int)
parser.add_argument("-a","--alpha", help="Multiplicative scaling parameter alpha, to scale resulting differences by.",type=int,default=1)
parser.add_argument("-r","--red", help="Whether to replace the red layer or the entire image (TRUE just replaces red layer - defaults to False)",action="store_true")
parser.add_argument("-m","--method", help="Method of differencing. One of 'abs', 'var', 'range' or 'naive'. Defaults to 'abs'",default="abs",required=True)
parser.add_argument("-no","--norm", help="Wether to normalising the differencing componenent of the image.",action="store_true")
parser.add_argument("-p","--pca", help="Whether to undergo PCA with whitening on the three RGB frames to reduce to two RGB frames. Defaults to False.",action="store_true")
parser.add_argument("-lo","--log_image", help="Whether log transform differences.",action="store_true")

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

#image size (if resizing needed for surrounding images)
image_size = args.resize
#if you do not want to resize surrounding images replace with False

#pixel location of time_stamp in video if wanted to be removed
#this blacks out top left corner of video to 635 pixels across and 70 down
#if you do not want to remove timestamp replace with False
time_stamp_loc = args.time_stamp

#whether to read in center frame from surrounding video/images
center_image_surround = args.center_image_surround

#directory where differenced images will be saved
diff_folder = args.differenced_directory

#differenced image folder name
diff_name = args.name

#differenced parameters
diff = args.diff
alpha = args.alpha
method = args.method
red_layer = args.red
normalise = args.norm
do_pca = args.pca
log_im = args.log_image

#images
image_loc = training_dat_loc + 'images/'
#labels
label_loc = training_dat_loc + 'labels/'
#data.yaml directory
yaml_loc =  training_dat_loc + 'data.yaml'
#differencing location
diff_loc = diff_folder + diff_name + '/'


#prepare differenced directory
os.mkdir(diff_loc)
os.mkdir(diff_loc + '/images')
os.mkdir(diff_loc + '/images/train')
os.mkdir(diff_loc + '/images/test')
os.mkdir(diff_loc + '/images/valid')

shutil.copytree(label_loc,diff_loc + '/labels')
shutil.copy(yaml_loc,diff_loc + '/data.yaml')

#import image location data
val_frame_dat = pd.read_csv(image_to_vid_file)


im_num_list = list(range(0,val_frame_dat.shape[0]))

for im_num in im_num_list:
    #get image folder
    img_folder = sorted(os.listdir(val_frame_dat['image_folder'][im_num]))

    #get index in image folder
    im_index = img_folder.index(val_frame_dat['image_name'][im_num])

    #extract left and right image
    left_image_name = img_folder[im_index-1]
    left_image = Image.open(val_frame_dat['image_folder'][im_num] + left_image_name)
    right_image_name = img_folder[im_index+1]
    right_image = Image.open(val_frame_dat['image_folder'][im_num] + right_image_name)

    #give conditions if left or right image does not exist
    if(left_image_name[:-7]!=val_frame_dat['image_name'][im_num][:-7]):
        left_image = right_image
    if(right_image_name[:-7]!=val_frame_dat['image_name'][im_num][:-7]):
        right_image = left_image


    #read in center image
    if(center_image_surround==True):
        img_center = Image.open(val_frame_dat['image_location'][im_num])
    else:
        img_center = Image.open(image_loc + val_frame_dat['split'][im_num] + "/" + val_frame_dat['image_name'][im_num])

    #convert images to tensors
    img_right_tens_fixed = convert_tensor(right_image)
    img_left_tens_fixed = convert_tensor(left_image)
    img_center_tens = convert_tensor(img_center)

    #fix the left & right image before differencing
    if(image_size!=False):    
        img_center_tens = convert_tensor(convert_image(img_center_tens).resize(image_size))
        img_right_tens_fixed = convert_tensor(convert_image(img_right_tens_fixed).resize(image_size))
        img_left_tens_fixed = convert_tensor(convert_image(img_left_tens_fixed).resize(image_size))

    #apply three frame differencing
    if(method=="abs"):
        diff_im = alpha*(abs(img_center_tens-img_right_tens_fixed) + abs(img_center_tens-img_left_tens_fixed))
    elif(method=="var"):
        avg_im = (img_center_tens+img_right_tens_fixed+img_left_tens_fixed)/3
        diff_im_unscaled = ((img_center_tens-avg_im)**2 + (img_right_tens_fixed-avg_im)**2+ (img_left_tens_fixed-avg_im)**2)/2
        diff_im = diff_im_unscaled*(1/torch.max(diff_im_unscaled))*alpha
    elif(method=="range"):
        red_i = torch.stack((img_center_tens[0,:,:],img_right_tens_fixed[0,:,:],img_left_tens_fixed[0,:,:]),dim=0)
        green_i = torch.stack((img_center_tens[1,:,:],img_right_tens_fixed[1,:,:],img_left_tens_fixed[1,:,:]),dim=0)
        blue_i = torch.stack((img_center_tens[2,:,:],img_right_tens_fixed[2,:,:],img_left_tens_fixed[2,:,:]),dim=0)

        red_r = torch.max(red_i,dim=0)[0] - torch.min(red_i,dim=0)[0]
        green_r = torch.max(green_i,dim=0)[0] - torch.min(green_i,dim=0)[0]
        blue_r = torch.max(blue_i,dim=0)[0] - torch.min(blue_i,dim=0)[0]

        diff_im = torch.stack((red_r,green_r,blue_r),dim=0)*alpha
    elif(method=="naive"):
        diff_im = alpha*(2*img_center_tens-img_right_tens_fixed-img_left_tens_fixed)  + 0.5
    else:
        print("Please choose one of 'abs', 'var', 'range' or 'naive' for method.")
        break
        
    if(normalise==True):
        diff_mean = diff_im.mean()
        diff_std = diff_im.std()
        diff_im = (diff_im - diff_mean)/diff_std
    else:
        pass
    
    if(log_im==True):
        diff_im = np.log(diff_im+1)
    else:
        pass
    

    if(red_layer==True):
        diff_im_avg = (diff_im[0,:,:]+diff_im[1,:,:]+diff_im[2,:,:])/3
        if(do_pca==True):
            # Get the image shape
            im_depth, im_height, im_width,  = img_center_tens.shape

            # Reshape the image tensor into a 2D matrix
            im_matrix = np.reshape(convert_image(img_center_tens), (im_height * im_width, im_depth))

            # Perform PCA on the matrix
            pca = PCA(n_components=2,whiten=True)
            reduced_matrix = pca.fit_transform(im_matrix)

            # Reshape the reduced matrix back into a 3D tensor
            reduced_image = convert_tensor(np.reshape(reduced_matrix, (im_height,im_width, 2)))

            #apply sigmoid transform
            reduced_image = torch.sigmoid(reduced_image)
            
            diff_im_final = torch.stack((diff_im_avg,reduced_image[0,:,:],reduced_image[1,:,:]),dim=0)
        else:
            diff_im_final = torch.stack((diff_im_avg,img_center_tens[1,:,:],img_center_tens[2,:,:]),dim=0)
    else:
        diff_im_final=diff_im

    if(time_stamp_loc!=False):
        diff_im_final[:,time_stamp_loc[0]:time_stamp_loc[1],time_stamp_loc[2]:time_stamp_loc[3]]  = 0          
    #save image
    save_image(diff_im_final,diff_loc +"/images/"+ val_frame_dat['split'][im_num] + "/" + val_frame_dat['image_name'][im_num])