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

#argument library
import argparse
parser = argparse.ArgumentParser(description='This script performs image augmentation on images with surrounding videos \
    using optical flow estimated with the Farneback algorithm from OpenCV.')

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
parser.add_argument("-r","--red", help="Whether to replace the red layer or the entire image (TRUE just replaces red layer - defaults to False)",action="store_true")
parser.add_argument("-p","--pca", help="Whether to undergo PCA with whitening on the three RGB frames to reduce to two. In order to use this option, you need to \
    run the pca_script first on the training data and save the pca.pkl file in the differenced_directory. Defaults to false.",action="store_true")

args = parser.parse_args()

#create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

#set directories
image_to_vid_file = args.location_dat
#training data location
#folder with images, labels and yaml file are assumed to be within training data folder
training_dat_loc = args.training_loc

#image size (if resizing needed for surrounding images)
image_size = args.resize
#if you do not want to resize surrounding images replace with False

#pixel location of time_stamp in video if wanted to be removed
#if you do not want to remove timestamp replace with False
time_stamp_loc = args.time_stamp

#whether to read in center frame from surrounding video/images
center_image_surround = args.center_image_surround

#directory where differenced images will be saved
diff_folder = args.differenced_directory

#differenced image folder name
diff_name = args.name

#flow parameters
diff = args.diff
red_layer = args.red
do_pca = args.pca

#images
image_loc = training_dat_loc + 'images/'
#labels
label_loc = training_dat_loc + 'labels/'
#data.yaml directory
yaml_loc =  training_dat_loc + 'data.yaml'
#differencing location
diff_loc = diff_folder + diff_name + '/'


#prepare augmented image directory
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
    #read in video
    video = val_frame_dat['vid_location'][im_num]
    vidcap = cv2.VideoCapture(video)

    #extract frames
    frame2 = val_frame_dat['frame'][im_num] - diff
    vidcap.set(1, frame2)
    success2,image2 = vidcap.read()

    #case where we don't have all the images:
    if(success2==False):
        frame = val_frame_dat['frame'][im_num] + diff
        vidcap.set(1, frame)
        success1,image2 = vidcap.read()


    #read in center image
    if(center_image_surround==True):
        frame3 = val_frame_dat['frame'][im_num]
        vidcap.set(1, frame3)
        success,img_center = vidcap.read()
    else:
        img_center = Image.open(image_loc + val_frame_dat['split'][im_num] + "/" + val_frame_dat['image_name'][im_num])

    #convert images to tensors
    img_left_tens = convert_tensor(image2)
    img_center_tens = convert_tensor(img_center)

    #fix the left & right image before differencing
    if(center_image_surround==True):
        img_center_tens = torch.stack((img_center_tens[2,:,:],img_center_tens[1,:,:],img_center_tens[0,:,:]),dim=0)
        if(image_size!=False):    
            img_center_tens = convert_tensor(convert_image(img_center_tens).resize(image_size))


    img_left_tens_fixed = torch.stack((img_left_tens[2,:,:],img_left_tens[1,:,:],img_left_tens[0,:,:]),dim=0)
    if(image_size!=False):     
        img_left_tens_fixed = convert_tensor(convert_image(img_left_tens_fixed).resize(image_size))

    #apply optical flow
    prvs    = cv2.cvtColor(np.array(convert_image(img_left_tens_fixed)),cv2.COLOR_BGR2GRAY)
    middle    = cv2.cvtColor(np.array(convert_image(img_center_tens)),cv2.COLOR_BGR2GRAY)

    flow1 = cv2.calcOpticalFlowFarneback(prvs,middle, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow1[...,0], flow1[...,1])
    hsv = np.zeros_like(np.array(convert_image(img_center_tens)))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flow_img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    hsv_tens = convert_tensor(hsv)

    if(red_layer==True):
        flow_mag_img = hsv_tens[2,:,:]
        if(do_pca==True):
            if(im_num==0):
                pca = pk.load(open(diff_folder + "pca.pkl",'rb'))
            #result_new = pca_reload .transform(X)
            
            # Get the image shape
            im_depth, im_height, im_width,  = img_center_tens.shape

            # Reshape the image tensor into a 2D matrix
            im_matrix = np.reshape(convert_image(img_center_tens), (im_height * im_width, im_depth))

            # Perform PCA on the matrix
            #pca = PCA(n_components=2,whiten=True)
            reduced_matrix = pca.transform(im_matrix)

            # Reshape the reduced matrix back into a 3D tensor
            reduced_image = convert_tensor(np.reshape(reduced_matrix, (im_height,im_width, 2)))

            #apply sigmoid transform
            reduced_image = torch.sigmoid(reduced_image)
            
            diff_im_final = torch.stack((flow_mag_img,reduced_image[0,:,:],reduced_image[1,:,:]),dim=0)
        else:
            diff_im_final = torch.stack((flow_mag_img,img_center_tens[1,:,:],img_center_tens[2,:,:]),dim=0)
    else:
        diff_im_final= convert_tensor(flow_img)

    if(time_stamp_loc!=False):
        diff_im_final[:,time_stamp_loc[0]:time_stamp_loc[1],time_stamp_loc[2]:time_stamp_loc[3]]  = 0          

    #save image
    save_image(diff_im_final,diff_loc +"/images/"+ val_frame_dat['split'][im_num] + "/" + val_frame_dat['image_name'][im_num])