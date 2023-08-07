#importlibraries

#library to extract array job id
import sys

#libraries to get files and move directories
#import glob
#import shutil
#import os

#libraries to extract frame
import pandas as pd
import cv2
import numpy as np

#libraries to undergo differencing
import torch
from torchvision import transforms
from PIL import Image
#from torchvision.utils import save_image
#import skimage.metrics

#create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()

#image working directory
img_dir = "../../data/Kakadu_fish/training_data/images/"
#video directory
vid_dir = "../../data/Kakadu_fish/surrounding_videos/"
#where to save scores
score_dir = "../../data/Kakadu_fish/other/matching_images/scores/stage_2_scores/"
#where location data lies
loc_dir = "../../data/Kakadu_fish/other/matching_images/"
#matching stage 1
match_stage_1 = "../../data/Kakadu_fish/other/matching_images/matching_stage_1.csv"

#load image names and vid locations
m_st_1_dat = pd.read_csv(match_stage_1)


#choose image number
#im_num = int(sys.argv[1]) -1
im_num = 1



#create vectors
im_name = []
vid_loc = []
RMSE_val = 50
frame_nums = 0


#Load image
img = Image.open(img_dir +m_st_1_dat["image_name"][im_num])

if(m_st_1_dat["flipped"][im_num]=="yes"):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

img_tens = convert_tensor(img)

#load video
video = m_st_1_dat["video_location"][im_num]
vid = cv2.VideoCapture(video)

#count how many frames there are and start a counter as we go through the frames
num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
temp_frame = 0

while(1):
    
      
    # Capture the video frame
    # by frame
    success, image = vid.read()
    
    if ((temp_frame==num_frames)or(RMSE_val==0)):
            break
    
    temp_frame += 1
    #convert to tensor
    
    try:
        if image is not None:
            frame_tens = convert_tensor(image)
            
            #fix the frame before comparing with raw image
            frame_tens_fixed = torch.stack((frame_tens[2,:,:],frame_tens[1,:,:],frame_tens[0,:,:]),dim=0)
            
            #Calculate RMSE            
            RMSE_val_raw = sum(sum(sum((frame_tens_fixed - img_tens)**2))).numpy()**0.5
            
            if(RMSE_val_raw<RMSE_val):
                RMSE_val=RMSE_val_raw
                frame_nums = temp_frame
    except:
        pass

 
sim_dat = pd.DataFrame({'image_name':m_st_1_dat["image_name"][im_num],'video_location':video,'flipped':m_st_1_dat["flipped"][im_num],'frame_number':frame_nums,'RMSE':RMSE_val},index=[0])

sim_dat.to_csv(score_dir + m_st_1_dat["image_name"][im_num].split(".")[0] + "_image_sim_scores.csv",index=False)