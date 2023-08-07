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
score_dir = "../../data/Kakadu_fish/other/matching_images/scores/1000_frames/"
#where location data lies
loc_dir = "../../data/Kakadu_fish/other/matching_images/"

#load image names and vid locations
img_dat = pd.read_csv(loc_dir + "image_names.csv")
video_dat = pd.read_csv(loc_dir + "vid_locations.csv")

img_list = img_dat["image_name"]
videos = video_dat["video_location"]


#choose vid and image number
#num_vals = 10
num_vals = len(videos)
#vid_num = 0
im_num = int(sys.argv[1]) -1
#im_num = 1
#choose frame number/s
#frame_num = 18000
frame_step = 15000



#create vectors
im_name = []
vid_loc = []
flipped = []
RMSE_val = 1000
frame_nums = []

#Load image
img = Image.open(img_dir +img_list[im_num])
img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
img_tens = convert_tensor(img)
img_tens_flipped = convert_tensor(img_flipped)


for vid_num in range(num_vals):
    try:
        #read in video
        video = videos[vid_num]
        vidcap = cv2.VideoCapture(video)
        
        for frame_num in range(frame_step,int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)),frame_step):
            try:
                #extract frame
                frame = frame_num
                vidcap.set(1, frame)
                success,image = vidcap.read()

                #convert to tensor
                frame_tens = convert_tensor(image)

                #fix the frame before comparing with raw image
                frame_tens_fixed = torch.stack((frame_tens[2,:,:],frame_tens[1,:,:],frame_tens[0,:,:]),dim=0)
                
                #compare similarity

                #root mean square error
                RMSE_val_raw = sum(sum(sum((frame_tens_fixed - img_tens)**2))).numpy()**0.5
                RMSE_val_raw_flipped = sum(sum(sum((frame_tens_fixed - img_tens_flipped)**2))).numpy()**0.5
                
                if(RMSE_val_raw<RMSE_val):
                    RMSE_val=RMSE_val_raw
                    flipped="no"
                    frame_nums = frame_num
                    vid_loc = video
                    
                if(RMSE_val_raw_flipped<RMSE_val):
                    RMSE_val=RMSE_val_raw_flipped
                    flipped="yes"
                    frame_nums = frame_num
                    vid_loc = video
            except:
                pass
    except:
        pass
                    

im_name = [img_list[im_num]]

sim_dat = pd.DataFrame({'image_name':im_name,'video_location':vid_loc,'flipped':flipped,'frame_number':frame_nums,'RMSE':RMSE_val})

sim_dat.to_csv(score_dir + img_list[im_num].split(".")[0] + "_image_sim_scores.csv",index=False)