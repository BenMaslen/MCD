#importlibraries

#libraries to get files and move directories
import glob
import shutil
import os


#libraries to extract frame
import pandas as pd
import cv2
import numpy as np

#libraries to undergo differencing
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import skimage.metrics

#create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

#image working directory
img_dir = "../../data/Kakadu_fish/training_data/images/"
#video directory
vid_dir = "../../data/Kakadu_fish/surrounding_videos/"
#where to save scores
score_dir = "../../data/Kakadu_fish/other/matching_images/scores/"
#where location data lies
loc_dir = "../../data/Kakadu_fish/other/matching_images/"

#load image names and vid locations
img_dat = pd.read_csv(loc_dir + "image_names.csv")
video_dat = pd.read_csv(loc_dir + "vid_locations.csv")

img_list = img_dat["image_name"]
videos = video_dat["video_location"]


#choose vid and image number
num_vals = 50
#num_vals = len(img_list)
vid_num = 0
#im_num = 0
#choose frame number/s
frame_num = 100

#Create loop over videos

#create vectors
im_name = []
vid_loc = []
flipped = []
RMSE_vals = []
SSIM_vals = []

#read in video
video = videos[vid_num]
vidcap = cv2.VideoCapture(video)

#extract frame
frame = frame_num
vidcap.set(1, frame)
success,image = vidcap.read()

#convert to tensor
frame_tens = convert_tensor(image)

#fix the frame before comparing with raw image
frame_tens_fixed = torch.stack((frame_tens[2,:,:],frame_tens[1,:,:],frame_tens[0,:,:]),dim=0)


#prepare images for similarity metrics
frame_sim = skimage.img_as_float(convert_image(frame_tens_fixed))


for im_num in range(num_vals):
    try:
        #Load image
        img = Image.open(img_dir +img_list[im_num])
        img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_tens = convert_tensor(img)
        img_tens_flipped = convert_tensor(img_flipped)

        img_sim = skimage.img_as_float(img)
        img_sim_flipped = skimage.img_as_float(img_flipped)

        #compare similarity

        #root mean square error
        RMSE_vals.append(sum(sum(sum((frame_tens_fixed - img_tens)**2))).numpy()**0.5)
        RMSE_vals.append(sum(sum(sum((frame_tens_fixed - img_tens_flipped)**2))).numpy()**0.5)

        #ssim
        SSIM_vals.append(skimage.metrics.structural_similarity(img_sim,frame_sim,data_range=1,channel_axis=2))
        SSIM_vals.append(skimage.metrics.structural_similarity(img_sim_flipped,frame_sim,data_range=1,channel_axis=2))
    except:
        RMSE_vals.append("image_error")
        RMSE_vals.append("image_error")
        SSIM_vals.append("image_error")
        SSIM_vals.append("image_error")
        pass
    
flipped = ["no","yes"]*num_vals
vid_loc = [videos[vid_num]]*num_vals*2
im_name = np.repeat(img_list[:num_vals],2)

sim_dat = pd.DataFrame({'image_name':im_name,'video_location':vid_loc,'flipped':flipped,'RMSE':RMSE_vals,'SSIM':SSIM_vals})

sim_dat.to_csv(score_dir+video_dat["video_name"][vid_num].split(".")[0] + "_video_sim_scores.csv",index=False)