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

#create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

#Kakadu fish

#Imgae_to_video_locator_csv
image_to_vid_file = '../../data/Kakadu_fish/other/matching_images/stage_2_scores.csv'
#updated Imgae_to_video_locator_csv
image_to_vid_file_updated = '../../data/Kakadu_fish/other/matching_images/stage_2_scores_updated.csv'
#video location
vid_loc = '../../data/Kakadu_fish/surrounding_videos/'
#image location
image_loc = '../../data/Kakadu_fish/training_data/images/'
#labels
label_loc = '../../data/Kakadu_fish/training_data/labels/'
#data.yaml directory
yaml_loc =  '../../data/Kakadu_fish/training_data/data.yaml'
#image size (if resizing needed for surrounding images)
image_size = [1920,1080]
#directory where differenced images will be saved
diff_loc = '../../data/Kakadu_fish/augmented_images/red_diff_7/'


#import data
val_frame_dat = pd.read_csv(image_to_vid_file)
diffs = [0]*val_frame_dat.shape[0]

num_images = 50


for im_num in range(45,num_images):
    
    diff=202
    #read in video

    #kakadu Fish

    #read in video
    video = val_frame_dat['vid_folder'][im_num] + val_frame_dat['vid_name'][im_num]
    vidcap = cv2.VideoCapture(video)

    #read in center frame:
    img_center = Image.open(image_loc + val_frame_dat['image_name'][im_num])
    img_center_tens_fixed = convert_tensor(img_center)
    img_center_tens_fixed = convert_tensor(convert_image(img_center_tens_fixed).resize(image_size))


    if(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)==53100):
        diffs[im_num] = diff
    else:
        guess = round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/260)
        buffer = 10
        search_nums = range(guess-buffer,guess+buffer)
        
        for diff in search_nums:
            try:
                center_frame = val_frame_dat['frame'][im_num]
                frame2 = val_frame_dat['frame'][im_num] - diff
                vidcap.set(1, frame2)
                success,image2 = vidcap.read()
                #cv2.imwrite("../data/test/frame%d.jpg" % frame, image)
                print('Read a new frame: ', success)
            except:
                pass
            
            img_left_tens = convert_tensor(image2)
            img_left_tens_fixed = torch.stack((img_left_tens[2,:,:],img_left_tens[1,:,:],img_left_tens[0,:,:]),dim=0)
            img_left_tens_fixed = convert_tensor(convert_image(img_left_tens_fixed).resize(image_size))
            
            if(sum(sum(sum((img_center_tens_fixed-img_left_tens_fixed)**2)))**(0.5) == val_frame_dat['RMSE'][im_num]):
                diffs[im_num] = diff
                break
        
val_frame_dat['diffs'] = diffs

val_frame_dat.to_csv(image_to_vid_file_updated,index=False)