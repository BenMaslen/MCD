# importlibraries

# libraries to extract surrounding images
import os
import pandas as pd
import numpy as np
import shutil

# libraries to undergo image augmentation
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import pickle as pk
import random as random

# argument library
import argparse

parser = argparse.ArgumentParser(
    description="This script performs three frame differencing images with surrounding images using absolute value, variance, \
    range or direction based measures of spread ('abs','var','range', 'dir'). It can also calculate a 2 frame difference ('2_diff') and background \
    subtraction using frame differencing ('diff_BS')"
)

parser.add_argument(
    "-l", "--location_dat", help="Path to location dat csv", required=True
)
parser.add_argument(
    "-t",
    "--training_loc",
    help="training data folder where yaml file and folders for images & labels are assumed to be within",
    required=True,
)
parser.add_argument(
    "-rs",
    "--resize",
    help="image size (if resizing needed for surrounding images).\
If you do not want to resize surrounding images replace with False (default)",
    nargs="*",
    default=False,
    type=int,
)
parser.add_argument(
    "-s",
    "--time_stamp",
    help="Pixel location of time_stamp in video if wanted to be removed. \
This blacks out top left corner of video to x1 to x2 pixels across and y1 to y2 down. if you do not want to \
remove timestamp, replace with False (default)",
    nargs="*",
    default=False,
    type=int,
)
parser.add_argument(
    "-c",
    "--center_image_surround",
    help="whether to read in center frame from surrounding video or images",
    action="store_true",
)
parser.add_argument(
    "-ad",
    "--augment_directory",
    help="Directory where differenced images will be saved",
    required=True,
)
parser.add_argument("-n", "--name", help="Differenced image folder name", required=True)
parser.add_argument(
    "-bn",
    "--BS_num",
    help="Number of frames to base background subtraction average on.",
    default=10,
    type=int,
)
parser.add_argument(
    "-d",
    "--diff",
    help="Distance in frames between differenced frames and center frame (defaults to 1).",
    default=1,
    type=int,
)
parser.add_argument(
    "-u",
    "--upsilon",
    help="Multiplicative scaling parameter upsilon, to scale resulting differences by.",
    type=int,
    default=1,
)
parser.add_argument(
    "-r",
    "--red",
    help="Whether to replace the red layer or the entire image (TRUE just replaces red layer - defaults to False)",
    action="store_true",
)
parser.add_argument(
    "-m",
    "--method",
    help="Method of differencing. One of 'abs', 'var', 'range', 'dir', '2_diff' or 'diff_BS'. Defaults to 'abs'.",
    default="abs",
    required=True,
)
parser.add_argument(
    "-p",
    "--pca",
    help="Whether to undergo PCA with whitening on the three RGB frames to reduce to two. In order to use this option, you need to \
    run the pca_script first on the training data and save the pca.pkl file in the differenced_directory. Defaults to false.",
    action="store_true",
)

args = parser.parse_args()

# create a function that transforms images to tensors
convert_tensor = transforms.ToTensor()
convert_image = transforms.ToPILImage()

# set directories
image_to_vid_file = args.location_dat

# training data location
# folder with images, labels and yaml file are assumed to be within training data folder
training_dat_loc = args.training_loc

# image size (if resizing needed for surrounding images)
image_size = args.resize

# pixel location of time_stamp in video if wanted to be removed
time_stamp_loc = args.time_stamp

# whether to read in center frame from surrounding video/images
center_image_surround = args.center_image_surround

# directory where augmented images will be saved
diff_folder = args.augment_directory

# differenced image folder name
diff_name = args.name

# differenced parameters
diff = args.diff
upsilon = args.upsilon
method = args.method
red_layer = args.red
do_pca = args.pca
BS_num = args.BS_num

# images
image_loc = training_dat_loc + "images/"
# labels
label_loc = training_dat_loc + "labels/"
# data.yaml directory
yaml_loc = training_dat_loc + "data.yaml"
# differencing location
diff_loc = diff_folder + diff_name + "/"


# prepare augmented image directory
os.mkdir(diff_loc)
os.mkdir(diff_loc + "/images")
os.mkdir(diff_loc + "/images/train")
os.mkdir(diff_loc + "/images/test")
os.mkdir(diff_loc + "/images/valid")

shutil.copytree(label_loc, diff_loc + "/labels")
shutil.copy(yaml_loc, diff_loc + "/data.yaml")

# import image location data
val_frame_dat = pd.read_csv(image_to_vid_file)


im_num_list = list(range(0, val_frame_dat.shape[0]))

for im_num in im_num_list:
    # get image folder
    img_folder = sorted(os.listdir(val_frame_dat["image_folder"][im_num]))

    # get index in image folder
    im_index = img_folder.index(val_frame_dat["image_name"][im_num])

    # extract left and right image
    try:
        left_image_name = img_folder[im_index - diff]
        left_image = Image.open(val_frame_dat["image_folder"][im_num] + left_image_name)
    except:  # noqa: E722
        left_image_name = img_folder[im_index + diff]
        left_image = Image.open(val_frame_dat["image_folder"][im_num] + left_image_name)

    try:
        right_image_name = img_folder[im_index + diff]
        right_image = Image.open(
            val_frame_dat["image_folder"][im_num] + right_image_name
        )
    except:  # noqa: E722
        right_image_name = img_folder[im_index - diff]
        right_image = Image.open(
            val_frame_dat["image_folder"][im_num] + right_image_name
        )

    # give conditions if left or right image does not exist
    if left_image_name[:-7] != val_frame_dat["image_name"][im_num][:-7]:
        left_image = right_image
    if right_image_name[:-7] != val_frame_dat["image_name"][im_num][:-7]:
        right_image = left_image

    if method == "naive_BS_short" or method == "diff_BS_short" or method == "diff_BS":
        BS_count = 0

        background_image_names = os.listdir(
            val_frame_dat["image_folder"][im_num][:-6] + "empty/"
        )
        background_image_names_rand_sample = random.sample(
            background_image_names, BS_num
        )

        for rand_img_name in background_image_names_rand_sample:
            frame_BS_tens = convert_tensor(
                Image.open(
                    val_frame_dat["image_folder"][im_num][:-6]
                    + "empty/"
                    + rand_img_name
                )
            )

            if BS_count == 0:
                frame_BS_tens_total = frame_BS_tens
            else:
                frame_BS_tens_total = frame_BS_tens_total + frame_BS_tens
            BS_count += 1

        # fix the image before differencing
        frame_BS_tens_avg_fixed = frame_BS_tens_total / BS_count

        if image_size is not False:
            frame_BS_tens_avg_fixed = convert_tensor(
                convert_image(frame_BS_tens_avg_fixed).resize(image_size)
            )

    # read in center image
    if center_image_surround is True:
        img_center = Image.open(val_frame_dat["image_location"][im_num])
    else:
        img_center = Image.open(
            image_loc
            + val_frame_dat["split"][im_num]
            + "/"
            + val_frame_dat["image_name"][im_num]
        )

    # convert images to tensors
    img_right_tens_fixed = convert_tensor(right_image)
    img_left_tens_fixed = convert_tensor(left_image)
    img_center_tens = convert_tensor(img_center)

    # fix the left & right image before differencing
    if image_size is not False:
        img_center_tens = convert_tensor(
            convert_image(img_center_tens).resize(image_size)
        )
        img_right_tens_fixed = convert_tensor(
            convert_image(img_right_tens_fixed).resize(image_size)
        )
        img_left_tens_fixed = convert_tensor(
            convert_image(img_left_tens_fixed).resize(image_size)
        )

    # apply three frame differencing
    if method == "abs":
        diff_im = (
            upsilon
            * 0.5
            * (
                abs(img_center_tens - img_right_tens_fixed)
                + abs(img_center_tens - img_left_tens_fixed)
            )
        )

    elif method == "var":
        avg_im = (img_center_tens + img_right_tens_fixed + img_left_tens_fixed) / 3
        diff_im_unscaled = (
            (img_center_tens - avg_im) ** 2
            + (img_right_tens_fixed - avg_im) ** 2
            + (img_left_tens_fixed - avg_im) ** 2
        ) / 2
        diff_im = diff_im_unscaled * (1 / torch.max(diff_im_unscaled)) * upsilon

    elif method == "range":
        red_i = torch.stack(
            (
                img_center_tens[0, :, :],
                img_right_tens_fixed[0, :, :],
                img_left_tens_fixed[0, :, :],
            ),
            dim=0,
        )
        green_i = torch.stack(
            (
                img_center_tens[1, :, :],
                img_right_tens_fixed[1, :, :],
                img_left_tens_fixed[1, :, :],
            ),
            dim=0,
        )
        blue_i = torch.stack(
            (
                img_center_tens[2, :, :],
                img_right_tens_fixed[2, :, :],
                img_left_tens_fixed[2, :, :],
            ),
            dim=0,
        )

        red_r = torch.max(red_i, dim=0)[0] - torch.min(red_i, dim=0)[0]
        green_r = torch.max(green_i, dim=0)[0] - torch.min(green_i, dim=0)[0]
        blue_r = torch.max(blue_i, dim=0)[0] - torch.min(blue_i, dim=0)[0]

        diff_im = torch.stack((red_r, green_r, blue_r), dim=0) * upsilon

    elif method == "dir":
        diff_im = (
            upsilon
            * 0.25
            * (2 * img_right_tens_fixed - img_center_tens - img_left_tens_fixed)
            + 0.5
        )

    elif method == "2_diff":
        diff_im = upsilon * 0.5 * (img_center_tens - img_left_tens_fixed) + 0.5

    elif method == "diff_BS":
        diff_im = upsilon * 0.5 * (img_center_tens - frame_BS_tens_avg_fixed) + 0.5
    else:
        print(
            "Please choose one of 'abs', 'var', 'range', 'dir', '2_diff', or 'diff_BS' for method."
        )
        break

    if red_layer is True:
        diff_im_avg = (diff_im[0, :, :] + diff_im[1, :, :] + diff_im[2, :, :]) / 3
        if do_pca is True:
            if im_num == 0:
                pca = pk.load(open(diff_folder + "pca.pkl", "rb"))

            # Get the image shape
            (
                im_depth,
                im_height,
                im_width,
            ) = img_center_tens.shape

            # Reshape the image tensor into a 2D matrix
            im_matrix = np.reshape(
                convert_image(img_center_tens), (im_height * im_width, im_depth)
            )

            # Perform PCA on the matrix
            reduced_matrix = pca.transform(im_matrix)

            # Reshape the reduced matrix back into a 3D tensor
            reduced_image = convert_tensor(
                np.reshape(reduced_matrix, (im_height, im_width, 2))
            )

            # apply sigmoid transform
            reduced_image = torch.sigmoid(reduced_image)

            diff_im_final = torch.stack(
                (diff_im_avg, reduced_image[0, :, :], reduced_image[1, :, :]), dim=0
            )
        else:
            diff_im_final = torch.stack(
                (diff_im_avg, img_center_tens[1, :, :], img_center_tens[2, :, :]), dim=0
            )
    else:
        diff_im_final = diff_im

    if time_stamp_loc is not False:
        diff_im_final[
            :,
            time_stamp_loc[0] : time_stamp_loc[1],
            time_stamp_loc[2] : time_stamp_loc[3],
        ] = 0

    # save image
    save_image(
        diff_im_final,
        diff_loc
        + "/images/"
        + val_frame_dat["split"][im_num]
        + "/"
        + val_frame_dat["image_name"][im_num],
    )
