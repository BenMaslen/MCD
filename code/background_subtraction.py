# importlibraries

# libraries to extract frames
import os
import pandas as pd
import cv2
import numpy as np
import shutil

# libraries to undergo image augmentation
import torch
from torchvision import transforms
from PIL import Image
import pickle as pk

# argument library
import argparse

parser = argparse.ArgumentParser(
    description="This script performs background subtraction on images with surrounding images \
    using the K Nearest Neighbours (KNN) algorithm from open-CV."
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
    "-ad",
    "--augment_directory",
    help="Directory where differenced images will be saved",
    required=True,
)
parser.add_argument("-n", "--name", help="Differenced image folder name", required=True)
parser.add_argument(
    "-r",
    "--red",
    help="Whether to replace the red layer or the entire image (TRUE just replaces red layer - defaults to False)",
    action="store_true",
)
parser.add_argument(
    "-k",
    "--kernel",
    help="Kernel for open/close morphology operations. Default is np.ones((5,5),np.uint8).",
    default=np.ones((5, 5), np.uint8),
)
parser.add_argument(
    "-o",
    "--open",
    help="Whether to use 'open' morphological tranformation with associated kernel. Default is False.",
    action="store_true",
)
parser.add_argument(
    "-cl",
    "--close",
    help="Whether to use 'close' morphological tranformation with associated kernel. Default is False.",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--pca",
    help="Whether to undergo PCA with whitening on the three RGB frames to reduce to two. In order to use this option, you need to \
    run the pca_script first on the training data and save the pca.pkl file in the differenced_directory. Defaults to false.",
    action="store_true",
)
parser.add_argument(
    "-td",
    "--threshold_dist",
    help="Distance to threshold for KNN background subtraction. Defaults to 400.",
    default=400,
    type=int,
)
parser.add_argument(
    "-li",
    "--lead_in",
    help="Length of the history for KNN background subtraction. Defaults to 500.",
    default=500,
    type=int,
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

# directory where augmented images will be saved
diff_folder = args.augment_directory

# differenced image folder name
diff_name = args.name

# differenced parameters
red_layer = args.red

# background subtraction parameters

# kernel for open/close morphology operations
kernel = args.kernel
# here are some example kernels:

# square 5x5 kernel:
# kernel = np.ones((5,5),np.uint8)

# circular kernel
# kernel = np.array([[0, 0, 1, 0, 0],
#   [0, 1, 1, 1, 0],
#   [1, 1, 1, 1, 1],
#   [0, 1, 1, 1, 0],
#   [0, 0, 1, 0, 0]], dtype=np.uint8)

# whether to use 'open' morphological tranformation (needs a kernel)
open_morph = args.open

# whether to use 'close' morphological tranformation (needs a kernel)
close_morph = args.close

# lead in
lead_in = args.lead_in

# threshold dist
threshold_dist = args.threshold_dist

# whether to do pca on the three colour layers
do_pca = args.pca

# images
image_loc = training_dat_loc + "images/"
# labels
label_loc = training_dat_loc + "labels/"
# data.yaml directory
yaml_loc = training_dat_loc + "data.yaml"
# differencing location
diff_loc = diff_folder + diff_name + "/"


# prepare augmented image directory:
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

# apply back sub
for im_num in im_num_list:
    target_frame = val_frame_dat["frame"][im_num]

    cap = cv2.VideoCapture(val_frame_dat["vid_location"][im_num])

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - lead_in))

    fgbg = cv2.createBackgroundSubtractorKNN(
        detectShadows=False, history=lead_in, dist2Threshold=threshold_dist
    )

    while 1:
        ret, frame = cap.read()
        if frame is None:
            break

        fgmask = fgbg.apply(frame)

        if open_morph is True or close_morph is True:
            if open_morph is True:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            if close_morph is True:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_number == target_frame:
            im = Image.fromarray(fgmask)
            frame = convert_image(frame)

            if image_size is not False:
                im = im.resize(image_size)
                frame = frame.resize(image_size)

            if red_layer is True:
                img_center_tens = convert_tensor(frame)
                img_center_tens = torch.stack(
                    (
                        img_center_tens[2, :, :],
                        img_center_tens[1, :, :],
                        img_center_tens[0, :, :],
                    ),
                    dim=0,
                )
                im_torch = convert_tensor(im)

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
                    # pca = PCA(n_components=2,whiten=True)
                    reduced_matrix = pca.transform(im_matrix)

                    # Reshape the reduced matrix back into a 3D tensor
                    reduced_image = convert_tensor(
                        np.reshape(reduced_matrix, (im_height, im_width, 2))
                    )

                    # apply sigmoid transform
                    reduced_image = torch.sigmoid(reduced_image)

                    im = convert_image(
                        torch.stack(
                            (
                                im_torch[0, :, :],
                                reduced_image[0, :, :],
                                reduced_image[1, :, :],
                            ),
                            dim=0,
                        )
                    )

                else:
                    im = convert_image(
                        torch.stack(
                            (
                                im_torch[0, :, :],
                                img_center_tens[1, :, :],
                                img_center_tens[2, :, :],
                            ),
                            dim=0,
                        )
                    )

            if time_stamp_loc is not False:
                im_tens = convert_tensor(im)
                im_tens[
                    :,
                    time_stamp_loc[0] : time_stamp_loc[1],
                    time_stamp_loc[2] : time_stamp_loc[3],
                ] = 0
                im = convert_image(im_tens)

            im.save(
                diff_loc
                + "images/"
                + val_frame_dat["split"][im_num]
                + "/"
                + val_frame_dat["image_name"][im_num]
            )
            break
