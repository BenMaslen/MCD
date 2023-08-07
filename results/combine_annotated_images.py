import os
from PIL import Image

# Define the paths to the 4 directories containing the input images
dir1 = "../../results/Kakadu_fish/result_analysis/valid/labels/"
dir2 = "../../results/Kakadu_fish/result_analysis/valid/raw/"
dir3 = "../../results/Kakadu_fish/result_analysis/valid/back_sub_td_50_li_20_fs_50_pca/"
dir4 = "../../results/Kakadu_fish/result_analysis/valid/naive_alpha_15_diff_100_r_pca/"

# Define the path to the directory where the new images will be saved
save_dir = "../../results/Kakadu_fish/result_analysis/valid/combined_attempt_long_diff_2/"

# Loop over all the images in dir1 (assuming all directories contain the same image filenames)
for filename in os.listdir(dir1):
    if filename.endswith(".jpg"):
        try:
            # Open the images from each directory using PIL
            img1 = Image.open(os.path.join(dir1, filename))
            img2 = Image.open(os.path.join(dir2, filename))
            img3 = Image.open(os.path.join(dir3, filename))
            img4 = Image.open(os.path.join(dir4, filename))

            # Calculate the size of the new image by doubling the width and height of the input images
            width, height = img1.size
            new_size = (2*width, 2*height)

            # Create a new image with the calculated size and paste the input images onto it
            new_img = Image.new("RGB", new_size)
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (width, 0))
            new_img.paste(img3, (0, height))
            new_img.paste(img4, (width, height))

            # Save the new image to the specified directory
            new_filename = os.path.splitext(filename)[0] + "_joined.jpg"
            new_img.save(os.path.join(save_dir, new_filename))
        except:
            pass