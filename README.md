# Moving Critter Detection (MCD)

This repository contains code to augment image data using movement quantification methods, as a means to improve object detection algorithms. For more details about the methodology please refer to the paper titled 'The Motion Picture: Leveraging Movement to Enhance AI Object Detection in Ecology' (not yet submitted).


### Set up
In order to run the scripts in this repository, first clone the repository, then using a python virtual environment (or conda environment), install the packages in the requirements.txt file. 
This can be done using the below code:
```
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```

### Code files

Within the code repository, are the background_subtraction.py, differencing.py and optical_flow.py scripts which enable users to augment annotated images for object detection using background subtraction, frame differencing and optical flow respectively. These require the videos the annotated images were sourced from, as well as the name of the video and frame location. It also assumes your annotated images are in YOLO txt format (see below). This information should be recorded in a csv file with the same format (importantly - the same column names) as the example location file in data/location_data/location_dat_surrounding_videos.csv. The differencing_surrounding_images.py and optical_flow_surrounding_images.py scripts augment images using frame differencing and optical flow respectively, however assume you have surrounding images, and have a location file in the format of data/location_data/location_dat_surrounding_images.csv. Finally the pca_script.py performs Principal Component Analysis (PCA) on your images, which will need to be executed prior to using PCA as an option in any of the movmeent augmentation scripts.

In order to find out more about the individual movement augmentation script files and their options, simply run their help files through:
```
python <movement_script>.py -h
```
The augmentation_script_execution_e_g.txt script provides example code for using the movement augmentation scripts and the YOLOv8l_implimentation_e_g.txt script provides example code for running the annotated images with YOLOv8l, as well as running the tracking algorithm ByteTrack. For further details please see https://github.com/ultralytics/ultralytics.

### Example data

To trial these methods on a benchmark dataset, please refer to the Tassie BRUV data set on DRYAD (yet to be uploaded), which was implimented in the paper 'The Motion Picture: Leveraging Movement to Enhance AI Object Detection in Ecology' (not yet submitted) and is already set up in the required format.


#### YOLO txt format
The movement augmentation scripts assume your images are labelled in yolo txt format. An example of this format can be found in the training data in the Tassie BRUV Dryad repository (yet to be uploaded). The directory should be set up as follows:

training_dat/
  - data.yaml
  - images/
    - test/
    - train/
    - valid/
  - labels/
    - test/
    - train/
    - valid/

