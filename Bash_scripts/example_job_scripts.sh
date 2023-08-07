#SBATCH --job-name=yolov5s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --time=03:30:00

#SBATCH --gres=gpu:4
##SBATCH --mail-type=ALL
##SBATCH --mail-user=b.maslen@unsw.edu.au

cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/yolov8

module load python/3.9.4

source yolov8_venv/bin/activate

yolov8_venv/bin/yolo detect train data=../../data/Tassy_BRUV/augmented_images/abs_alpha_7_diff_2_red/data.yaml \
model=models/yolov8n.pt epochs=5 imgsz=1280 device=0,1,2,3 workers=8 \
name=../../../../results/Tassy_BRUV/train/tassy_bruv_8m_abs_alpha_7_diff_2_red_5

python metrics.py -m '../../results/Tassy_BRUV/tassy_bruv_8m_raw_300_img_sz_640/weights/best.pt'

yolov8_venv/bin/yolo detect train data=../../data/Tassy_BRUV/training_data/data.yaml \
model=../../results/Tassy_BRUV/tassy_bruv_8l_raw_300_img_sz_1080/weights/best.pt epochs=300 imgsz=1080 device=0,1,2,3 workers=8 \
name=../../../../results/Tassy_BRUV/tassy_bruv_8l_raw_600_img_sz_1080

python metrics.py -m '../../results/Tassy_BRUV/tassy_bruv_8l_raw_600_img_sz_1080/weights/best.pt'



#!/bin/bash -l
#SBATCH --account=OD-220461
#SBATCH --job-name=yolov8l_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=00:20:00

#SBATCH --gres=gpu:1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=b.maslen@unsw.edu.au

cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/yolov8

module load python/3.9.4

source yolov8_venv/bin/activate

yolov8_venv/bin/yolo detect predict source=../../data/Deepfish/augmented_images/range_alpha_0_diff_1_r_pca_f/images/valid/ \
model=../../results/Deepfish/range_alpha_0_diff_1_r_pca_f/weights/best.pt save=True save_txt=True conf=0.5


#!/bin/bash -l
#SBATCH --account=OD-220461
#SBATCH --job-name=yolov8l_pred
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=00:20:00

#SBATCH --gres=gpu:1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=b.maslen@unsw.edu.au

cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/yolov8

module load python/3.9.4

source yolov8_venv/bin/activate

mkdir ../../results/Deepfish/range_alpha_0_diff_1_r_pca_f/val_images/

yolov8_venv/bin/yolo detect predict source=../../data/Deepfish/augmented_images/range_alpha_0_diff_1_r_pca_f/images/valid/ \
model=../../results/Deepfish/range_alpha_0_diff_1_r_pca_f/weights/best.pt save=True save_txt=True conf=0.5 \
name=../../../../results/Deepfish/range_alpha_0_diff_1_r_pca_f/val_images/


module load python/3.11

source diff_cond_11_venv/bin/activate


#!/bin/bash -l
#SBATCH --account=OD-220461
#SBATCH --job-name=differencing_TB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000gb
#SBATCH --time=1:00:00

##SBATCH --mail-type=ALL
##SBATCH --mail-user=b.maslen@unsw.edu.au

cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/augmentation/

module load python/3.11

source diff_cond_11_venv/bin/activate

python -m pca_script -l '../../data/Tassy_BRUV/location_data/location_dat.csv'\
 -t '../../data/Tassy_BRUV/training_data/' -c \
 -di '../../data/Tassy_BRUV/augmented_images/' -n 800



#!/bin/bash -l
#SBATCH --account=OD-220461
#SBATCH --job-name=differencing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=1000gb
#SBATCH --time=5:00:00

##SBATCH --mail-type=ALL
##SBATCH --mail-user=b.maslen@unsw.edu.au

cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/augmentation/

module load python/3.11

source diff_cond_11_venv/bin/activate

python -m pca_script -l '../../data/Tassy_BRUV/location_data/location_dat.csv'\
 -t '../../data/Tassy_BRUV/training_data/' -c \
 -di '../../data/Tassy_BRUV/augmented_images/pca_1000/' -n 1000

python -m pca_script -l '../../data/Tassy_BRUV/location_data/location_dat.csv'\
 -t '../../data/Tassy_BRUV/training_data/' -c \
 -di '../../data/Tassy_BRUV/augmented_images/' -n 1000











