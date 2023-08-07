#Script to log into to CSIRO HPC 

#login
ssh mas203@pearcey.hpc.csiro.au
ssh mas203@petrichor.hpc.csiro.au
ssh mas203@bracewell.hpc.csiro.au

#change to bowen cloud storage
cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/code/
cd /datasets/work/d61-icv-rpsa/work/Ben_M/Movement_OD_WD/data/Tassy_BRUV/augmented_images

cp -r back_sub_td_100_li_100_pca_b back_sub_td_100_li_100_pca_b_one_class

#scratch
cd /scratch1/mas203/

#to start a new vnc
vnc -s

#job history
sacct -u mas203

#job history for specific date
sacct -S 2023-02-18 -u mas203
sacct -u mas203

sacct -j JOBID

#check how much walltime your job has left
squeue -h -j 19265022 -o "%L"

#create virtual env and install requirements
module load python/3.9.4
python -m virtualenv yolov7_venv
python -m virtualenv yolov8_venv
python -m virtualenv diff_cond_venv
source yolov7_venv/bin/activate
source yolov8_venv/bin/activate
source augm_venv/bin/activate
source diff_cond_venv/bin/activate
which python; which pip
python -m pip install -r requirements.txt
python -m pip install setuptools==59.5.0
python -m pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
python -m pip install -U scikit-learn
python -m pip install -U scikit-learn

#installing flownet2
module load python/3.9.4
python -m virtualenv fn2_p37_venv
python -m virtualenv fn2_p39_venv
source fn2_p37_venv/bin/activate
source fn2_p39_venv/bin/activate
which python; which pip
module load pytorch/1.10.0-py39-cuda112
module unload pytorch/1.10.0-py39-cuda112


module load pytorch/2.0.0-py39-cuda121 #doesn't work
module unload pytorch/1.11.0-py39-cuda112  #doesn't work
module unload pytorch/1.10.0-py39-cuda112  #doesn't work
module unload pytorch/1.12.1-py39-cuda112 #doesn't work
module load pytorch/1.8.1-py39-cuda112 #doesn't work

python -m pip install typing_extensions
python -m pip install icpc
python -m pip install filelock jinja2 networkx sympy
python -m pip install numpy

bash install.sh


#submit the job
sbatch ssd_resnet50_job

#check the status of the job
squeue -u mas203

#to cancel a job
scancel #jobid

#rsync yolo runs
rsync -r runs /datasets/work/d61-icv-rpsa/work/Ben_M/Object_differencing/yolo_v5

#code to test images

python3.9 detect.py --weights runs/train/deepfish_5s_raw_s/weights/best.pt --img 1280 --conf 0.5 --source data/deepfish_data/raw_s/images/test --save-txt --name deepfish_5s_raw_s

#code to validate

python3.9 val.py --weights runs/train/deepfish_5s_raw_s/weights/best.pt --img 1280 --data deepfish.yaml --name deepfish_5s_raw_s_t --task 'test'

#yaml file path

cd ../../data/Tassy_BRUV/training_data/

#project code

#SBATCH --account=OD-220461

sinteractive -J tassy_bruv_interactive -n 1 -t 00:10:00 -m 128gb -g gpu:1 -A: OD-220461


