#!/bin/bash
#SBATCH --time=0:3:0
#SBATCH --account=def-mahsa77
#SBATCH --job-name=4k8k
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=out_file.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=noobfromkz@gmail.com


DATA_DIR='/home/adias/projects/def-mahsa77/dataset/8K/'

module load python/3.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index torchvision
pip install -U pip
pip install -U setuptools
pip install --no-index opencv-python

python esrgan.py --batch_size 1 --hr_height 64 --hr_width 64 --dataset_path $DATA_DIR


