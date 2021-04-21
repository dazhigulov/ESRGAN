#!/bin/bash
#SBATCH --time=72:0:0
#SBATCH --account=def-mahsa77
#SBATCH --job-name=4k8k
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j-%x.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=weskerlyw@gmail.com



module load python/3.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index torchvision
pip install -U pip
pip install -U setuptools
pip install --no-index opencv-python

python esrgan.py --batch_size 4 --hr_height 512 --hr_width 512 --channels 3 --dataset_path '/home/yuweiluo/projects/def-mahsa77/dataset/8K/'


