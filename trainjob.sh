#!/bin/bash
#SBATCH --time=00:3:00
#SBATCH --account=def-mahsa77
#SBATCH --job-name=ITMO
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
module load arch/avx512 StdEnv/2018.3
nvidia-smi


module load cuda cudnn
source ~/pytorch/bin/activate

python esrgan.py > result.txt

