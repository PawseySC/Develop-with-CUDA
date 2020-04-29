#!/bin/bash --login
#SBATCH --reservation=courseq
#SBATCH --partition=gpuq
#SBATCH --tasks=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=00:05:00
#SBATCH --export=NONE
 
module load cuda

srun --export=all -n 1 make
