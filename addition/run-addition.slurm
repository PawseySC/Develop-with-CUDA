#!/bin/bash --login
#SBATCH --reservation=courseq
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=p100
#SBATCH --time=00:05:00
#SBATCH --export=NONE
 
module load broadwell gcc/5.4.0 cuda

srun --export=all -n 1 ./addition > addition.txt 
