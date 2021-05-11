#!/bin/bash
#SBATCH -J job_name       	# job name, don't use spaces
#SBATCH --partition=main       	# main is the safe option
#SBATCH -c 2                   	# number of cores/cpu's per task, 2
#SBATCH --gres=gpu:1            # number of gpus 1
#SBATCH --mem=12400	        # use 12 GB mem


#SBATCH --mail-type=END,FAIL   # email status changes
#SBATCH --time=14-1:00:00      # time limit 14 days
#SBATCH --constraint="avx"     # use avx2 partition

# Choose your file folder
cd ~/file_folder
# add anaconda to slurm
module load anaconda3
# add openmpi module to slurm
module load openmpi/1.10.2
# add cuda for gpu computations
module load nvidia/cuda-10.1  
# open anaconda environment
source activate opensim-rl
mpirun -n 1 python3 file_test.py
