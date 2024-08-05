#!/bin/bash

#SBATCH -A IscrC_EMPI 
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=test #Job Name
#SBATCH --error=test_error%j_err #Error dump file
#SBATCH --output=test_output%j_out
#SBATCH --time 00:10:00 #Be careful with this !!                          
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:2
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fmerenda2@studenti.unisa.it # Notification via email
module unload cuda

module load gcc
module load cuda/12.3

sh oneapi-for-nvidia-gpus-2024.2.0-cuda-12.0-linux.sh --install-dir /leonardo/home/userexternal/fmerenda/intel/oneapi/
source /leonardo/pub/userexternal/bcosenza/oneapi/setvars.sh --include-intel-llvm --force
sycl-ls

echo "--------------------------- Testing MPI - Cineca Test ----"
srun --cpu-freq=high -N 1 --ntasks-per-node=1 ./build/hpc-final-project
# srun --cpu-freq=high -N 1 --ntasks-per-node=1 ./build/device-query-sycl
