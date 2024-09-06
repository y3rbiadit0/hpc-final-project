#!/bin/bash

#SBATCH -A IscrC_EMPI 
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=test #Job Name
#SBATCH --error=test_error%j_err #Error dump file
#SBATCH --output=test_output%j_out
#SBATCH --time 00:00:20 #Be careful with this !!                          
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:2
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=FILL_ME # Notification via email
module unload cuda

module load gcc
module load cuda/12.3

echo "Setup OneAPI - NVidia GPUs... - Running Script"
sh oneapi-for-nvidia-gpus-2024.2.0-cuda-12.0-linux.sh --install-dir /leonardo/home/userexternal/fmerenda/intel/oneapi/ >> /dev/null
echo "OK"
echo "Setup OneAPI Vars Script ... - Running Script"
source /leonardo/home/userexternal/fmerenda/intel/oneapi/setvars.sh --include-intel-llvm --force >> /dev/null
echo "OK"
sycl-ls

echo "--------------------------- Testing SYCL/CUDA - Cineca Test ---------------------------"
srun --cpu-freq=high -N 1 --ntasks-per-node=1 ./build/src/gpu_to_gpu_single_node/single-node-sycl-cuda
