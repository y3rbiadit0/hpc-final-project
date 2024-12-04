#!/bin/bash

#SBATCH -A IscrC_NETTUNE_0
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=intra_node #Job Name
#SBATCH --error=./results/intra_node_error_%j_err.txt #Error dump file
#SBATCH --output=./results/intra_node_output_%j_out.txt
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

sh ./scripts/oneapi-for-nvidia-gpus-2025.0.0-cuda-12.0-linux.sh --install-dir /leonardo/home/userexternal/fmerenda/intel/oneapi/ >> /dev/null
source /leonardo/home/userexternal/fmerenda/intel/oneapi/setvars.sh --include-intel-llvm --force >> /dev/null

srun --cpu-freq=high -N 1 --ntasks-per-node=1 /leonardo/home/userexternal/fmerenda/hpc-final-project/build/src/gpu_to_gpu_single_node/single-node-sycl-cuda