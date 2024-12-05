#!/bin/bash

#SBATCH -A IscrC_NETTUNE_0 
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=inter_node #Job Name
#SBATCH --error=./results/inter_node_error_%j_err.txt #Error dump file
#SBATCH --output=./results/inter_node_output_%j_out.txt
#SBATCH --time 00:15:00 #Be careful with this !!                          
#SBATCH --nodes=2                                
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:1
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=FILL_ME # Notification via email

module load cuda
module load openmpi/4.1.6--nvhpc--23.11

sh ./scripts/oneapi-for-nvidia-gpus-2025.0.0-cuda-12.0-linux.sh --install-dir /leonardo/home/userexternal/fmerenda/opt/intel/oneapi/ >> /dev/null
source /leonardo/home/userexternal/fmerenda/opt/intel/oneapi/setvars.sh --include-intel-llvm --force >> /dev/null

mpirun -np 2 ./build/src/gpu_to_gpu_two_nodes/multiple-nodes-sycl-cuda-mpi