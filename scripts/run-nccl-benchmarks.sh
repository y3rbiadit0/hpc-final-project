#!/bin/bash

#SBATCH -A IscrC_NETTUNE_0
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=nccl_tests #Job Name
#SBATCH --error=./results/nccl_tests_%j_err.txt #Error dump file
#SBATCH --output=./results/nccl_tests_%j_out.txt
#SBATCH --time 01:00:00 #Be careful with this !!                          
#SBATCH --nodes=2                              
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:1
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=FILL_ME # Notification via email


module load openmpi
module load cuda
module load nccl
# export NCCL_HOME=/leonardo/home/userexternal/fmerenda/nccl-2.19.3-1-cuoct3jempfrtirmnjwtxwr2wwgqrrbv

# ./nccl-tests-binaries/sendrecv_perf -b 8 -e 2048M -f 2 -g 1

NCCL_DEBUG=Info mpirun -np 2 /leonardo/home/userexternal/fmerenda/opt/nccl-tests/build/sendrecv_perf -b 8 -e 2048M -f 2 -g 1f