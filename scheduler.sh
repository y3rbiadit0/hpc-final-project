#!/bin/bash

#SBATCH -A IscrC_EMPI 
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=test #Job Name
#SBATCH --error=test_error%j_err #Error dump file
#SBATCH --output=test_output%j_out
#SBATCH --time 00:00:20 #Be careful with this !!                          
#SBATCH --nodes=1                                
#SBATCH --ntasks-per-node=2 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:2
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=FILL_WITH_EMAIL # Notification via email

module load openmpi/4.1.6--gcc--12.2.0
module load cuda/12.1

echo "--------------------------- Testing MPI - Cineca Test ----"
srun --cpu-freq=high -N 2 --ntasks-per-node=2 ./hello_world_mpi
