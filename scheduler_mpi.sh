#!/bin/bash

#SBATCH -A IscrC_EMPI 
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=test #Job Name
#SBATCH --error=test_error%j_err #Error dump file
#SBATCH --output=test_output%j_out
#SBATCH --time 00:00:20 #Be careful with this !!                          
#SBATCH --nodes=2                                
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:1
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fmerenda2@studenti.unisa.it # Notification via email

module load cuda
module load openmpi/4.1.6--nvhpc--23.11

echo "--------------------------- Testing MPI - Cineca Test ----"
sycl-ls
mpirun -np 2 ./build/hpc-final-project-mpi