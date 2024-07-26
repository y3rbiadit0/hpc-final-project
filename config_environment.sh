module unload openmpi
#module load intel-oneapi-compilers/2023.2.1
source /leonardo/home/userexternal/fmerenda/intel/oneapi/setvars.sh --force
module load openmpi/4.1.6--nvhpc--23.11
module load cuda/12.3
