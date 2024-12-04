#!/bin/bash

#SBATCH -A IscrC_NETTUNE_0
#SBATCH -p boost_usr_prod #Partition
#SBATCH --job-name=osu_benchmark #Job Name
#SBATCH --error=./results/osu_benchmark_%j_err.txt #Error dump file
#SBATCH --output=./results/osu_benchmark_%j_out.txt
#SBATCH --time 01:00:00 #Be careful with this !!                          
#SBATCH --nodes=2                                
#SBATCH --ntasks-per-node=1 #Number of cpu cores by each node  - we can check on leonardo maximum number - 4 cores/ 1 for gpu                
#SBATCH --gres=gpu:1
# #SBATCH --mem-per-cpu=0 # This allocates memory for a dedicated way kind of
#SBATCH --profile=All
#SBATCH --mail-type=ALL
#SBATCH --mail-user=FILL_ME # Notification via email


module load openmpi/4.1.6--nvhpc--23.11
module load osu-micro-benchmarks/7.3--openmpi--4.1.6--nvhpc--23.11

# OSU Benchmark by default
OSU_BENCHMARK="osu_latency"

# Default Parameters for OSU Benchmark
MESSAGE_SIZE="1073741824" # 1 GiB
WARMUP_ITERATIONS="100"  # Warm-up iterations
MAIN_ITERATIONS="100"    # Main test iterations
DEVICE="cuda D D"        # Communication between CUDA devices

# Parse command-line arguments
while getopts "t:m:w:i:d:" opt; do
  case $opt in
    t) # Test name (e.g., osu_latency, osu_bw)
      OSU_BENCHMARK=$OPTARG
      ;;
    m) # Message size (in bytes)
      MESSAGE_SIZE=$OPTARG
      ;;
    w) # Warm-up iterations
      WARMUP_ITERATIONS=$OPTARG
      ;;
    i) # Main iterations
      MAIN_ITERATIONS=$OPTARG
      ;;
    d) # Device configuration
      DEVICE=$OPTARG
      ;;
    *)
      echo "Usage: $0 [-t <test_name>] [-m <message_size>] [-w <warmup_iterations>] [-i <main_iterations>] [-d <device>]"
      exit 1
      ;;
  esac
done


# GPUDirect RDMA + GDR Copy (both enabled)
mpirun -np 2 -x UCX_NET_DEVICES=all  -x UCX_IB_GPU_DIRECT_RDMA=1 -x UCX_TLS=rc,cuda_copy,gdr_copy -x UCX_RNDV_THRESH=1024  $OSU_BENCHMARK -m $MESSAGE_SIZE -x $WARMUP_ITERATIONS -i $MAIN_ITERATIONS -d $DEVICE

# GPUDirect RDMA disabled and GDR Copy enabled
mpirun -np 2 -x UCX_NET_DEVICES=all  -x UCX_IB_GPU_DIRECT_RDMA=0 -x UCX_TLS=rc,cuda_copy,gdr_copy -x UCX_RNDV_THRESH=1024  $OSU_BENCHMARK -m $MESSAGE_SIZE -x $WARMUP_ITERATIONS -i $MAIN_ITERATIONS -d $DEVICE

# GPUDirect RDMA disabled and GDR Copy disabled
mpirun -np 2 -x UCX_NET_DEVICES=all  -x UCX_IB_GPU_DIRECT_RDMA=0 -x UCX_TLS=rc,cuda_copy -x UCX_RNDV_THRESH=1024  $OSU_BENCHMARK -m $MESSAGE_SIZE -x $WARMUP_ITERATIONS -i $MAIN_ITERATIONS -d $DEVICE
