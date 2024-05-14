MPICC?=mpicc
FLAGS?=-lcudart

all: mpi_cuda_hello_world mpi_hello_world hello_world_sycl_mpi sycl_mpi_map_device

mpi_cuda_hello_world:
	${MPICC} -o hello_world_cuda_mpi hello_world_cuda_mpi.c ${FLAGS}

mpi_hello_world:
	${MPICC} -o hello_world_mpi hello_world_mpi.c

hello_world_sycl_mpi:
	# 1. Add mpiicpc: `source /opt/intel/oneapi/setvars.sh`
	# 2. Set dpcpp compiler: export I_MPI_CXX=dpcpp
	# Run compilation with MPI + SYCL!
	mpiicpc -fsycl -std=c++17 -lsycl -ltbb hello_world_sycl_mpi.cpp -o hello_world_sycl_mpi

sycl_mpi_map_device:
	# 1. Add mpiicpc: `source /opt/intel/oneapi/setvars.sh`
	# 2. Set dpcpp compiler: export I_MPI_CXX=dpcpp
	# Run compilation with MPI + SYCL!
	mpiicpc -fsycl -std=c++17 -lsycl -ltbb sycl_mpi_map_device.cpp -o sycl_mpi_map_device

.PHONY: clean
clean:
	rm hello_world_mpi hello_world_cuda_mpi hello_world_sycl_mpi sycl_mpi_map_device
