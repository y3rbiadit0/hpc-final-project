MPICC?=mpicc
FLAGS?=-lcudart

all: mpi_cuda_hello_world mpi_hello_world

mpi_cuda_hello_world:
	${MPICC} -o hello_world_cuda_mpi hello_world_cuda_mpi.c ${FLAGS}

mpi_hello_world:
	${MPICC} -o hello_world_mpi hello_world_mpi.c

.PHONY: clean
clean:
	rm hello_world_mpi hello_world_cuda_mpi
