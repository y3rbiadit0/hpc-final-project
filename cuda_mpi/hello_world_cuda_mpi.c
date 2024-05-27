#include <stdio.h>

#include <stdlib.h>

#include <mpi.h>

#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
	// MPI - Node ID
	int node_id, mpi_size;
	
	// Buffer Related Variables
    int LBUF = 1000000;
	float *sBuf_h, *rBuf_h;
	float *sBuf_d, *rBuf_d;
	int bufferSize;
	int i;
	
	// Cuda Related Variables
	int deviceCount;
	int device_id;


	// Init MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Init Cuda
	cudaGetDeviceCount(&deviceCount);
	device_id = node_id % deviceCount;
	
	// Map MPI-Process to a GPU
	cudaSetDevice(device_id);
	printf("Number of GPUs found = %d\n", deviceCount);
    printf("Assigned GPU %d to MPI rank %d of %d.\n", device_id, node_id, mpi_size);

	// Allocate buffers host/device
	bufferSize = sizeof(float) * LBUF;
	sBuf_h = malloc(bufferSize);
	rBuf_h = malloc(bufferSize);

	for (i = 0; i < LBUF; ++i) {
		sBuf_h[i] = (float) node_id;
		rBuf_h[i] = -1.;
	}
	
	if (node_id == 0) {
		printf("rBuf_h[0] = %f\n", rBuf_h[0]);
	}

	// Allocate Device Buffers
	cudaMalloc((void **) &sBuf_d, bufferSize);
	cudaMalloc((void **) &rBuf_d, bufferSize);
	
	// Move Data from CPU -> Device
	cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
	
	// Send Data GPU to GPU
	if (node_id == 0) {
    	MPI_Recv(rBuf_d, LBUF, MPI_REAL, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (node_id == 1) {
        MPI_Send(sBuf_d, LBUF, MPI_REAL, 0, 0, MPI_COMM_WORLD);
    } else {

        printf("Unexpected node value %d\n", node_id);
        exit(-1);

    }

	cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);

	if (node_id == 0) {
        printf("rBuf_h[0] = %f\n", rBuf_h[0]);
	}

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
