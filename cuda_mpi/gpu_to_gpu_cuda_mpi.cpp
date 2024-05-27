#include <stdio.h>

#include <stdlib.h>

#include <mpi.h>

#include <cuda_runtime.h>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"

TimeReport gpu_to_gpu_cuda_mpi(int argc, char *argv[], ExperimentArgs experimentArgs) {
	// MPI - Node ID
	int node_id, mpi_size;
	
	// Buffer Related Variables
    int bufferSize = experimentArgs.bufferSize;
	float *sBuf_h, *rBuf_h;
	float *sBuf_d, *rBuf_d;
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
	//sBuf_h = malloc(bufferSize);
	//rBuf_h = malloc(bufferSize);
	
	if (node_id == 0) {
        sBuf_h = malloc(bufferSize);
		printf("sBuf_h[0] = %f\n", sBuf_h[0]);
	}
    else{
        rBuf_h = malloc(bufferSize);
    }

    //TODO
    for (i = 0; i < bufferSize; ++i) {
		sBuf_h[i] = (float) node_id;
		rBuf_h[i] = -1.;
	}

	// Allocate Device Buffers
	//cudaMalloc((void **) &sBuf_d, bufferSize);
	//cudaMalloc((void **) &rBuf_d, bufferSize);
	
	// Move Data from CPU -> Device
	//cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
	
	// Send Data GPU to GPU
	if (node_id == 0) {
       // Allocate Device Buffers
	    cudaMalloc((void **) &sBuf_d, bufferSize);
        // Move Data from CPU -> Device
	    cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
        //Send information to other devices
        MPI_Send(sBuf_d, LBUF, MPI_REAL, 0, 0, MPI_COMM_WORLD);
    } else if (node_id == 1) {       
        // Allocate Device Buffers
        cudaMalloc((void **) &rBuf_d, bufferSize);
        //Received information from master
    	MPI_Recv(rBuf_d, LBUF, MPI_REAL, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Move Data from Device -> CPU
        cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);
    } else {

        printf("Unexpected node value %d\n", node_id);
        exit(-1);

    }

	//cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);

	if (node_id == 0) {
        printf("rBuf_h[0] = %f\n", rBuf_h[0]);
	}

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
