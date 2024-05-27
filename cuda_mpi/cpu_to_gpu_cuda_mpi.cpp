#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"


TimeReport cpu_to_gpu_cuda_mpi (int argc, char* argv[], ExperimentArgs experimentArgs){
    TimeReport timeReport= TimeReport();

    //MPI - Node ID
    int node_id, mpi_size;

    //Buffer related variables
    int bufferSize = experimentArgs.bufferSize;
	float *Buf_h;
	float *Buf_d;
	int i;

    //CUDA Related Variables
    int deviceCount;
    int device_id;

    //Init MPI
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
	Buf_h = malloc(bufferSize);

    for (i = 0; i < bufferSize; ++i) {
		Buf_h[i] = (float) node_id;
	}

	// Allocate Device Buffers
	cudaMalloc((void **) &Buf_d, bufferSize);
	
    timeReport.latency.start();

	// Move Data from CPU -> Device
	cudaMemcpy(Buf_d, Buf_h, bufferSize, cudaMemcpyHostToDevice);
    
    timeReport.latency.end();

    return timeReport;

}