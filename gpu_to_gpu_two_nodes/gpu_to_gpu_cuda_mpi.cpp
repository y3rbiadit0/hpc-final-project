#include <stdio.h>

#include <stdlib.h>

#include <mpi.h>

#include <cuda_runtime.h>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"

class GPUtoGPU_CUDA_MPI: public Experiment{

TimeReport run(ExperimentArgs experimentArgs) {
	TimeReport timeReport= TimeReport();
	// MPI - Node ID
	int node_id, mpi_size;
	
	// Buffer Related Variables
	float *sBuf_h, *rBuf_h;
	float *sBuf_d, *rBuf_d;
	int i;
	
	// CUDA Related Variables
	int deviceCount;
	int device_id;


	// Init MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Init CUDA
	cudaGetDeviceCount(&deviceCount);
	device_id = node_id % deviceCount;
	
	// Map MPI-Process to a GPU
	cudaSetDevice(device_id);
	printf("Number of GPUs found = %d\n", deviceCount);
    printf("Assigned GPU %d to MPI rank %d of %d.\n", device_id, node_id, mpi_size);


	const int nelem = experimentArgs.bufferSize;        // Buffer Elements number
    const size_t nsize = nelem * sizeof(float);
	int bufferSize = nsize;

	if (node_id == 1) {
		// Allocate Device Buffers
        sBuf_h = (float*) malloc(bufferSize);
		printf("sBuf_h[0] = %f\n", sBuf_h[0]);
	    cudaMalloc((void **) &sBuf_d, bufferSize);

		for (i = 0; i < bufferSize; ++i) {
			sBuf_h[i] = (float) node_id;
		}

		// Move Data from CPU -> Device
	    cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
	}
    else{
		// Allocate Device Buffers
        rBuf_h = (float*) malloc(bufferSize);
		printf("rBuf_h[0] = %f\n", rBuf_h[0]);
        cudaMalloc((void **) &rBuf_d, bufferSize);
    }

 
    MPI_Barrier(MPI_COMM_WORLD);
	if(node_id==1){
		//Measure latency
		timeReport.latency.start();
        //Send information to other devices
        MPI_Send(sBuf_d, bufferSize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

		timeReport.latency.end();

	}
	else{		
		//Measure latency
		timeReport.latency.start();
        //Received information from master
    	MPI_Recv(rBuf_d, bufferSize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		timeReport.latency.start();
        // Move Data from Device -> CPU
        cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);
	}

	return timeReport;

}
};