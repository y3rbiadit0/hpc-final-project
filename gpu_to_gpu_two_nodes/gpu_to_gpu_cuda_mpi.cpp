#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../report_lib/statistic.h"
#include "../report_lib/experiment.h"
#include "../report_lib/experiment_runner.h"

class GPUtoGPU_CUDA_MPI: public Experiment <double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
	TimeReport timeReport= TimeReport();
	// MPI - Node ID
	int node_id, mpi_size;
	
	// Buffer Related Variables
	double *sBuf_h, *rBuf_h;
	double *sBuf_d, *rBuf_d;
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


	const int nelem = experimentArgs.numberOfElems;        // Buffer Elements number
    const unsigned long int nsize = experimentArgs.getBufferSize();
	int bufferSize = nsize;

	if (node_id == 1) {
		// Allocate Device Buffers
        sBuf_h = (double*) malloc(bufferSize);
		printf("sBuf_h[0] = %f\n", sBuf_h[0]);
	    cudaMalloc((void **) &sBuf_d, bufferSize);

		for (i = 0; i < bufferSize; ++i) {
			sBuf_h[i] = (double) node_id;
		}

		// Move Data from CPU -> Device
	    cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
	}
    else{
		// Allocate Device Buffers
        rBuf_h = (double*) malloc(bufferSize);
		printf("rBuf_h[0] = %f\n", rBuf_h[0]);
        cudaMalloc((void **) &rBuf_d, bufferSize);
    }

 
    MPI_Barrier(MPI_COMM_WORLD);
	if(node_id==0){
		//Measure latency
		timeReport.latency.start();
        //Send information to other devices
        MPI_Send(sBuf_d, experimentArgs.numberOfElems, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

		timeReport.latency.end();

	}
	else{		
		//Measure latency
		timeReport.latency.start();
        //Received information from master
    	MPI_Recv(rBuf_d, experimentArgs.numberOfElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		timeReport.latency.end();
        // Move Data from Device -> CPU
        cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);
	}

	return timeReport;

}
};