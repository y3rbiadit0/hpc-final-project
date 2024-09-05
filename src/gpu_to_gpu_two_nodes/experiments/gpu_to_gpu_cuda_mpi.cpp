#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#include "statistic.hpp"
#include "experiment.hpp"
#include "data_validator.hpp"
class GPUtoGPU_CUDA_MPI: public Experiment <double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
	TimeReport timeReport= TimeReport();
	// MPI - Node ID
	int node_id, mpi_size;
	
	// Buffer Related Variables
	double *sBuf_h, *rBuf_h;
	double *sBuf_d, *rBuf_d;
	
	// CUDA Related Variables
	int deviceCount;
	int device_id;
	
	DataValidator<double> dataValidator = DataValidator<double>();

	// Init MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// Init CUDA
	cudaGetDeviceCount(&deviceCount);
	device_id = node_id % deviceCount;
	
	// Map MPI-Process to a GPU
	cudaSetDevice(device_id);
	const unsigned long int bufferSize =  experimentArgs.getBufferSize();

	
	if (node_id == 0) {
		// Allocate Device Buffers
	
        sBuf_h = (double*) malloc(bufferSize);
		dataValidator.init_buffer_with_value(sBuf_h, experimentArgs.numberOfElems, 1.0);
	    cudaMalloc((void **) &sBuf_d, bufferSize);
		cudaMemcpy(sBuf_d, sBuf_h, bufferSize, cudaMemcpyHostToDevice);
	}
    else{
		// Allocate Device Buffers
        rBuf_h = (double*) malloc(bufferSize);
        cudaMalloc((void **) &rBuf_d, bufferSize);
    }

 
    MPI_Barrier(MPI_COMM_WORLD);
	timeReport.latency.start();
	if(node_id==0){
        //Send information to other devices
		MPI_Send(sBuf_d, experimentArgs.numberOfElems, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
	}
	else{		
        //Received information from master
    	MPI_Recv(rBuf_d, experimentArgs.numberOfElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
	}
	timeReport.latency.end();
	if(node_id == 1){
		double* validateBuffer = (double*) malloc(bufferSize);
		for(long i = 0; i < experimentArgs.numberOfElems; i++){
			validateBuffer[i] = 1.0;
		}
		cudaMemcpy(rBuf_h, rBuf_d, bufferSize, cudaMemcpyDeviceToHost);
		dataValidator.validate_data(validateBuffer, rBuf_h, experimentArgs.numberOfElems);
		free(validateBuffer);
	}
	

	if(node_id == 0){
		cudaFree(sBuf_d);
		free(sBuf_h);
	}
	if(node_id == 1){
		cudaFree(rBuf_d);
		free(rBuf_h);
	}
	
	return timeReport;

}
};