#include <cuda_runtime.h>
#include <iostream>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"
#include "../utils/data_validator.hpp"
using namespace std;

class GPUtoGPU_CUDA_NVLINK: public Experiment<double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    TimeReport timeReport = TimeReport();
    DataValidator<double> dataValidator;

    // Memory Copy Size 
    float size =  experimentArgs.getBufferSize();

    // GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    // Allocate GPU Memory
    double* dev_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&dev_0, size);
    double* dev_1;
    cudaSetDevice(gpuid_1);
    cudaMalloc((void**)&dev_1, size);

    // Allocate Host Memory
    double *buffer_dev_0_host = static_cast<double*>(malloc(size));
    double *buffer_dev_1_host = static_cast<double*>(malloc(size));
    
    // Init Buffer
    dataValidator.init_buffer(buffer_dev_0_host, experimentArgs.numberOfElems);
    cudaMemcpy(dev_0, buffer_dev_0_host, size, cudaMemcpyHostToDevice);

 
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Enable P2P Access
        cudaSetDevice(gpuid_0);
        cudaDeviceEnablePeerAccess(gpuid_1, 0);
        cudaSetDevice(gpuid_1);
        cudaDeviceEnablePeerAccess(gpuid_0, 0);
    }
 
    // Init Timing Data
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 
    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
 
    //Do a P2P memcp
    cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);
    
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    timeReport.latency.time_ms = time_ms;
    
    //Validate Data Integrity
    cudaMemcpy(buffer_dev_0_host, dev_0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_dev_1_host, dev_1, size, cudaMemcpyDeviceToHost);
    dataValidator.validate_data(buffer_dev_0_host, buffer_dev_1_host, experimentArgs.numberOfElems);
    
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Shutdown P2P Settings
        cudaSetDevice(gpuid_0);
        cudaDeviceDisablePeerAccess(gpuid_1);
        cudaSetDevice(gpuid_1);
        cudaDeviceDisablePeerAccess(gpuid_0);
    }
 
    // Clean Up
    cudaFree(dev_0);
    cudaFree(dev_1);
    free(buffer_dev_0_host);
    free(buffer_dev_1_host);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
	return timeReport;

}
};