#include <cuda_runtime.h>
#include <iostream>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"
#include "../utils/data_validator.hpp"
using namespace std;

class GPUtoGPU_CUDA_NVLINK_BIDI: public Experiment<double>{

void checkError(cudaError_t err){
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed! Error: " << cudaGetErrorString(err) << std::endl;
    }
}

TimeReport run(ExperimentArgs<double> experimentArgs) {
    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaEvent_t start;
    cudaEvent_t stop;
    TimeReport timeReport = TimeReport();
    DataValidator<double> dataValidator;

    // Memory Copy Size 
    unsigned long int size =  experimentArgs.getBufferSize();

    // GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    // Allocate GPU Memory
    double* dev_0_upstream;
    double* dev_0_downstream;
    cudaSetDevice(gpuid_0);
    checkError(cudaMalloc((void**)&dev_0_upstream, size));    
    checkError(cudaMalloc((void**)&dev_0_downstream, size));


    double* dev_1_upstream;
    double* dev_1_downstream;
    cudaSetDevice(gpuid_1);
    checkError(cudaMalloc((void**)&dev_1_upstream, size));    
    checkError(cudaMalloc((void**)&dev_1_downstream, size));

    // Allocate Host Memory
    double *buffer_dev_0_host_upstream = static_cast<double*>(malloc(size));
    double *buffer_dev_1_host_upstream = static_cast<double*>(malloc(size));

    double *buffer_dev_2_host_downstream = static_cast<double*>(malloc(size));
    double *buffer_dev_3_host_downstream = static_cast<double*>(malloc(size));

    // Init Buffer
    dataValidator.init_buffer(buffer_dev_0_host_upstream, experimentArgs.numberOfElems);
    cudaMemcpy(dev_0_upstream, buffer_dev_0_host_upstream, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_1_downstream, buffer_dev_0_host_upstream, size, cudaMemcpyHostToDevice);

 
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
 
    // ~~ Start Test ~~
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaEventRecord(start);
 
    //Do a P2P memcp
    cudaSetDevice(gpuid_0);
    cudaMemcpyAsync(dev_0_upstream, dev_1_upstream, size, cudaMemcpyDeviceToDevice, stream0);
    cudaSetDevice(gpuid_1);
    cudaMemcpyAsync(dev_1_downstream, dev_0_downstream, size, cudaMemcpyDeviceToDevice, stream1);
    
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    
    checkError(cudaEventRecord(stop));
    checkError(cudaEventSynchronize(stop));
    float time_ms = 0.0;
    cudaEventElapsedTime(&time_ms, start, stop);
    timeReport.latency.time_ms = time_ms;
    
    //Validate Data Integrity
    cudaMemcpy(buffer_dev_0_host_upstream, dev_0_upstream, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_dev_1_host_upstream, dev_1_upstream, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_dev_2_host_downstream, dev_0_downstream, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_dev_3_host_downstream, dev_1_downstream, size, cudaMemcpyDeviceToHost);
    dataValidator.validate_data(buffer_dev_0_host_upstream, buffer_dev_1_host_upstream, experimentArgs.numberOfElems);
    dataValidator.validate_data(buffer_dev_2_host_downstream, buffer_dev_3_host_downstream, experimentArgs.numberOfElems);

    
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Shutdown P2P Settings
        cudaSetDevice(gpuid_0);
        cudaDeviceDisablePeerAccess(gpuid_1);
        cudaSetDevice(gpuid_1);
        cudaDeviceDisablePeerAccess(gpuid_0);
    }
 
    // Clean Up
    cudaFree(dev_0_upstream);
    cudaFree(dev_1_upstream);
    cudaFree(dev_0_downstream);
    cudaFree(dev_1_downstream);
    free(buffer_dev_0_host_upstream);
    free(buffer_dev_1_host_upstream);
    free(buffer_dev_2_host_downstream);
    free(buffer_dev_3_host_downstream);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
	return timeReport;

}
};