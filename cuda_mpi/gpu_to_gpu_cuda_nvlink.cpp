#include <stdio.h>

#include <stdlib.h>

#include <mpi.h>

#include <cuda_runtime.h>

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"

class GPUtoGPU_CUDA_NVLINK: public Experiment{

TimeReport run(ExperimentArgs experimentArgs) {
	TimeReport timeReport= TimeReport();
	// GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    // Memory Copy Size
    float size =  experimentArgs.bufferSize * sizeof(float); // 2^26 = 67MB
 
    // Allocate Memory for GPU 0
    float* dev_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&dev_0, size);
 
	// Allocate Memory for GPU 1
    float* dev_1;
    cudaSetDevice(gpuid_1);
    cudaMalloc((void**)&dev_1, size);
 
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Enable P2P Access
        cudaSetDevice(gpuid_0);
        cudaDeviceEnablePeerAccess(gpuid_1, 0);
        cudaSetDevice(gpuid_1);
        cudaDeviceEnablePeerAccess(gpuid_0, 0);
    }
 

    // Init Timing Data
    uint32_t repeat = 10;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 

	//Time Report Start
	timeReport.latency.start();
    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
 
    //Do a P2P memcpy
    for (int i = 0; i < repeat; ++i) {
        cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);
    }
 
    cudaEventRecord(stop, stream);
	//Time Report End
	timeReport.latency.end();
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    double time_s = time_ms / 1e3;
 
    double gb = size * repeat / (double)1e9;
    double bandwidth = gb / time_s;
 
    printf("Seconds: %f\n", time_s);
    printf("Unidirectional Bandwidth: %f (GB/s)\n", bandwidth);
 
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
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
	return timeReport;

}
};