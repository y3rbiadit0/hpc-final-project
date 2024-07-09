#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"
#include "../utils/data_validator.hpp"

class GPUtoGPU_CUDA_PCIE: public Experiment<double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    TimeReport timeReport = TimeReport();
    DataValidator<double> dataValidator;

    // Memory Copy Size 
    float size =  experimentArgs.getBufferSize();

    // Log memory usage / Amount to transfer
    cout << "Number of elems: "<< experimentArgs.numberOfElems << " - Memory Buffer Size (Mb): " << size / 1024 << endl; 

       // GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    // Allocate Memory
    double* dev_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&dev_0, size);
    dataValidator.init_buffer(dev_0, experimentArgs.numberOfElems);
 
    double* dev_1;
    cudaSetDevice(gpuid_1);
    cudaMalloc((void**)&dev_1, size);
 
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);
 
    // Note: Not enabling Peer Access to enforce PCI-E Communication.
    
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
 
    dataValidator.validate_data(dev_0, dev_1, experimentArgs.numberOfElems);

    printf("Seconds: %f\n", timeReport.latency.get_time_s());
    printf("Unidirectional Bandwidth: %f (GB/s)\n", timeReport.bandwidth_gb(size, timeReport.latency.time_ms));
 
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