#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"

class CPUtoGPU_CUDA: public Experiment<double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    TimeReport timeReport = TimeReport();
    
    // Memory Copy Size 
    float size =  experimentArgs.getBufferSize();

    // Log memory usage / Amount to transfer
    cout << "Number of elems: "<< experimentArgs.numberOfElems << " - Memory Buffer Size (Mb): " << size / 1024 << std::endl; 

    // GPUs
    int gpuid_0 = 0;
 
    // Allocate Memory CPU

    double *buffer_host = (double*) malloc(size);
    
    // Allocate Memory GPU
    double* buffer_device_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&buffer_device_0, size);

    // Init Timing Data
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 
    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(buffer_host, buffer_device_0, size, cudaMemcpyHostToDevice, stream);   
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    timeReport.latency.time_ms = time_ms;
 
    printf("Seconds: %f\n", timeReport.latency.get_time_s());
    printf("Unidirectional Bandwidth: %f (GB/s)\n", timeReport.bandwidth_gb(size, timeReport.latency.time_ms));
  
    // Clean Up
    cudaFree(buffer_device_0);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
	return timeReport;

}
};