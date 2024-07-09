#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "../report_lib/experiment.h"
#include "../report_lib/time_report.h"
#include "../utils/data_validator.hpp"

class CPUtoGPU_CUDA: public Experiment<double>{

TimeReport run(ExperimentArgs<double> experimentArgs) {
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    TimeReport timeReport = TimeReport();
    DataValidator<double> dataValidator = DataValidator<double>();
    
    // Memory Copy Size 
    unsigned long int size =  experimentArgs.getBufferSize();

    // Log memory usage / Amount to transfer
    cout << "Number of elems: "<< experimentArgs.numberOfElems << " - Memory Buffer Size (Mb): " << size / 1024 << std::endl; 

    // GPUs
    int gpuid_0 = 0;
 
    // Allocate Memory CPU

    double *buffer_host = static_cast<double*>(malloc(size));
    double *buffer_host_device = static_cast<double*>(malloc(size));
    dataValidator.init_buffer(buffer_host, experimentArgs.numberOfElems);

    // Allocate Memory GPU
    double* buffer_device_0;
    cudaSetDevice(gpuid_0);
    cudaError_t err = cudaMalloc((void**)&buffer_device_0, size);

    // Init Timing Data
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 
    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(buffer_device_0, buffer_host, size, cudaMemcpyHostToDevice, stream);   
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    timeReport.latency.time_ms = time_ms;
    
    printf("Seconds: %f\n", timeReport.latency.get_time_s());
    printf("Unidirectional Bandwidth: %f (GB/s)\n", timeReport.bandwidth_gb(size, timeReport.latency.time_ms));
    
    //Validate Data Integrity
    cudaMemcpy(buffer_host_device, buffer_device_0, size, cudaMemcpyDeviceToHost);   
    dataValidator.validate_data(buffer_host, buffer_host_device, experimentArgs.numberOfElems);
    
    
    // Clean Up
    cudaFree(buffer_device_0);
    free(buffer_host_device);
    free(buffer_host);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
	return timeReport;

}
};