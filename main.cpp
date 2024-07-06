#include <iostream>
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"

#include "report_lib/statistic.h"
// #include "sycl_mpi/cpu_to_gpu.cpp"
// #include "cuda_mpi/cpu_to_gpu_cuda_mpi.cpp"
#include "cuda_mpi/gpu_to_gpu_cuda_nvlink.cpp"

using namespace std;


double getLatency(TimeReport timeReport){
    return timeReport.latency.get_time_s();
}

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 0; 
    unsigned int numberOfRuns = 5;
    unsigned int numberOfElems = 1048576;

    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems
    );
    
    GPUtoGPU_CUDA_NVLINK cuda_gpu_to_gpu_nvlink = GPUtoGPU_CUDA_NVLINK();
    
    ExperimentRunner gpu_to_gpu_cuda_nvlink= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_nvlink);


    MPI_Init(&argc, &argv);
    gpu_to_gpu_cuda_nvlink.runExperiment();

    MPI_Finalize();
}


