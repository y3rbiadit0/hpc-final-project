#include <iostream>
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"

#include "report_lib/statistic.h"
// #include "sycl_mpi/cpu_to_gpu.cpp"
// #include "cuda_mpi/cpu_to_gpu_cuda_mpi.cpp"
#include "cuda_mpi/gpu_to_gpu_cuda_nvlink.cpp"
#include "cuda_mpi/gpu_to_gpu_cuda_pcie.cpp"

using namespace std;


double getLatency(TimeReport timeReport){
    return timeReport.latency.get_time_s();
}

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 20; 
    unsigned int numberOfRuns = 50;
    unsigned int numberOfElems = 2097152;

    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems
    );
    
    GPUtoGPU_CUDA_NVLINK cuda_gpu_to_gpu_nvlink = GPUtoGPU_CUDA_NVLINK();
    GPUtoGPU_CUDA_PCIE cuda_gpu_to_gpu_pcie = GPUtoGPU_CUDA_PCIE();

    ExperimentRunner gpu_to_gpu_cuda_nvlink= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_nvlink);
    ExperimentRunner gpu_to_gpu_cuda_pcie= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_pcie);


    MPI_Init(&argc, &argv);
    gpu_to_gpu_cuda_nvlink.runExperiment();
    gpu_to_gpu_cuda_pcie.runExperiment();

    MPI_Finalize();
}


