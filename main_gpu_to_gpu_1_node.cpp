#include <iostream>

#include "report_lib/statistic.h"
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"
#include "gpu_to_gpu_single_node/gpu_to_gpu_cuda_nvlink.cpp"
#include "gpu_to_gpu_single_node/gpu_to_gpu_cuda_pcie.cpp"
#include "gpu_to_gpu_single_node/gpu_to_gpu_sycl_nvlink.cpp"
#include "gpu_to_gpu_single_node/cpu_to_gpu_cuda.cpp"

using namespace std;


double getLatency(TimeReport timeReport){
    return timeReport.latency.get_time_s();
}

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 0; 
    unsigned int numberOfRuns = 1;
    unsigned int numberOfElems = 2097152;

    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems
    );
    
    CPUtoGPU_CUDA cuda_cpu_to_gpu = CPUtoGPU_CUDA();
    GPUtoGPU_CUDA_NVLINK cuda_gpu_to_gpu_nvlink = GPUtoGPU_CUDA_NVLINK();
    GPUtoGPU_CUDA_PCIE cuda_gpu_to_gpu_pcie = GPUtoGPU_CUDA_PCIE();
    GPUtoGPU_CUDA_SYCL_NVLINK cuda_gpu_to_gpu_sycl_nvlink = GPUtoGPU_CUDA_SYCL_NVLINK();
    
    //CPU to GPU Experiments
    ExperimentRunner cpu_to_gpu_cuda= ExperimentRunner<double>(&experimentArgs, &cuda_cpu_to_gpu);

    //GPU to GPU Experiments
    ExperimentRunner gpu_to_gpu_cuda_nvlink= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_nvlink);
    ExperimentRunner gpu_to_gpu_cuda_pcie= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_pcie);
    ExperimentRunner gpu_to_gpu_cuda_sycl_nvlink= ExperimentRunner<double>(&experimentArgs, &cuda_gpu_to_gpu_sycl_nvlink);
    
    cout<< "CPU_TO_GPU -- CUDA RESULTS:"<< std::endl;
    cpu_to_gpu_cuda.runExperiment();   
    cout<< "GPU_TO_GPU -- CUDA RESULTS:"<< std::endl;
    gpu_to_gpu_cuda_nvlink.runExperiment();
    cout<< "GPU_TO_GPU -- CUDA RESULTS:"<< std::endl;
    gpu_to_gpu_cuda_pcie.runExperiment();
    //gpu_to_gpu_cuda_sycl_nvlink.runExperiment();

}


