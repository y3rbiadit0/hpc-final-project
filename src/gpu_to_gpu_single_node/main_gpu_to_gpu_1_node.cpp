#include <iostream>

#include "statistic.hpp"
#include "experiment.hpp"
#include "experiment_runner.cpp"
#include "gpu_to_gpu_cuda_nvlink.cpp"
#include "gpu_to_gpu_cuda_pcie.cpp"
#include "gpu_to_gpu_sycl_nvlink.cpp"
#include "gpu_to_gpu_sycl_pcie.cpp"
#include "cpu_to_gpu_cuda.cpp"
#include "gpu_to_gpu_cuda_nvlink_bidi.cpp"
#include "gpu_to_gpu_sycl_nvlink_bidi.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 0; 
    unsigned int numberOfRuns = 1;
    unsigned int memorySizeInMB = 256;

    //(size_in_mb * 1024 kilobytes * 1024 bytes) / sizeof(double) = total number of elements
    long unsigned int numberOfElems = memorySizeInMB * 1024 * 1024 / sizeof(double);
    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems, false
    );

    ExperimentArgs experimentArgsBidirectional = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems, true
    );
    
    cout << "Number of elems: "<< experimentArgs.numberOfElems << " - Memory Buffer Size (MB): " << experimentArgs.getBufferSize() / (1024*1024) << " - Memory Buffer Size (GB): " << experimentArgs.getBufferSize() / (1048576*1024)  << std::endl; 
    

    CPUtoGPU_CUDA cpu_to_gpu_cuda_exp = CPUtoGPU_CUDA();
    GPUtoGPU_CUDA_NVLINK gpu_to_gpu_cuda_nvlink_exp = GPUtoGPU_CUDA_NVLINK();
    GPUtoGPU_CUDA_NVLINK_BIDI gpu_to_gpu_cuda_nvlink_bidi_exp = GPUtoGPU_CUDA_NVLINK_BIDI();
    GPUtoGPU_CUDA_PCIE gpu_to_gpu_cuda_pcie_exp = GPUtoGPU_CUDA_PCIE();
    GPUtoGPU_SYCL_NVLINK_BIDI gpu_to_gpu_sycl_nvlink_bidi_exp = GPUtoGPU_SYCL_NVLINK_BIDI();
    GPUtoGPU_SYCL_NVLINK gpu_to_gpu_sycl_nvlink_exp = GPUtoGPU_SYCL_NVLINK();
    GPUtoGPU_SYCL_PCIE gpu_to_gpu_sycl_pcie_exp = GPUtoGPU_SYCL_PCIE();

    //CPU to GPU Experiments
    ExperimentRunner cpu_to_gpu_cuda= ExperimentRunner<double>(&experimentArgs, &cpu_to_gpu_cuda_exp);

    //GPU to GPU Experiments
    ExperimentRunner gpu_to_gpu_cuda_nvlink= ExperimentRunner<double>(&experimentArgs, &gpu_to_gpu_cuda_nvlink_exp);
    ExperimentRunner gpu_to_gpu_cuda_nvlink_bidi= ExperimentRunner<double>(&experimentArgsBidirectional, &gpu_to_gpu_cuda_nvlink_bidi_exp);
    ExperimentRunner gpu_to_gpu_cuda_pcie= ExperimentRunner<double>(&experimentArgs, &gpu_to_gpu_cuda_pcie_exp);
    ExperimentRunner gpu_to_gpu_sycl_nvlink_bidi = ExperimentRunner<double>(&experimentArgsBidirectional, &gpu_to_gpu_sycl_nvlink_bidi_exp);
    ExperimentRunner gpu_to_gpu_sycl_nvlink= ExperimentRunner<double>(&experimentArgs, &gpu_to_gpu_sycl_nvlink_exp);
    ExperimentRunner gpu_to_gpu_sycl_pcie= ExperimentRunner<double>(&experimentArgs, &gpu_to_gpu_sycl_pcie_exp);
    
    cout<< "CPU_TO_GPU -- CUDA RESULTS:"<< std::endl;
    cpu_to_gpu_cuda.runExperiment();
    cout<< "GPU_TO_GPU -- CUDA PCIE RESULTS:"<< std::endl;
    gpu_to_gpu_cuda_pcie.runExperiment();   
    cout<< "GPU_TO_GPU -- CUDA NVLINK BIDIRECTIONAL RESULTS:"<< std::endl; 
    gpu_to_gpu_cuda_nvlink_bidi.runExperiment();
    cout<< "GPU_TO_GPU -- CUDA NVLINK RESULTS:"<< std::endl;
    gpu_to_gpu_cuda_nvlink.runExperiment();
    cout<< "GPU_TO_GPU -- SYCL NVLINK RESULTS:"<< std::endl;
    gpu_to_gpu_sycl_nvlink.runExperiment();
    cout<< "GPU_TO_GPU -- SYCL NVLINK BIDIRECTIONAL RESULTS:"<< std::endl;
    gpu_to_gpu_sycl_nvlink_bidi.runExperiment();
    cout<< "GPU_TO_GPU -- SYCL PCIE RESULTS:"<< std::endl;
    gpu_to_gpu_sycl_pcie.runExperiment();

}
