#include <iostream>

#include "report_lib/statistic.h"
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"
#include "gpu_to_gpu_single_node/gpu_to_gpu_cuda_nvlink.cpp"
#include "gpu_to_gpu_single_node/gpu_to_gpu_cuda_pcie.cpp"
#include "gpu_to_gpu_single_node/gpu_to_gpu_sycl_nvlink.cpp"
#include "gpu_to_gpu_single_node/gpu_to_gpu_sycl_pcie.cpp"
#include "gpu_to_gpu_single_node/cpu_to_gpu_cuda.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 0; 
    unsigned int numberOfRuns = 1;
    unsigned int numberOfElems = 2097152;

    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems
    );
    
   
}