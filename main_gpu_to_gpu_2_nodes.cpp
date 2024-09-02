#include <iostream>

#include "report_lib/statistic.h"
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"
#include "gpu_to_gpu_two_nodes/gpu_to_gpu_cuda_mpi.cpp"


using namespace std;

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 20; 
    unsigned int numberOfRuns = 30;
    unsigned int memorySizeInMB = 16;

    //(size_in_mb * 1024 kilobytes * 1024 bytes) / sizeof(double) = total number of elements
    long unsigned int numberOfElems = memorySizeInMB * 1024 * 1024 / sizeof(double);
    
    constexpr bool isBidirectional = false;    
    
    
    ExperimentArgs<double> experimentArgs = ExperimentArgs<double>(argc, argv, numberOfWarmups, numberOfRuns, numberOfElems, isBidirectional);
    
    GPUtoGPU_CUDA_MPI cuda_mpi_gpu_to_gpu = GPUtoGPU_CUDA_MPI();
    
    ExperimentRunner<double> gpu_to_gpu_mpi_cuda_experiment= ExperimentRunner<double>(&experimentArgs, &cuda_mpi_gpu_to_gpu);

	MPI_Init(&argc, &argv);

    cout<< "GPU_TO_GPU_MPI -- CUDA RESULTS:"<< std::endl;
    cout << "Number of elems: "<< experimentArgs.numberOfElems << " - Memory Buffer Size (MB): " << experimentArgs.getBufferSize() / (1024*1024) << " - Memory Buffer Size (GB): " << experimentArgs.getBufferSize() / (1048576*1024)  << std::endl; 
    gpu_to_gpu_mpi_cuda_experiment.runExperiment();   
    MPI_Finalize();
    
    return 0;
}