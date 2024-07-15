#include <iostream>

#include "report_lib/statistic.h"
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"
#include "gpu_to_gpu_two_nodes/gpu_to_gpu_cuda_mpi.cpp"


using namespace std;

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 0; 
    unsigned int numberOfRuns = 1;
    unsigned int numberOfElems = 20;

    
    ExperimentArgs experimentArgs = ExperimentArgs<double>(
        argc, argv, numberOfWarmups, numberOfRuns, numberOfElems
    );
    
    GPUtoGPU_CUDA_MPI cuda_mpi_gpu_to_gpu = GPUtoGPU_CUDA_MPI();
    
    ExperimentRunner gpu_to_gpu_mpi_cuda_experiment= ExperimentRunner<double>(&experimentArgs, &cuda_mpi_gpu_to_gpu);

	MPI_Init(&argc, &argv);

    cout<< "GPU_TO_GPU_MPI -- CUDA RESULTS:"<< std::endl;
    gpu_to_gpu_mpi_cuda_experiment.runExperiment();   
    MPI_Finalize();
    
    return 0;
}