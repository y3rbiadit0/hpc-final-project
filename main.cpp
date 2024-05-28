#include <iostream>
#include "report_lib/experiment.h"
#include "report_lib/experiment_runner.h"

#include "report_lib/statistic.h"
#include "sycl_mpi/cpu_to_gpu.cpp"

using namespace std;


double getLatency(TimeReport timeReport){
    return timeReport.latency.get_time();
}

int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 5; 
    unsigned int numberOfRuns = 5;
    unsigned int bufferSize = 1048576;

    
    ExperimentArgs experimentArgs = ExperimentArgs(
        argc, argv, numberOfWarmups, numberOfRuns, bufferSize
    );
    
    CPUtoGPU_Sycl_MPI experiment = CPUtoGPU_Sycl_MPI();

    ExperimentRunner cpu_to_gpu_sycl_mpi = ExperimentRunner(&experimentArgs, &experiment);

    MPI_Init(&argc, &argv);

    cpu_to_gpu_sycl_mpi.runExperiment();

    MPI_Finalize();
}


