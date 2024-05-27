#include <iostream>
#include "report_lib/experiment.h"
#include "sycl_mpi/gpu_to_gpu.cpp"


int main(int argc, char* argv[]) {
    unsigned int numberOfWarmups = 3; 
    unsigned int numberOfRuns = 5;
    unsigned int bufferSize = 100;

    ExperimentArgs experimentArgs = ExperimentArgs(
        numberOfWarmups, numberOfRuns, bufferSize
    );
    ExperimentRunner experimentRunner = ExperimentRunner(&experimentArgs);


    //1. Do the warmpup Runs
    for (int i = 0; i < experimentArgs.numberOfWarmup; i++){
        //TODO: runExperiment
    }

    //2. Report Timings on experiment
    for (int i = 0; i < experimentArgs.numberOfWarmup; i++){
        //TODO: runExperiment
        //Dummy Line just to compile
        TimeReport timeReport = TimeReport();
        experimentRunner.timeReports.push_back(timeReport);
    }

    //3. Calculate timings on reports
}
