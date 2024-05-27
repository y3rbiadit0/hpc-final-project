#ifndef HPC_TIME_EXPERIMENT_H
#define HPC_TIME_EXPERIMENT_H
#include "time_report.h"
#include <string>
#include <vector>
class ExperimentStatistic {
    std::vector<double> data;    

    public:
        double max();
        double min();
        double median();
        double avg();
        double dumpToCSV(std::string file_path);

};


class ExperimentArgs{
    public:
    // Parameterized Constructor
    ExperimentArgs(unsigned int numberOfWarmupArg, unsigned int numberOfRunsArg, unsigned int bufferSizeArg)
    {
        numberOfRuns = numberOfRunsArg;
        numberOfWarmup = numberOfWarmupArg;
        bufferSize = bufferSizeArg;
    }

        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        unsigned int bufferSize;
};

class ExperimentRunner {
    public:
        ExperimentRunner(ExperimentArgs* args){
            experimentArgs = args;
            timeReports = std::vector<TimeReport>(args->numberOfRuns);
        }

        ExperimentArgs *experimentArgs;
        std::vector<TimeReport> timeReports;    
};



#endif