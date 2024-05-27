#ifndef HPC_TIME_EXPERIMENT_H
#define HPC_TIME_EXPERIMENT_H
#include "time_report.h"
#include <string>

class ExperimentStatistic {
    unsigned int dataArraySize = 0;
    double* dataArray = nullptr;

    public:
        double max();
        double min();
        double median();
        double avg();
        double mean();
        double dumpToCSV(std::string file_path);

};


class ExperimentArgs{
    public:
        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        unsigned int bufferSize;
};

class ExperimentRunner {
    
    TimeReport* timeReports = nullptr;
    public:
        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        unsigned int numberOf


};



#endif