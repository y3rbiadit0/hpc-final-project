#ifndef HPC_TIME_STATISTIC_H
#define HPC_TIME_STATISTIC_H

#include <string>
#include <vector>

#include "time_report.h"

class ExperimentStatistic {
    std::vector<double> data;    

    public:
        ExperimentStatistic(std::vector<double> values){
            data = values;
        }
        double max();
        double min();
        double median();
        double avg();
        double dumpToCSV(std::string file_path);

};

#endif