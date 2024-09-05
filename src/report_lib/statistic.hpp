#ifndef HPC_TIME_STATISTIC_H
#define HPC_TIME_STATISTIC_H

#include <string>
#include <vector>

#include "time_report.hpp"

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
        void dumpToCSV(std::string file_path);

};

#endif