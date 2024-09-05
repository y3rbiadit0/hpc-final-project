#ifndef HPC_TIME_REPORT_H
#define HPC_TIME_REPORT_H

#include "time_counter.hpp"

class TimeReport {
    

    public:
        TimeCounter latency;
        double bandwidth_gb(double data_size, double time);   

};

#endif //HPC_TIME_REPORT_H
