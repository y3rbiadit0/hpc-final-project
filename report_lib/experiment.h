#ifndef HPC_TIME_EXPERIMENT_H
#define HPC_TIME_EXPERIMENT_H

#include "experiment_args.h"
#include "time_report.h"

template <typename T>
class Experiment {

    public: 
        virtual TimeReport run(ExperimentArgs<T> args) = 0;

};

#endif