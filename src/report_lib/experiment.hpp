#ifndef HPC_TIME_EXPERIMENT_H
#define HPC_TIME_EXPERIMENT_H

#include "experiment_args.hpp"
#include "time_report.hpp"

template <typename T>
class Experiment {

    public: 
        virtual TimeReport run(ExperimentArgs<T> args) = 0;

};

#endif