#ifndef HPC_TIME_EXPERIMENT_ARGS_H
#define HPC_TIME_EXPERIMENT_ARGS_H

template <typename T>
class ExperimentArgs{
    public:
    // Parameterized Constructor
    ExperimentArgs(int argc, char *argv[], unsigned int numberOfWarmupArg, unsigned int numberOfRunsArg, unsigned int numberOfElemsArg)
    {
        argc = argc;
        argv = argv;
        numberOfRuns = numberOfRunsArg;
        numberOfWarmup = numberOfWarmupArg;
        numberOfElems = numberOfElemsArg;
    }
        int argc;
        char *argv;
        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        unsigned int numberOfElems;
    
    unsigned long int getBufferSize() const { return numberOfElems * sizeof(T); }
};

#endif