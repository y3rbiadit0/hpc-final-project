#ifndef HPC_TIME_EXPERIMENT_ARGS_H
#define HPC_TIME_EXPERIMENT_ARGS_H

class ExperimentArgs{
    public:
    // Parameterized Constructor
    ExperimentArgs(int argc, char *argv[], unsigned int numberOfWarmupArg, unsigned int numberOfRunsArg, unsigned int bufferSizeArg)
    {
        argc = argc;
        argv = argv;
        numberOfRuns = numberOfRunsArg;
        numberOfWarmup = numberOfWarmupArg;
        bufferSize = bufferSizeArg;
    }
        int argc;
        char *argv;
        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        unsigned int bufferSize;
};

#endif