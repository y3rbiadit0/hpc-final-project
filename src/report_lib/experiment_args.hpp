#ifndef HPC_TIME_EXPERIMENT_ARGS_H
#define HPC_TIME_EXPERIMENT_ARGS_H

template <typename T>
class ExperimentArgs{
    public:
        int argc;
        char **argv;
        unsigned int numberOfWarmup;
        unsigned int numberOfRuns;
        long unsigned int numberOfElems;
        bool isBidirectional;
    // Parameterized Constructor
    ExperimentArgs(int argc, char **argv, unsigned int numberOfWarmupArg, unsigned int numberOfRunsArg, long unsigned int numberOfElemsArg, bool isBidirectional)
    {
        this->argc = argc;
        this->argv = argv;
        this->numberOfRuns = numberOfRunsArg;
        this->numberOfWarmup = numberOfWarmupArg;
        this->numberOfElems = numberOfElemsArg;
        this->isBidirectional = isBidirectional;
    }
        
    
    unsigned long int getBufferSize() const { return numberOfElems * sizeof(T);}
};

#endif