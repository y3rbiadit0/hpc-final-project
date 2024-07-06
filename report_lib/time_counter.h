#ifndef HPC_TIME_COUNTER_H
#define HPC_TIME_COUNTER_H

class TimeCounter {
    
    public:
        double time_ms;
        // Call MPI_WTime
        void start();
        // Call MPI_WTime
        void end();
        double get_time_s();
};

#endif