#ifndef HPC_TIME_COUNTER_H
#define HPC_TIME_COUNTER_H
#include <chrono>
class TimeCounter {
    
    public:
        std::chrono::time_point<std::chrono::high_resolution_clock> chrono_time;
        double time_ms;
        // Call MPI_WTime
        void start();
        // Call MPI_WTime
        void end();
        double get_time_s();
};

#endif