#ifndef HPC_TIME_COUNTER_H
#define HPC_TIME_COUNTER_H

class TimeCounter {
    private:
        double time;
    public:
        // Call MPI_WTime
        void start();
        // Call MPI_WTime
        void end();
        double get_time();
};

#endif