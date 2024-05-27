#ifndef HPC_TIME_COUNTER_H
#define HPC_TIME_COUNTER_H

class TimeCounter {
    private:
        double time;
    public:
        // Call MPI_WTime
        double start();
        // Call MPI_WTime
        double end();

        // return `time`
        double get_time();
};

#endif