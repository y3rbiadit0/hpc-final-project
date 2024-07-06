#include <mpi.h>
#include "time_counter.h"

void TimeCounter::start(){
    time_ms = MPI_Wtime();
}

void TimeCounter::end(){
    time_ms = MPI_Wtime() - time_ms;    
}

double TimeCounter::get_time_s(){
    return time_ms / 1e3;
}

