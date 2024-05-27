#include <mpi.h>
#include "time_counter.h"

void TimeCounter::start(){
    time = MPI_Wtime();
}

void TimeCounter::end(){
    time = MPI_Wtime() - time;    
}

double TimeCounter::get_time(){
    return time;
}

