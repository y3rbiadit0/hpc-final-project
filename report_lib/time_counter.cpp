#include "time_counter.h"
#include <mpi.h>

double TimeCounter::start(){
    time = MPI_Wtime();
}

double TimeCounter::end(){
    time = MPI_Wtime() - time;    
}

double TimeCounter::get_time(){
    return time;
}

