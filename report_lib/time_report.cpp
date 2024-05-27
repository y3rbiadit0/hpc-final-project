#include "time_report.h"

double TimeReport::bandwidth(double data_size, double time){
    double bandwidth = 0.0;
    if (time > 0.0) {
            bandwidth = data_size / time; // Bandwidth in bytes per second
        } else {
            bandwidth = 0.0;
        }
    return bandwidth;

}