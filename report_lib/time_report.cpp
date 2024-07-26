#include "time_report.h"

double TimeReport::bandwidth_gb(double data_size_bytes, double time_ms){
    double bandwidth = 0.0;
    double time_s = time_ms / 1e3;
    double data_size_gb = 0.0;
    
    if (time_s > 0.0) {
        data_size_gb = data_size_bytes / (double) 1073741824;
        bandwidth = data_size_gb / time_s;
    }
    return bandwidth;

}