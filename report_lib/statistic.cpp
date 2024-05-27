#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "statistic.h"

double ExperimentStatistic::max(){
    if (data.empty()) {
        throw std::runtime_error("Cannot compute max of an empty vector");
    }
    return *std::max_element(data.begin(), data.end());
}
double ExperimentStatistic::min(){
if (data.empty()) {
        throw std::runtime_error("Cannot compute max of an empty vector");
    }
    return *std::min_element(data.begin(), data.end());

}
double ExperimentStatistic::median(){
    if (data.empty()) {
        throw std::runtime_error("Cannot compute median of an empty vector");
    }
    size_t size = data.size();
    std::sort(data.begin(), data.end());
    if (size % 2 == 0) {
        //Return the average value between the two values at the middle
        return (data[size / 2 - 1] + data[size / 2]) / 2;
    } else {
        return data[size / 2];
    }
}
double ExperimentStatistic::avg(){
    if (data.empty()) {
        throw std::runtime_error("Cannot compute average of an empty vector");
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}
void ExperimentStatistic::dumpToCSV(std::string file_path){
    //TODO: To implement
}



