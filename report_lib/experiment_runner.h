#ifndef HPC_TIME_EXPERIMENT_RUNNER_H
#define HPC_TIME_EXPERIMENT_RUNNER_H
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "time_report.h"
#include "experiment.h"
#include "experiment_args.h"
#include "statistic.h"

using namespace std;



class ExperimentRunner {
    
    public:
        ExperimentRunner(ExperimentArgs* args, Experiment* experimentToRun){
            experimentArgs = args;
            experiment = experimentToRun;
            timeReports = std::vector<TimeReport>();
        }

        ExperimentArgs *experimentArgs;
        std::vector<TimeReport> timeReports;
        Experiment* experiment;

        void runWarmups(){
            for (int i = 0; i < experimentArgs->numberOfWarmup; i++){
                experiment->run(*experimentArgs);
            }
        }


        void runExperiment(){
            //1. Do the Warmup Runs
            runWarmups();

            //2. Report Timings on Experiment
            for (int i = 0; i < experimentArgs->numberOfRuns; i++){
                TimeReport timeReport = experiment->run(*experimentArgs);
                timeReports.push_back(timeReport);
            }

            dumpStatistics();               
        }

        void dumpStatistics(){
            // Data Mappers
            auto getLatency = [](TimeReport& timeReport) -> double {
                return timeReport.latency.get_time();
            };

            auto getBandwidth = [&](TimeReport& timeReport) -> double {
                return timeReport.bandwidth(experimentArgs->bufferSize, timeReport.latency.get_time());
            };

            // Bandwidth/Timing Data            
            std::vector<double> latencyData(timeReports.size());
            std::vector<double> bandwidthData(timeReports.size());
            
            //Map TimeReport to latency (double)
            std::transform(timeReports.begin(), timeReports.end(), latencyData.begin(), getLatency);
            std::transform(timeReports.begin(), timeReports.end(), bandwidthData.begin(), getBandwidth);

            ExperimentStatistic latencyStatisticData = ExperimentStatistic(latencyData);
            ExperimentStatistic bandwidthStatisticData = ExperimentStatistic(bandwidthData); 

            // Latency/Bandwidth Statistic Data
            std::cout << "Latency Average Time: " << latencyStatisticData.avg() << "s" << std::endl;
            std::cout << "Latency Median Time: " << latencyStatisticData.median() << "s" << std::endl;
            std::cout << "Latency Max Time: " << latencyStatisticData.max() << "s" << std::endl;
            std::cout << "Latency Min Time: " << latencyStatisticData.min() << "s" << std::endl;


            std::cout << "Bandwidth Average: " << bandwidthStatisticData.avg() << "bytes/s" << std::endl;
            std::cout << "Bandwidth Median: " << bandwidthStatisticData.median() << "bytes/s" << std::endl;
            std::cout << "Bandwidth Max: " << bandwidthStatisticData.max() << "bytes/s" << std::endl;
            std::cout << "Bandwidth Min: " << bandwidthStatisticData.min() <<  "bytes/s" << std::endl;
        }

};





#endif