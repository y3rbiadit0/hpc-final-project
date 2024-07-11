#include <assert.h>

#include <sycl/sycl.hpp>
#include <cassert>
#include <numeric>
#include <vector>
#include <chrono>
#include <cstdlib> // For setenv

#include "../report_lib/experiment.h"
#include "../report_lib/experiment_args.h"

using namespace sycl;

class GPUtoGPU_SYCL_PCIE : public Experiment<double>{
  TimeReport run(ExperimentArgs<double> experimentArgs)
  {
    
    TimeReport timeReport = TimeReport();
    std::vector<sycl::device> devices;
    DataValidator<double> dataValidator;
    
    
    for (const auto &plt : sycl::platform::get_platforms()) {
      if (plt.get_backend() == sycl::backend::ext_oneapi_cuda){
        devices.push_back(plt.get_devices()[0]);
      }
    }

    if (devices.size() < 2) {
      std::cout << "Cannot test P2P capabilities, at least two devices are "
                   "required, exiting."
                << std::endl;
      return timeReport;
    }

    std::vector<sycl::queue> queues;
    std::transform(devices.begin(), devices.end(), std::back_inserter(queues), [](const sycl::device &D) { return sycl::queue{D}; });
    

    if (!devices[0].ext_oneapi_can_access_peer(devices[1], sycl::ext::oneapi::peer_access::access_supported)) {
      std::cout << "P2P access is not supported by devices, exiting." << std::endl;
      return timeReport;
    }
    devices[0].ext_oneapi_disable_peer_access(devices[1]);
    unsigned int numberOfElems = experimentArgs.numberOfElems;
    
    // Init Host Buffers
    double *buffer_dev_0_host = static_cast<double*>(malloc(experimentArgs.getBufferSize()));
    double *buffer_dev_1_host = static_cast<double*>(malloc(experimentArgs.getBufferSize())); 
    dataValidator.init_buffer(buffer_dev_0_host, experimentArgs.numberOfElems);

    // Allocate Dev Buffers 
    double *buffer_dev_0 = malloc<double>(numberOfElems, queues[0], usm::alloc::device);
    double *buffer_dev_1 = malloc<double>(numberOfElems, queues[1], usm::alloc::device);
    
    // Init Device Buffer
    queues[0].memcpy(buffer_dev_0, buffer_dev_0_host, experimentArgs.getBufferSize());
    
    // P2P copy performed here:
    auto start = std::chrono::high_resolution_clock::now();
    queues[1].copy(buffer_dev_0, buffer_dev_1, numberOfElems).wait();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_ms = end - start;
    timeReport.latency.time_ms = time_ms.count();
 
    //Validate Data
    queues[1].copy(buffer_dev_1, buffer_dev_1_host, numberOfElems).wait();
    dataValidator.validate_data(buffer_dev_0_host, buffer_dev_1_host, experimentArgs.numberOfElems);

    // Free memory
    free(buffer_dev_0_host);
    free(buffer_dev_1_host);
    sycl::free(buffer_dev_0, queues[0]);
    sycl::free(buffer_dev_1, queues[1]);

    return timeReport;
  }
  
  };
