#include <sycl/sycl.hpp>
#include <cassert>
#include <numeric>
#include <vector>
#include <chrono>

#include "experiment.hpp"
#include "time_report.hpp"
#include "data_validator.hpp"

using namespace sycl;

class GPUtoGPU_SYCL_NVLINK_BIDI : public Experiment<double> {
public:
  TimeReport run(ExperimentArgs<double> experimentArgs) {
    TimeReport timeReport = TimeReport();
    std::vector<sycl::device> devices;
    DataValidator<double> dataValidator;
    
    // Find CUDA platforms and get their devices
    for (const auto &plt : sycl::platform::get_platforms()) {
      if (plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
        devices.push_back(plt.get_devices()[0]);
      }
    }

    if (devices.size() < 2) {
      std::cout << "Cannot test P2P capabilities, at least two devices are required, exiting."
                << std::endl;
      return timeReport;
    }

    std::vector<sycl::queue> queues;
    std::transform(devices.begin(), devices.end(), std::back_inserter(queues), [](const sycl::device &D) { 
      return sycl::queue{D}; 
    });
    
    // Check if P2P access is supported and enable it
    if (!devices[0].ext_oneapi_can_access_peer(devices[1], sycl::ext::oneapi::peer_access::access_supported)) {
      std::cout << "P2P access is not supported by devices, exiting." << std::endl;
      return timeReport;
    }
    devices[0].ext_oneapi_enable_peer_access(devices[1]);
    devices[1].ext_oneapi_enable_peer_access(devices[0]);

    unsigned int numberOfElems = experimentArgs.numberOfElems;
    
    // Initialize host buffers
    double *buffer_dev_0_host = static_cast<double*>(malloc(experimentArgs.getBufferSize()));
    double *buffer_dev_1_host = static_cast<double*>(malloc(experimentArgs.getBufferSize())); 

    dataValidator.init_buffer(buffer_dev_0_host, experimentArgs.numberOfElems);

    // Allocate device buffers 
    double *buffer_dev_0_upstream = malloc<double>(numberOfElems, queues[0], usm::alloc::device);
    double *buffer_dev_0_downstream = malloc<double>(numberOfElems, queues[0], usm::alloc::device);

    double *buffer_dev_1_upstream = malloc<double>(numberOfElems, queues[1], usm::alloc::device);
    double *buffer_dev_1_downstream = malloc<double>(numberOfElems, queues[1], usm::alloc::device); 

    // Initialize device buffer for GPU 0
    queues[0].memcpy(buffer_dev_0_upstream, buffer_dev_0_host, experimentArgs.getBufferSize()).wait();
    queues[0].memcpy(buffer_dev_1_downstream, buffer_dev_0_host, experimentArgs.getBufferSize()).wait();

    // Start the bidirectional P2P copy simultaneously
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy from GPU 0 to GPU 1
    auto event1 = queues[1].copy(buffer_dev_0_upstream, buffer_dev_1_upstream, numberOfElems);
    
    // Copy from GPU 1 to GPU 0 simultaneously
    auto event2 = queues[0].copy(buffer_dev_1_downstream, buffer_dev_0_downstream, numberOfElems);

    // Wait for both copy operations to complete
    event1.wait();
    event2.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_ms = end - start;
    timeReport.latency.time_ms = time_ms.count();

    // Validate data transfer in both directions
    queues[1].copy(buffer_dev_1_upstream, buffer_dev_1_host, numberOfElems).wait();
    dataValidator.validate_data(buffer_dev_0_host, buffer_dev_1_host, experimentArgs.numberOfElems);

    queues[0].copy(buffer_dev_0_downstream, buffer_dev_1_host, numberOfElems).wait();
    dataValidator.validate_data(buffer_dev_0_host, buffer_dev_1_host, experimentArgs.numberOfElems);

    // Free memory
    free(buffer_dev_0_host);
    free(buffer_dev_1_host);
    sycl::free(buffer_dev_0_upstream, queues[0]);
    sycl::free(buffer_dev_1_upstream, queues[1]);
    sycl::free(buffer_dev_0_downstream, queues[0]);
    sycl::free(buffer_dev_1_downstream, queues[1]);

    // Disable P2P access to reset the device status for subsequent runs
    devices[0].ext_oneapi_disable_peer_access(devices[1]);
    devices[1].ext_oneapi_disable_peer_access(devices[0]);

    return timeReport;
  }
};
