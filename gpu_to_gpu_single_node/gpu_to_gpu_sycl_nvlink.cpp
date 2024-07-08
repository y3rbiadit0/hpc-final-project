#include <assert.h>

#include <sycl/sycl.hpp>
#include <cassert>
#include <numeric>
#include <vector>


#include "../report_lib/experiment.h"
#include "../report_lib/experiment_args.h"
using namespace sycl;

class GPUtoGPU_CUDA_SYCL_NVLINK : public Experiment<double>{
  TimeReport run(ExperimentArgs<double> args)
  {

    TimeReport timeReport = TimeReport();
    std::vector<sycl::device> devices;
    
    
    for (const auto &plt : sycl::platform::get_platforms()) {
      std::cout <<"Device: " << plt.get_backend() << std::endl;
      if (plt.get_backend() == sycl::backend::ext_oneapi_cuda)
        devices.push_back(plt.get_devices()[0]);
    }


    if (devices.size() <= 2) {
      std::cout << "Cannot test P2P capabilities, at least two devices are "
                   "required, exiting."
                << std::endl;
      return timeReport;
    }

    std::vector<sycl::queue> queues;
    std::transform(devices.begin(), devices.end(), std::back_inserter(queues),
                   [](const sycl::device &D)
                   { return sycl::queue{D}; });
    ////////////////////////////////////////////////////////////////////////

    if (!devices[0].ext_oneapi_can_access_peer(devices[1], sycl::ext::oneapi::peer_access::access_supported)) {
      std::cout << "P2P access is not supported by devices, exiting." << std::endl;
      return timeReport;
    }

    // Enables Devs[0] to access Devs[1] memory.
    devices[0].ext_oneapi_enable_peer_access(devices[1]);
    unsigned int numberOfElems = args.numberOfElems;
    
    std::vector<int> input(numberOfElems);
    std::iota(input.begin(), input.end(), 0);

    int *arr0 = malloc<int>(numberOfElems, queues[0], usm::alloc::device);
    queues[0].memcpy(arr0, &input[0], numberOfElems * sizeof(int));

    int *arr1 = malloc<int>(numberOfElems, queues[1], usm::alloc::device);
    // P2P copy performed here:
    queues[1].copy(arr0, arr1, numberOfElems).wait();

    int out[numberOfElems];
    queues[1].copy(arr1, out, numberOfElems).wait();

    sycl::free(arr0, queues[0]);
    sycl::free(arr1, queues[1]);

    bool ok = true;
    for (int i = 0; i < numberOfElems; i++)
    {
      if (out[i] != input[i])
      {
        printf("%d %d\n", out[i], input[i]);
        ok = false;
        break;
      }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");

    return timeReport;
  }
  
  };
