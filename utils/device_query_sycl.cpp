#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    int deviceCount = devices.size();
    std::cout << "Number of GPUs found = " << deviceCount << std::endl;

    for (const auto& device : devices) {
            std::cout << "  Device Name: " << device.get_info<sycl::info::device::name>() << "\n";
            std::cout << "  Device Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
            std::cout << "  Device Driver Version: " << device.get_info<sycl::info::device::driver_version>() << "\n";
            std::cout << "  Device Global Memory: " << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) << " MB\n";
            std::cout << "  Device Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
            std::cout << "  Device Max Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
            std::cout << "  Device Local Memory: " << device.get_info<sycl::info::device::local_mem_size>() / 1024 << " KB\n";
            std::cout << "  Device Max Clock Frequency: " << device.get_info<sycl::info::device::max_clock_frequency>() << " MHz\n";
            std::cout << "  Device Is Available: " << (device.get_info<sycl::info::device::is_available>() ? "Yes" : "No") << "\n";
            std::cout << "------------------------------------------" << "\n";
    }
    
    auto platforms = sycl::platform::get_platforms();
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
        auto devices = platform.get_devices();
        for (const auto& device : devices) {
            std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << "\n";
        }
    }
    return 0;
}
