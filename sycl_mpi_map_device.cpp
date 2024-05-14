#include <CL/sycl.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace sycl;

int main(int argc, char *argv[]) {
    // MPI - Node ID
    int node_id, mpi_size;

    // Buffer Related Variables
    const int LBUF = 1000000;
    std::vector<float> sBuf_h(LBUF), rBuf_h(LBUF);
    int bufferSize = sizeof(float) * LBUF;

    // Cuda Related Variables
    int deviceCount;
    int device_id;

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Init SYCL
    auto devices = device::get_devices(info::device_type::gpu);
    deviceCount = devices.size();

    if (deviceCount == 0) {
        std::cerr << "No GPU devices found.\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    device_id = node_id % deviceCount;
    device my_device = devices[device_id];

    queue my_queue(my_device);

    std::cout << "Number of GPUs found = " << deviceCount << std::endl;
    std::cout << "Assigned GPU " << device_id << " to MPI rank " << node_id << " of " << mpi_size << ".\n";

    // Initialize host buffers
    for (int i = 0; i < LBUF; ++i) {
        sBuf_h[i] = static_cast<float>(node_id);
        rBuf_h[i] = -1.0f;
    }

    if (node_id == 0) {
        std::cout << "rBuf_h[0] = " << rBuf_h[0] << std::endl;
    }

    // Allocate Device Buffers
    float* sBuf_d = malloc_device<float>(LBUF, my_device, my_queue.get_context());
    float* rBuf_d = malloc_device<float>(LBUF, my_device, my_queue.get_context());

    // Move Data from CPU -> Device
    my_queue.memcpy(sBuf_d, sBuf_h.data(), bufferSize).wait();

    // Send Data
    if (node_id == 0) {
        MPI_Recv(rBuf_d, LBUF, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (node_id == 1) {
        MPI_Send(sBuf_d, LBUF, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    } else {
        std::cerr << "Unexpected node value " << node_id << std::endl;
        exit(-1);
    }

    // Move Data from Device -> CPU
    my_queue.memcpy(rBuf_h.data(), rBuf_d, bufferSize).wait();

    if (node_id == 0) {
        std::cout << "rBuf_h[0] = " << rBuf_h[0] << std::endl;
    }

    // Free Device Buffers
    free(sBuf_d, my_queue.get_context());
    free(rBuf_d, my_queue.get_context());

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
