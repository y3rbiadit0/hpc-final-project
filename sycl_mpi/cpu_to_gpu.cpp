#include <assert.h>
#include <mpi.h>
#include <sycl/sycl.hpp>

void validate_nodes_size(int mpi_size, int node_id);

TimeReport cpu_to_gpu_sycl_mpi(int argc, char *argv[], ExperimentArgs args)
{

    TimeReport timeReport = TimeReport();

    // MPI - Node ID
    int node_id, mpi_size;

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    validate_nodes_size(mpi_size, node_id);

    // Initialize SYCL - Variables
    int tag = 0;
    const int nelem = args.bufferSize;        // Buffer Elements number
    const size_t nsize = nelem * sizeof(int); // Buffer

    // Initialize SYCL - Queue
    sycl::queue q{};

    // Create and initialize the host buffer -- Initialize with 1s
    std::vector<int> hostData(nsize, 1);
    sycl::buffer<int, 1> buffer(hostData.data(), sycl::range<1>(nsize));

    // Measure the time to transfer the buffer
    timeReport.latency.start();

    q.submit([&](sycl::handler& cgh) {
        auto acc = buffer.get_access<sycl::access::mode::read_write>(cgh);
        cgh.copy(acc, hostData.data());
    }).wait();

    timeReport.latency.end();

    //Finalize job
    MPI_Finalize();
    return timeReport;
}

void validate_nodes_size(int mpi_size, int node_id)
{
    if (mpi_size != 1)
    {
        if (node_id == 0)
        {
            printf(
                "This program requires exactly 1 MPI ranks, "
                "but you are attempting to use %d! Exiting...\n",
                mpi_size);
        }
        MPI_Finalize();
        exit(0);
    }
}