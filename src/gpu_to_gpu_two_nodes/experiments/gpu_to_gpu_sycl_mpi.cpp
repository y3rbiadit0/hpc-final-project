#include <sycl/sycl.hpp>
#include <iostream>
#include <mpi.h>
#include <cstdlib>

#include "statistic.hpp"
#include "experiment.hpp"
#include "data_validator.hpp"

class GPUtoGPU_SYCL_MPI: public Experiment<double> {
public:
   TimeReport run(ExperimentArgs<double> experimentArgs) {
        TimeReport timeReport = TimeReport();

        // MPI - Node ID
        int node_id, mpi_size;

        // Buffer Related Variables
        double* sBuf_shared = nullptr;  
        double* rBuf_shared = nullptr;

        // SYCL Related Variables
        sycl::queue queue;
        sycl::device device;
        sycl::context context;

        DataValidator<double> dataValidator = DataValidator<double>();

        // Init MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Initialize SYCL queue
        queue = sycl::queue(sycl::gpu_selector_v);
        context = queue.get_context();
        device = queue.get_device();

        const size_t numElems = experimentArgs.numberOfElems;
        const size_t bufferSize = experimentArgs.getBufferSize();

        // Allocate Shared Memory
        if (node_id == 0) {
            sBuf_shared = (double*) sycl::malloc_device(bufferSize, device, context);
            dataValidator.init_buffer_with_value(sBuf_shared, numElems, 1.0);
        } else {
            rBuf_shared = (double*) sycl::malloc_device(bufferSize, device, context);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        timeReport.latency.start();

        if (node_id == 0) {
            // Directly copy data from shared memory (device buffer) to MPI buffer
            MPI_Send(sBuf_shared, numElems, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        } else {
            // Receive data into shared memory
            MPI_Recv(rBuf_shared, numElems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        timeReport.latency.end();

        // Clean up
        if (node_id == 0) {
            sycl::free(sBuf_shared, context);
        }
        if (node_id == 1) {
            sycl::free(rBuf_shared, context);
        }

        return timeReport;
    }
};
