#include <assert.h>
#include <mpi.h>

#include <sycl/sycl.hpp>


int main(int argc, char *argv[]) {
  // MPI - Node ID
	int node_id, mpi_size;
  
  // Init MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_size != 2) {
    if (node_id == 0) {
      printf(
          "This program requires exactly 2 MPI ranks, "
          "but you are attempting to use %d! Exiting...\n",
          mpi_size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Initialize SYCL - Queue
  sycl::queue q{};

  // Initialize SYCL - Variables
  int tag = 0;
  const int nelem = 20; // Buffer Elements number
  const size_t nsize = nelem * sizeof(int); //Buffer
  std::vector<int> data(nelem, -1);

  // Create SYCL USM for each node in device (GPU)
  int *devp = sycl::malloc_device<int>(nelem, q);

  // Send and Receive Data
  if (node_id == 0) {
    // Copy the data to the node_id 0 device and wait for the memory copy to
    // complete.
    q.memcpy(devp, &data[0], nsize).wait();

    // Operate on the node_id 0 data.
    auto pf = [&](sycl::handler &h) {
      auto kern = [=](sycl::id<1> id) { devp[id] *= 2; };
      h.parallel_for(sycl::range<1>{nelem}, kern);
    };

    q.submit(pf).wait();

    // Send the data from node_id 0 to node_id 1.
    MPI_Send(devp, nsize, MPI_BYTE, 1, tag, MPI_COMM_WORLD);
    printf("Sent %d elements from %d to 1\n", nelem, node_id);
  } else {
    assert(node_id == 1);
    MPI_Status status;
    
    // Receive the data sent from node_id 0.
    MPI_Recv(devp, nsize, MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);
    printf("received status==%d\n", status.MPI_ERROR);

    // Copy the data back to the host and wait for the memory copy to complete.
    q.memcpy(&data[0], devp, nsize).wait();

    sycl::free(devp, q);

    // Validate buffer values
    for (int i = 0; i < nelem; ++i) assert(data[i] == -2);
  }
  MPI_Finalize();
  return 0;
}