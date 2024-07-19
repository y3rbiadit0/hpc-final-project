#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("COMM RANK: %d -- COMM SIZE: %d\n", rank, size);
	MPI_Status stat;

	if(size != 2){
		if(rank == 0){
			printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
		}
		MPI_Finalize();
		exit(0);
	}

	// Map MPI ranks to GPUs
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
	cudaSetDevice(0);
    
    printf("Assigned GPU %d to MPI rank %d of %d.\n", 0, rank, size);

	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for(int i=0; i<=27; i++){

        printf("Loop Interation %d\n", i);
		long int N = 1 << i;
	
		// Allocate memory for A on CPU
        printf("MemoryAllocation \n");
		double *A = (double*)malloc(N*sizeof(double));

        printf("Initialization \n");
		// Initialize all elements of A to random values
		for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
		}

        printf("Cuda malloc \n");
		double *d_A;
		cudaMalloc((void **)&d_A, N*sizeof(double)) ;
		cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) ;
	
		int tag1 = 10;
		int tag2 = 20;
	
		int loop_count = 50;

		// Warm-up loop
        printf("Warmup\n");
		for(int i=1; i<=5; i++){
			if(rank == 0){
                printf("MPI SEND RANK 0\n");
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                printf("MPI rECEIVE RANK 0\n");
				MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
			else if(rank == 1){
                printf("MPI rECEIVE RANK 1\n");
				MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                printf("MPI SEND RANK 1\n");
				MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}
		}
        printf("Time Ping pong\n");
		// Time ping-pong for loop_count iterations of data transfer size 8*N bytes
		double start_time, stop_time, elapsed_time;
		start_time = MPI_Wtime();
	
		for(int i=1; i<=loop_count; i++){
			if(rank == 0){
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
			else if(rank == 1){
				MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}
		}

		stop_time = MPI_Wtime();
		elapsed_time = stop_time - start_time;

		long int num_B = 8*N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

		if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

		cudaFree(d_A);
		free(A);
	}

	MPI_Finalize();

	return 0;
}