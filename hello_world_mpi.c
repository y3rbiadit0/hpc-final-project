/*
 * Alumno:  Merenda, Franco N.
 * Carrera: IoT Master 2024
 *     */

#include<stdio.h>
#include<mpi.h>

int main(int argc, char **argv){
	int node;
	int size_of_cluster;

	//Init MPI_Library
	MPI_Init(&argc, &argv);
		
	//Set communicator which will belong process ( MPI_COMM_WORLD)
	MPI_Comm_rank(MPI_COMM_WORLD, &node);
	MPI_Comm_size(MPI_COMM_WORLD, &size_of_cluster);
	
	if (node == 0) {
		// 1. Get Availables GPUs.
		// 2. Assign GPU to each process.
		// 3. Print information from each process.
	}


	//Finalize MPI
	MPI_Finalize(); 
	
	return 0;		
}
