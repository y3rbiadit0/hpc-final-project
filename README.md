# Project

This project contains a program to test the communication of NVLinks using MPI/NVLink.


## IMPORTANT: Load modules
```shell
module load openmpi/4.1.6--gcc--12.2.0
module load cuda/12.1
```

## Make Commands

1. Create binary for MPI-Cuda example.
```shell 
make mpi_cuda_hello_world
```

2. Create binary for MPI example
```shell 
make mpi_hello_world
```

3. Clean files
```shell
make clean
```



## SYCL FULL PROJECT

go to ~/intel/ and source setvars, then compile this project!! This is to get access to latest acess to ext-peer-access functions
