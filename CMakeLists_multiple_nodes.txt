cmake_minimum_required(VERSION 3.10)
project(hpc-final-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the custom directory for MPI
set(CMAKE_PREFIX_PATH "/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/")

# Find the DPC++ compiler
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER "mpic++")
set(CMAKE_C_COMPILER "mpicc")

# 2. Set include/lib CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# 3. Set include/lib MPI Library
include_directories(${MPI_INCLUDE_DIRS})
link_directories(${MPI_LIBRARIES})


# Build executables
include_directories(gpu_to_gpu_two_nodes/ report_lib/ utils/)
file(GLOB GPU_TO_GPU_SRC ${CMAKE_SOURCE_DIR}/src/gpu_to_gpu_two_nodes/*.cpp)
file(GLOB REPORT_LIB_SRC ${CMAKE_SOURCE_DIR}/report_lib/*.cpp)
file(GLOB UTILS_SRC ${CMAKE_SOURCE_DIR}/utils/data_validator.hpp)

add_executable(hpc-final-project-mpi ${REPORT_LIB_SRC} ${GPU_TO_GPU_SRC} ${UTILS_SRC} main_gpu_to_gpu_2_nodes.cpp)

# Set the target compile options
target_compile_options(hpc-final-project-mpi PRIVATE -lcudart) 


# Set target link libraries
target_link_libraries(hpc-final-project-mpi ${MPI_LIBRARIES} ${CUDA_LIBRARIES} -lcudart)
