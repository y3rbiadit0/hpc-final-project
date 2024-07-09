cmake_minimum_required(VERSION 3.10)
project(hpc-final-project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the DPC++ compiler
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)


# 1. Set Compilers
set(CMAKE_C_COMPILER "mpiicpc")
set(DPCPP_COMPILER "/leonardo/home/userexternal/fmerenda/intel/oneapi/compiler/latest/bin/dpcpp")
set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER}) # Set the DPC++ compiler as the C++ compiler

# Set include/lib for SYCL
include_directories(/leonardo/home/userexternal/fmerenda/intel/oneapi/compiler/latest/include)
link_directories(/leonardo/home/userexternal/fmerenda/intel/oneapi/compiler/latest/lib)

# 2. Set include/lib CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# 3. Set include/lib MPI Library
include_directories(${MPI_INCLUDE_DIRS})
link_directories(${MPI_LIBRARIES})


# Build executables
include_directories(gpu_to_gpu_single_node/ report_lib/ utils/)
file(GLOB GPU_TO_GPU_SRC ${CMAKE_SOURCE_DIR}/src/gpu_to_gpu_single_node/*.cpp)
file(GLOB REPORT_LIB_SRC ${CMAKE_SOURCE_DIR}/report_lib/*.cpp)
file(GLOB UTILS_SRC ${CMAKE_SOURCE_DIR}/utils/data_validator.hpp)

add_executable(hpc-final-project ${REPORT_LIB_SRC} ${GPU_TO_GPU_SRC} ${UTILS_SRC} main_gpu_to_gpu_1_node.cpp)
add_executable(device-query-sycl utils/device_query_sycl.cpp)

# Set the target compile options
target_compile_options(hpc-final-project PRIVATE -fsycl) 
target_compile_options(device-query-sycl PRIVATE -fsycl) 

# Set target link libraries
target_link_libraries(hpc-final-project ${MPI_LIBRARIES} ${CUDA_LIBRARIES} -lcudart -lsycl)
target_link_libraries(device-query-sycl ${CUDA_LIBRARIES} -lcudart -lsycl)
