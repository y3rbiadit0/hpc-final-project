cmake_minimum_required(VERSION 3.20)
project(single-node-sycl-cuda)


find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)


# 1. Set DPCPP Compiler

set(DPCPP_COMPILER "/leonardo/home/userexternal/fmerenda/intel/oneapi/compiler/latest/bin/dpcpp")
set(CMAKE_C_COMPILER "mpiicpc")
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


include_directories(../utils)
include_directories(../report_lib)
include_directories(experiments)

set(REPORT_LIB_SRC
    ../report_lib/experiment_args.hpp
    ../report_lib/experiment.hpp
    ../report_lib/experiment_runner.cpp
    ../report_lib/statistic.cpp
    ../report_lib/statistic.hpp
    ../report_lib/time_counter.cpp
    ../report_lib/time_counter.hpp
    ../report_lib/time_report.cpp
    ../report_lib/time_report.hpp
)

set(UTILS_SRC
    ../utils/data_validator.hpp
)

add_executable(${PROJECT_NAME} ${REPORT_LIB_SRC} ${UTILS_SRC} main_gpu_to_gpu_1_node.cpp)

# Set the target compile options
target_compile_options(${PROJECT_NAME} PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -lpthread) 

# Set target link libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${CUDA_LIBRARIES}  -lsycl -lpthread)
