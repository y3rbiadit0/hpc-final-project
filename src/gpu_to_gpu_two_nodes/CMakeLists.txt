cmake_minimum_required(VERSION 3.10)
project(multiple-nodes-sycl-cuda-mpi)


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


# Build executables
add_executable(${PROJECT_NAME} ${REPORT_LIB_SRC} ${UTILS_SRC} main_gpu_to_gpu_2_nodes.cpp)

# Set the target compile options
target_compile_options(${PROJECT_NAME} PRIVATE -lcudart)

# Set target link libraries
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
