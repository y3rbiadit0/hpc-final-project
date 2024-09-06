cmake_minimum_required(VERSION 3.20)

project(GPU_Communication LANGUAGES CXX)


add_subdirectory(src/gpu_to_gpu_two_nodes)
add_subdirectory(src/gpu_to_gpu_single_node)