cmake_minimum_required(VERSION 3.16)
project(gpu_overlap CUDA CXX C)  # enable cuda language
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 11)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE gpu_overlap.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)