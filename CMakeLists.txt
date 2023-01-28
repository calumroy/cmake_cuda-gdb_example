
# cmake_minimum_required(VERSION 3.16)
# project(gpu_overlap CUDA CXX)

# enable_language(CUDA)
# set(CUDA_VERBOSE_BUILD ON)
# find_package(CUDA REQUIRED)

# set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# cuda_add_executable(${PROJECT_NAME} gpu_overlap.cu)

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")

# IF (CMAKE_BUILD_TYPE MATCHES "Debug")
#     target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# ENDIF()

# # Required for CUDA debugging in Visual Studio Code. THis adds a -G flag to compilation.
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()

cmake_minimum_required(VERSION 3.16)
project(gpu_overlap CUDA CXX C)  # enable cuda language
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE gpu_overlap.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)