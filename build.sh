#!/bin/sh
# Go to library source tree root and run the usual:
# make this an executable with chmod +x ./build.sh

# A bash script to create a build dir if it doesn't exist and then move into it.
# If the build dir exists, it will be deleted and recreated.

# Make the build dir if it doesn't exist.
mkdir -p build
# Move into the build dir.
cd build
## Debug mode code
# Debug but also optmised code to more closely resemble release code
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc ..
#cmake --build .
#cmake make ..
make dbg=1