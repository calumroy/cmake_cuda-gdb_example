# cmake_cuda-gdb_example
cmake cuda-gdb example in vscode.

build the project with 
```
./build.sh
```
Note this builds a debug version of the code to use CUDA-GDB for debugging.

## VSCODE
In vscode to debug the GPU code i.e the sliding_window_kernel function install the plugin
**Nsight Visual Studio Code Edition** and put a break point in the function and run the Launch task in the debug tab
named **CUDA C++: Launch**

