#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ":\n";
        std::cout << "  Name: " << prop.name << "\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor * 4 << " bytes\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
    }

    return 0;
}