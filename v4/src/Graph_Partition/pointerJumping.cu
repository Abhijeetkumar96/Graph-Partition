#include "Graph_Partition/pointerJumping.h"

#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <random>
#include <chrono>

__global__
void pointer_jumping_kernel(int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {

        if(next[tid] != tid)
        {
            next[tid] = next[next[tid]];
        }
    }
}

void pointer_jumping(int* d_next, int n)
{
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    auto parallel_start = std::chrono::high_resolution_clock::now();  
    for (int j = 0; j < std::ceil(std::log2(n)); ++j)
    {
        pointer_jumping_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, n);
        cudaDeviceSynchronize();
    }

    auto parallel_end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(parallel_end - parallel_start).count();
    printf("Total time for parallel pointer jumping : %ld microseconds (%d number of keys)\n", parallel_duration, n);  
}
