#include "Graph_Partition/listranking.h"
//List Ranking starts here
__global__
void list_ranking_kernel(int* next, int* dist, int* new_next, int* new_dist, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    if(tid < n)
    {
        // printf("\nNext array for %d iteration : %d", itr_no, next[tid]);
        // printf("\nDist array : %d", dist[tid]);
        if(next[tid] != tid)
        {
            new_dist[tid] = dist[tid] + dist[next[tid]];
            new_next[tid] = next[next[tid]];
        }
        else
        {
          new_dist[tid] = 0;
          new_next[tid] = tid;
        }
    }
}


int* listRanking(int* d_next, int n, int last_edge) {
    
    std::vector<int> dist(n, 1);
    dist[last_edge] = 0;
    //Allocate GPU memory
    int *d_dist;
    int size = n * sizeof(int);
    cudaMalloc((void**)&d_dist, size);

    // std::vector<int> new_next(n);
    //Allocate two more arrays in GPU to avoid Race - conditions
    int *d_new_dist, *d_new_next;
    cudaMalloc((void**)&d_new_dist, size);
    cudaMalloc((void**)&d_new_next, size);

    //Copy data from CPU to GPU
    cudaMemcpy(d_dist, dist.data(), size, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int j = 0; j < std::ceil(std::log2(n)); ++j)
    {
        list_ranking_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_next, d_dist, d_new_next, d_new_dist, n);
        cudaDeviceSynchronize();
        int* temp = d_new_next;
        d_new_next = d_next;
        d_next = temp;
        temp = d_new_dist;
        d_new_dist = d_dist;
        d_dist = temp;

    }
    cudaDeviceSynchronize();

    #ifdef DEBUG
        std::vector<int> new_dist(n);
        cudaMemcpy(new_dist.data(), d_dist, size, cudaMemcpyDeviceToHost);
        std::cout << "Printing final distance array / Euler Tour array :\n";
        int j = 0;
        for(auto i : new_dist)
            std::cout << "dist[" << j++ <<"] = " << i << std::endl;
        std::cout << std::endl;
    #endif 
    
    // cudaFree(d_dist);
    cudaFree(d_new_dist);
    // cudaFree(d_new_next); becoz of pointer swapping, no need to free.
    return d_dist;
}