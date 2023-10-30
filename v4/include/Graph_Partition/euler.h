#ifndef EULER_H
#define EULER_H

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int parallelSubgraphComputation(int* d_parent, const int root, const int N, int* d_child_count, int* d_child_num, int* starting_index);

#endif // EULER_H
