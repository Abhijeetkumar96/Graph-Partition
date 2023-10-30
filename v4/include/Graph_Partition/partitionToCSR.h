#ifndef PARTITION_TO_CSR_H
#define PARTITION_TO_CSR_H

#include "Common_Files/Graph_CSR.h"

#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

int firstOccurance(const int* d_where, const int n, const int target, thrust::device_vector<int2>& v);

void createCSRFromPartitionedGraph(const int* d_parent, const int n, const int target, int numVertices, int numEdges, 
                        int* offset, int* neighbour, const int neighbour_size, const int offset_size,
                        std::vector<int>& edge_u, std::vector<int>& where, int*& mapped_vert,
                        unweightedGraph& G1, unweightedGraph& G2, unweightedGraph& G3);

#endif