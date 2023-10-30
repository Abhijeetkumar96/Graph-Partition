#ifndef PARTITION_H
#define PARTITION_H

#include <vector>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "Common_Files/Graph_CSR.h"
#include "Common_Files/error_checking.h"

class PartitionDataStructure {

private:
    int N;
    int* parent = nullptr;
    int* levels = nullptr;
    int* d_parent = nullptr;
    int* d_child_count = nullptr;
    int* d_child_num = nullptr;
    int* starting_index = nullptr;

public: 
    // Constructor to initialize the object
    PartitionDataStructure(int N);
    ~PartitionDataStructure(); // Destructor to free memory

    // Declared the EulerianPartition function as a friend
    friend void EulerianPartition(unweightedGraph& G, PartitionDataStructure& partition_ds, unweightedGraph& G1, unweightedGraph& G2, unweightedGraph& G3, std::vector<int>& where, int*& re_numbering);
};

void EulerianPartition(unweightedGraph& G, PartitionDataStructure& partition_ds, unweightedGraph& G1, unweightedGraph& G2, unweightedGraph& G3, std::vector<int>& where, int*& re_numbering);

#endif //PARTITION_H