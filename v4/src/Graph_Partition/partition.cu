#include <iostream>
#include <cuda_runtime.h>

#include "Graph_Partition/partition.h"
#include "Graph_Partition/euler.h"
#include "Graph_Partition/partitionToCSR.h"

#include "Common_Files/bfs.h"
#include "Common_Files/Graph_CSR.h"
#include "Common_Files/error_checking.h"


/*  unweightedGraph Class: Member Variables Description
 * -----------------------------------------------------
 * 
 * - totalVertices
 * - totalEdges: Counts bi-directional edges.
 * - undirected_edge_list_size: Count of uni-directional edges.
 * - offset
 * - neighbour
 * - U, V: Edge stream.
 */

// func EulerianPartition : input -> (G, partition_ds), output->(G1, G2, G3, where, re_numbering)

/*
PartitionDataStructure Class: Member Variables Description

│
├── BFS Data
│   ├── parent        : Tracks the parent of each node of main graph G.
│   └── levels        : Holds the level (or depth) of each node in the spanning tree.
│
└── Euler (GPU) Data
    ├── d_parent      : Corresponding parent nodes on device.
    ├── d_child_count : Number of child nodes for each node.
    ├── d_child_num   : Specific child node numbers.
    └── starting_index: Initial index for partition segments.
*/


PartitionDataStructure::PartitionDataStructure(int N) {
    this->N = N; // this->N refers to the member variable N, and the standalone N refers to the constructor parameter.
    parent = new int[N];
    levels = new int[N];

    int d_size = N * sizeof(int);

    check_for_error(cudaMalloc((void**)&d_parent, d_size), "Error allocating memory for d_parent");
    check_for_error(cudaMalloc((void**)&d_child_count, d_size), "Error allocating memory for d_child_count");
    check_for_error(cudaMalloc((void**)&d_child_num, d_size), "Error allocating memory for d_child_num");
    check_for_error(cudaMalloc((void**)&starting_index, (N+1) * sizeof(int)), "Error allocating memory for starting_index");

    check_for_error(cudaMemset(d_child_count, 0, d_size), "Error in cudaMemset of d_child_count");
    check_for_error(cudaMemset(d_child_num, 0, d_size), "Error in cudaMemset of d_child_num");
    check_for_error(cudaMemset(starting_index, 0, (N+1) * sizeof(int)), "Error in cudaMemset of starting_index");

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        levels[i] = -1;
    }
}

PartitionDataStructure::~PartitionDataStructure() {
    std::cout <<"Executing partition destructor\n";
    if(parent) 
        delete[] parent;
    if(levels) 
        delete[] levels;
    
    if(d_parent) 
        check_for_error(cudaFree(d_parent), "Error freeing memory for d_parent");

    if(d_child_count) 
        check_for_error(cudaFree(d_child_count), "Error freeing memory for d_child_count");

    if(d_child_num) 
        check_for_error(cudaFree(d_child_num), "Error freeing memory for d_child_num");

    if(starting_index) 
        check_for_error(cudaFree(starting_index), "Error freeing memory for starting_index");

    parent = levels = d_parent = d_child_count = d_child_num = starting_index = nullptr;
}

void EulerianPartition(unweightedGraph& G, PartitionDataStructure& partition_ds, unweightedGraph& G1, unweightedGraph& G2, unweightedGraph& G3, std::vector<int>& where, int*& re_numbering) {
	
    std::cout << "Partitioning started." << std::endl;

    #ifdef DEBUG
        std::cout << "Debugging mode is ON." << std::endl;
    #endif

    int N = partition_ds.N;
    int* parentPtr = partition_ds.parent;
    int* levelsPtr = partition_ds.levels;
    int root = 1;
    // int root = G.root;
	bfs(G, root, parentPtr, levelsPtr);

	std::cout << "\n\nbfs is done\n";

    #ifdef DEBUG
    	std::cout << "\nthe parent array is :-\n";
    	for (int i = 0; i < G.totalVertices; ++i) {
    		std::cout << i <<" : " << parentPtr[i] << "\n";
    	}
    	std::cout << std::endl;
    #endif

    // Setting up local values
    int* d_parent = partition_ds.d_parent;
    int* d_child_count = partition_ds.d_child_count;
    int* d_child_num = partition_ds.d_child_num;
    int* starting_index = partition_ds.starting_index;

    parentPtr[root] = -1;
    check_for_error(cudaMemcpy(d_parent, parentPtr, N * sizeof(int), cudaMemcpyHostToDevice), "Unable to copy parent array to device");
    parentPtr[root] = root;
    int target = parallelSubgraphComputation(d_parent, root, N, d_child_count, d_child_num, starting_index);

    check_for_error(cudaMemcpy(where.data(), d_parent, N * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back where array to host");

    #ifdef DEBUG
        std::cout << "Where array : \n";
        for(auto i : where)
            std::cout << i <<" ";
        std::cout << std::endl;
    #endif

    std::cout << "Preparing CSR for partitioned graph..." << std::endl;

    createCSRFromPartitionedGraph(d_parent, N, target, G.totalVertices, G.totalEdges/2, G.offset, G.neighbour, G.totalEdges, G.totalVertices + 1, G.vertexlist, where, re_numbering, G1, G2, G3);
}
