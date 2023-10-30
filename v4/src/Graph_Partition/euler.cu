// nvcc -arch=sm_61 -std=c++17 -O2 -DDEBUG test_euler.cu euler_v4.cu utility.cpp -o new_main

#include "Graph_Partition/euler.h"
#include "Graph_Partition/listranking.h"
#include "Graph_Partition/pointerJumping.h"
#include "Graph_Partition/euler_utility.h"
#include "Common_Files/utility.h"
#include "Common_Files/error_checking.h"

#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <chrono>

__global__ 
void find_degrees(int *d_parent, int *d_child_count, int *d_child_num, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < n && d_parent[idx] != -1) {
        /*
            Previously:
                int old_count = atomicAdd(&d_child_count[d_parent[idx]], 1);
                assign the child a number (old child count of parent)
                d_child_num[idx] = old_count;
            Updated to avoid extra variable initialization for each thread, 
            reducing unnecessary overhead and improving performance.
        */
        d_child_num[idx] = atomicAdd(&d_child_count[d_parent[idx]], 1);
    }
}

struct PopulateChildListFunctor {
    int offset_value;
    int root;
    int n_nodes; //Total number of edges
    int *prefix_sum;
    int *child_num;
    int *d_child_list;
    int *parent;
    int2 *d_edge_num;


    PopulateChildListFunctor(int _root, int _n_nodes, int *_prefix_sum, int *_child_num, int *_child_list, int *_parent, int2 *_edge_num)
        : root(_root), n_nodes(_n_nodes), prefix_sum(_prefix_sum), child_num(_child_num), d_child_list(_child_list), parent(_parent), d_edge_num(_edge_num) {}

    __device__ void operator()(const int& i) 
    {
        if(i != root) {
            /*
                Previously:
                    int position = prefix_sum[parent[i]] + child_num[i];
                    d_child_list[position] = i;
                    int pos = prefix_sum[parent[i]] + child_num[i];
                    d_edge_num[pos] = make_int2(parent[i],i);
                    int new_pos = pos + n_nodes - 1;
                    d_edge_num[new_pos] = make_int2(i, parent[i]);
                 Updated to avoid extra variable initialization for each thread, 
                reducing unnecessary overhead and improving performance.
            */

            d_child_list[prefix_sum[parent[i]] + child_num[i]] = i;
            d_edge_num[prefix_sum[parent[i]] + child_num[i]] = make_int2(parent[i],i);
            d_edge_num[prefix_sum[parent[i]] + child_num[i] + n_nodes - 1] = make_int2(i, parent[i]);
        }
    }
};

struct FindSuccessorFunctor 
{
    int root;
    int n_nodes;
    int* prefix_sum;
    int* child_num;
    int* child_list;
    int* parent;
    int* successor;
    int2* d_edge_num;
    int* child_count;
    int* d_last_edge;

    FindSuccessorFunctor(int _root, int _n_nodes, int *_prefix_sum, int *_child_num, int *_child_list, int *_parent, int *_successor, int2 *_edge_num, int *_child_count, int *_d_last_edge)
        :root(_root), n_nodes(_n_nodes), prefix_sum(_prefix_sum), child_num(_child_num), child_list(_child_list), parent(_parent), successor(_successor), d_edge_num(_edge_num), child_count(_child_count), d_last_edge(_d_last_edge) {}

    __device__ void operator()(const int& i) {
        
        int u = d_edge_num[i].x;
        int v = d_edge_num[i].y;
        //step 1, check if forward edge, i.e. from parent to child
        if(parent[v] == u) {
            
          //i] Check if v has any calculate_children
            if(child_count[v] > 0) 
            {
                //if yes then go to the first child of v;
                //new edge will be from v to 0th child of v
                //d_child_list[prefix_sum[v]] will give me first child of v, as prefix_sum[v] denotes the starting of d_child_list of v
                successor[i] = find_edge_num(v, child_list[prefix_sum[v]]);
                #ifdef DEBUG
                    printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v]]);
                #endif
                return;
            }
            else {
              
              //No child, then go back to parent.
              successor[i] = find_edge_num(v, parent[v]);
              #ifdef DEBUG
                printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, parent[v]);
              #endif
              return;
            }
        }

        //it is an back-edge
        else 
        {
            //check if it is going back to root  
            if(v == root)
            {
                if(child_num[u] == child_count[root] - 1) //checking if it is the last child
                {
                    int val = find_edge_num(u,v);
                    successor[i] = val;
                    *d_last_edge = val;

                    #ifdef DEBUG
                        printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v]]);
                        // printf("\nSuccessor[%d] = %d", i, val);
                    #endif
                    return;
                }
                else {
                    successor[i] = find_edge_num(v, child_list[prefix_sum[root] + child_num[u] + 1]);
                    #ifdef DEBUG
                        printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v] + child_num[u] + 1]);
                    #endif
                    return;
                }
            }

            //find child_num of u
            //check if it was the last child of v, if yes then the edge will be back to parent.
            int child_no_u = child_num[u];
            if(child_no_u == child_count[v] - 1) {
                //go to the parent of 0;
                successor[i] = find_edge_num(v, parent[v]);
                #ifdef DEBUG
                    printf("\nSuccessor[%d,%d] = %d, %d",u, v, v, parent[v]);
                #endif
            }
            else {
            //It is not the last child
            successor[i] = find_edge_num(v, child_list[prefix_sum[v] + child_num[u] + 1 ]);
            #ifdef DEBUG
                printf("\nSuccessor[%d,%d] = %d, %d", u, v, v, child_list[prefix_sum[v] + child_num[u] + 1 ]);
            #endif
            }
        }
    }
    __device__ __forceinline__ int find_edge_num(int u, int v) 
    {
        //check if forward edge
        if(parent[v] == u) 
        {
            #ifdef DEBUG
                printf("\n(%d, %d) <- %d", u, v, prefix_sum[u] + child_num[v]);
            #endif
            return prefix_sum[u] + child_num[v];
        }
        else 
        {
            #ifdef DEBUG
                printf("\n(%d, %d) <- %d", u, v, prefix_sum[v] + child_num[u] + + n_nodes - 1);
            #endif
            return prefix_sum[v] + child_num[u] + n_nodes;
        }
    }
};

struct updateRank {
    int* euler_tour_arr;
    int n_edges;
    int* rank;

    updateRank(int *_euler_tour_arr, int _n_edges, int *_rank)
    : euler_tour_arr(_euler_tour_arr), n_edges(_n_edges), rank(_rank) {}

    __device__
    void operator()(const int& i)
    {
        rank[i] = n_edges - 1 - euler_tour_arr[i];
    }
};

__device__ __forceinline__ int findedge_num(int u, int v, int* parent, int* prefix_sum, int* child_num, int n_nodes) 
{
    // printf("Child num array : ");
    // for(int i = 0; i < n_nodes; ++i)
    // printf("child_num[%d] = %d ",i, child_num[i]);
    if(parent[v] == u)
    {
    //printf("u = %d, v = %d, parent[%d] = %d, prefix_sum[%d] = %d, child_num[%d] = %d\n", u, v, v, parent[v], parent[v], prefix_sum[parent[v]], v, child_num[v]);
        return prefix_sum[u] + child_num[v];
    }
    else
    {
    // printf("u = %d, v = %d, parent[%d] = %d, prefix_sum[%d] = %d, child_num[%d] = %d\n", u, v, u, parent[u], parent[u], prefix_sum[parent[u]], u, child_num[u]);
        return prefix_sum[v] + child_num[u] + n_nodes - 1;
    }
}

__global__ 
void computeSubGraphSize(int* sub_graph_size, int* child_count, int* child_num, int* d_child_list, 
                         int* starting_index, int* parent, int* rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) 
    return;

//    if(i == 0)
//    {
//      printf("Child count array : ");
//      for(int j = 0; j < n; ++j)
//          printf("\nchild_count[%d] = %d ", j, child_count[j]);

//          printf("Child num array : ");
//            for(int j = 0; j < n; ++j)
//                    printf("child_num[%d] = %d ", j, child_num[j]);
//    }

    if (!child_count[i]) {
        sub_graph_size[i] = 0;
    }

    else {
        int node_0 = i;
        int node_1 = d_child_list[starting_index[i] + 0];
        int node_2 = d_child_list[starting_index[i] + child_count[i] - 1];

        // printf("node_0 = %d, node_1 = %d, node_2 = %d \n", node_0, node_1, node_2);

        int edge_1 = findedge_num(node_0, node_1, parent, starting_index, child_num, n);
        int edge_2 = findedge_num(node_2, node_0, parent, starting_index, child_num, n);

        // printf("For vertex i = %d, first child = %d, second_child = %d,  edge_1 = %d, edge_2 = %d\n", i, node_1, node_2, edge_1, edge_2);

        // printf("\nFor vertex %d, first child = %d, second child = %d, rank[%d] = %d , rank[%d] = %d ", i, node_1, node_2, edge_1, rank[edge_1], edge_2, rank[edge_2]);

        sub_graph_size[i] = (((rank[edge_2] - rank[edge_1]) + 1) / 2 + 1) - 1;
    }
}


/**
 * Function: parallelSubgraphComputation()
 *
 * Description:
 *   - This function conducts an Euler tour on a graph represented by its parent 
 *     relationships and returns the child_count of each node.
 *
 * Parameters:
 *   - @param d_parent: An integer pointer representing the parent of each node in the graph. 
 *                     Essential for actual computation.
 *   - @param root: The root node of the tree.
 *   - @param N: The total number of nodes in the forest.
 *   
 *   - @param d_child_count: An integer pointer for storing the count of children for each node.
 *   - @param d_child_num: An integer pointer storing a unique number for each child in the graph.
 *   - @param starting_index: An integer pointer representing the starting index for each node 
 *                            in the csr_graph.
 *
 * Notes:
 *   - The function assumes d_parent, root, and N are primary inputs for computation.
 *   - The arrays d_child_count, d_child_num, and starting_index serve as intermediate storage 
 *     for results and necessary computations.
 */
int parallelSubgraphComputation(int* d_parent, int root, const int N, int* d_child_count, int* d_child_num, int* starting_index) {

    // Get device properties
    int n = N;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Assuming device 0, modify if you have multiple devices

    // Calculate optimal block size and grid size
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; // Maximum number of threads per block
    int maxBlocksPerGrid = deviceProp.maxGridSize[0]; // Maximum number of blocks per grid

    int blockSize = min(maxThreadsPerBlock, N); // Number of threads in each block
    int gridSize = min((N + blockSize - 1) / blockSize, maxBlocksPerGrid); // Number of blocks in the grid

    // Launch kernel with calculated dimensions
    find_degrees<<<gridSize, blockSize>>>(d_parent, d_child_count, d_child_num, n);
    check_for_error(cudaGetLastError(), "Error after some_kernel launch");
    cudaDeviceSynchronize();

    /*
    ** Prefix sum is done here:
    ** - The size is one greater than the input size to accommodate the sum that includes all input elements.
    ** - All elements of the starting_index array are initialized to 0, including the first one, 
         as we want the output to start from 0.
    ** - We perform an inclusive scan on the input.
    ** - We use an inclusive scan instead of an exclusive scan because we want to include the last element in the sum.
    ** - The result is stored starting from the second element of the 'd_output' array to ensure it starts from 0.
    */

    // Using Thrust's execution policies to specify algorithm backend:
    // - thrust::host runs the algorithm on the CPU
    // - thrust::device runs it on the GPU
    // By explicitly defining the policy, we ensure that the algorithm runs on the intended backend, 
    // preventing potential issues like segmentation faults from unintended host/device data access.

    thrust::inclusive_scan(thrust::device, d_child_count, d_child_count + n, starting_index + 1);
    cudaDeviceSynchronize();

    #if defined(DEBUG) || defined(VERIFY)
        std::vector<int> child_num(n);
        std::vector<int> child_count(n);
        std::vector<int> h_start_index(n + 1);

        int size = n * sizeof(int);
        
        check_for_error(cudaMemcpy(child_num.data(), d_child_num, size, cudaMemcpyDeviceToHost), "Unable to copy d_child_num back to device");
        check_for_error(cudaMemcpy(child_count.data(), d_child_count, size, cudaMemcpyDeviceToHost), "Unable to copy d_child_count back to device");
        check_for_error(cudaMemcpy(h_start_index.data(), starting_index, (n+1) * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy starting_index back to device");
    #endif

    #ifdef DEBUG
        std::cout << "Printing child_num array : \n";
        int j = 0;
        for(auto i : child_num)
            std::cout << j++ << " : " << i << std::endl;
        std::cout << std::endl;

        std::cout << "Printing child_count array : \n";
        j = 0;
        for(auto i : child_count)
            std::cout << j++ << " : " << i << std::endl;
        std::cout << std::endl;

        std::cout <<"prefix_sum : \n";
        for(int i = 0; i <= n; ++i)
            std::cout << h_start_index[i] <<" ";
        std::cout << std::endl;
    #endif

    #ifdef VERIFY
        std::vector<int> serial_result = serialPrefixSum(child_count);
        std::cout <<"Verifying prefix sum values\n";
        if(verify_vector(h_start_index, serial_result))
            std::cout <<"Successfully verified prefix_sum output." << std::endl;
        else
            std::cout <<"Error in prefix_sum output." << std::endl;
    #endif

    int edge_count = n - 1;
    int edges = edge_count * 2;

    #ifdef DEBUG
        std::cout <<"Edge count = " << edge_count << std::endl;
        std::cout <<"Edges = " << edges << std::endl;
    #endif

    thrust::device_vector<int2> d_edge_num(edges);
    thrust::device_vector<int> successor(edges);

    thrust::device_vector<int> d_child_list(n - 1, 0);

    thrust::for_each(thrust::device,
                     thrust::counting_iterator<int>(0), 
                     thrust::counting_iterator<int>(N), 
                     PopulateChildListFunctor(root,
                                               n,
                                              starting_index,
                                              d_child_num,
                                              thrust::raw_pointer_cast(d_child_list.data()),
                                              d_parent,
                                              thrust::raw_pointer_cast(d_edge_num.data()))); 

    cudaDeviceSynchronize();

    #ifdef DEBUG
        thrust::host_vector<int2> h_edge_num = d_edge_num; // Copying data from device to host

        std::cout <<"\nPrinting edge numbers : \n";
        for(int i = 0; i < h_edge_num.size(); ++i)
        {
            std::cout << i <<" : (" << h_edge_num[i].x <<", " << h_edge_num[i].y << ")" << std::endl;
        }

        std::cout <<"Child list array : \n";
        // PRINT(d_child_list);
        std::cout <<"d_child_list : \n";
        for(int i = 0; i < d_child_list.size(); ++i)
            std::cout << d_child_list[i] <<" ";
        std::cout << std::endl;
    #endif

    //To be used for list ranking
    int* d_last_edge;
    cudaMalloc((void**)&d_last_edge, sizeof(int));

    thrust::for_each(thrust::device,
                    thrust::counting_iterator<int>(0), 
                     thrust::counting_iterator<int>(edges), 
                     FindSuccessorFunctor(root,
                                          edge_count,
                                          starting_index,
                                          d_child_num,
                                          thrust::raw_pointer_cast(d_child_list.data()),
                                          d_parent,
                                          thrust::raw_pointer_cast(successor.data()),
                                          thrust::raw_pointer_cast(d_edge_num.data()),
                                          d_child_count,
                                          d_last_edge)); 

    cudaDeviceSynchronize();

     int h_last_edge;
    cudaMemcpy(&h_last_edge, d_last_edge, sizeof(int), cudaMemcpyDeviceToHost);

    #ifdef DEBUG
        //Copy back the data
        std::cout<<"\nPrinting successor array : \n";
        for(int i = 0; i<successor.size(); ++i) {
          std::cout<<"successor["<<i<<"] = "<<successor[i]<<"\n";
        }
        std::cout<<"h_last_edge = " << h_last_edge<<"\n"; 
    #endif

    //apply list ranking on successor to get Euler tour
    //store the tour in a different array
    int* d_euler_tour_arr = listRanking(thrust::raw_pointer_cast(successor.data()), edges, h_last_edge);

    #ifdef DEBUG
        std::vector<int> h_euler_tour_arr(edges);
        cudaMemcpy(h_euler_tour_arr.data(), d_euler_tour_arr, edges*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout <<"Euler tour array after applying listranking:\n";
        int jj = 0;
        for(auto i : h_euler_tour_arr)
            std::cout << "arr["<< jj++ <<"] : " << i << std::endl;
        std::cout << std::endl;
    #endif


    //After Eulerian Tour is ready, get the correct ranks
    //Update ranks, then calculate first and last

    //edges is 2 times the original number of edges

    thrust::device_vector<int> rank(edges);

    thrust::for_each(thrust::device,
                     thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(edges),
                     updateRank {d_euler_tour_arr,
                                edges,
                                thrust::raw_pointer_cast(rank.data())});
    
    cudaDeviceSynchronize();

    #ifdef DEBUG
        std::cout << "\nPrinting rank array : \n";
        j = 0;
        for(auto i : rank)
            std::cout << "rank[" <<j++ <<"] = " << i << std::endl;
        std::cout << std::endl;
    #endif

    thrust::device_vector<int> sub_graph_size(N);
    int* sub_graph_ptr = thrust::raw_pointer_cast(sub_graph_size.data());
    int* d_child_list_ptr = thrust::raw_pointer_cast(d_child_list.data());
    int* rank_ptr = thrust::raw_pointer_cast(rank.data());

    std::cout <<"Sub graph computation started. \n";
    // Launch the kernel
    computeSubGraphSize<<<gridSize, blockSize>>>(sub_graph_ptr, d_child_count, d_child_num, d_child_list_ptr, starting_index, d_parent, rank_ptr, N);
    std::cout <<"Sub graph computation over. \n";
    cudaDeviceSynchronize();

    #ifdef DEBUG
        std::cout <<"Sub graph array: \n";
        j = 0;
        for(auto i : sub_graph_size)
            std::cout << j++ <<" : " << i <<std::endl;
        std::cout << std::endl;
        std::cout <<"Parallel sub graph computation over.\n";
    #endif

    #ifdef VERIFY
        std::vector<int> h_parent(N);
        check_for_error(cudaMemcpy(h_parent.data(), d_parent, N * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy parent array to cpu");
        
        std::cout <<"Serial sub graph computation started. \n";
        std::vector<int> serial_child_count = serial_sub_graph(h_parent, root);
        std::cout <<"Serial sub graph computation ended. \n";

        thrust::host_vector<int> h_sub_graph(N);
        thrust::copy(sub_graph_size.begin(), sub_graph_size.end(), h_sub_graph.begin());

        //verify sub-graph 
        std::cout << "...SUB-GRAPH VERIFICATION STARTING...\n";
        if(verify_vector(h_sub_graph, serial_child_count))
            std::cout <<"Child count array verified.\n";
        else
            std::cout <<"Unable to verify the sub - graph array.\n";
    #endif

    #ifdef DEBUG
        std::cout << "Printing parallel sub graph : \n";
        for(int val : h_sub_graph) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "Printing serial sub graph : \n";
        for(int val : serial_child_count) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    #endif

    int target = N/2;
    int closestIndex = getIndexinParallel(sub_graph_size, target);
    
    // int value = output.first;
    // int closestIndex = output.second;
    std::cout << "Parallel output of Index with value closest to 0.5*n is: " << closestIndex << std::endl;

    #ifdef VERIFY
        int serial_index = getIndexClosestToHalfN(serial_child_count);
        if(serial_child_count[serial_index] == h_sub_graph[closestIndex]) {
            std::cout <<"Value at both the indices are matching\n";
            std::cout <<"Value = " << h_sub_graph[closestIndex] << std::endl;
        }
        else
            std::cout <<"Mismatch in values\n";
    #endif

    // d_parent[closestIndex] = closestIndex;
    // d_parent[src] = src;

    check_for_error(cudaMemcpy(&d_parent[closestIndex], &closestIndex, sizeof(int), cudaMemcpyHostToDevice), "Unable to update parent array with closestIndex value");
    check_for_error(cudaMemcpy(&d_parent[root], &root, sizeof(int), cudaMemcpyHostToDevice), "Unable to update root");

    pointer_jumping(d_parent, N);

    std::cout << "Graph partitioned successfully." << std::endl;

    cudaFree(d_last_edge);
    cudaFree(d_euler_tour_arr);
    return closestIndex; //Node at which partition occured.
}
