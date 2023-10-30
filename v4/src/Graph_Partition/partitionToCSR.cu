// nvcc -std=c++17 -Xcompiler -fopenmp -Iinclude -c 
// src/Graph_Partition/partitionToCSR.cu -o obj/partitionToCSR.o -O2 -DDEBUG

#include "Graph_Partition/partitionToCSR.h"

#include "Common_Files/error_checking.h"
#include "Common_Files/Graph_CSR.h"
#include "Common_Files/serialScan.h"
#include "Common_Files/utility.h"

#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/find.h>


__global__
void constructPairsKernel(const int* where, int2* v, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
    	v[i] = make_int2(where[i], i);
    }
}

__global__ 
void firstOccurance_kernel(int2 *arr, int n, int target, int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check the current and previous elements
    if (idx < n && arr[idx].x == target && (idx == 0 || arr[idx-1].x != target)) {
        *result = idx;
    }
}

__global__ 
void init_graph(int *graph_0, int *graph_1, int *mapped_vert, int2 *v, int pos, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // if(tid == 0)
    // {
    //     printf("pos = %d\n", pos);
    //     for(int i = 0; i < n; ++i)
    //         printf("\n v[%d] : v[i].y = %d ", i, v[i].y);
    //     printf("\n");
    // }

    if (tid < pos) {
        // Corresponds to init_graph_0
        graph_0[tid] = v[tid].y;
        mapped_vert[v[tid].y] = tid;
    } else if (tid >= pos && tid < n) {
        // Corresponds to init_graph_1
        graph_1[tid - pos] = v[tid].y;
        mapped_vert[v[tid].y] = tid - pos;
    }
}

struct compare_int2 {
    __host__ __device__ bool operator()(int2 a,int2 b){return (a.x!=b.x) ? (a.x<b.x):(a.y<b.y);}
} cmp;

int firstOccurance(const int* d_where, const int n, const int target, thrust::device_vector<int2>& v) {

    int2* v_ptr = thrust::raw_pointer_cast(v.data());
    const int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    constructPairsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_where, v_ptr, n);
    cudaDeviceSynchronize();
    thrust::sort(v.begin(), v.end(), cmp);
    #ifdef DEBUG
        // Copy from device_vector to host_vector
        thrust::host_vector<int2> h_v = v;

        std::cout << "After sorting: \n";
        for(auto& i : h_v) {
            std::cout << i.x << " " << i.y << std::endl;
        }
        std::cout << std::endl;

        std::cout <<"target = " << target << std::endl;
    #endif
    
    int *d_result, h_result = -1;

    check_for_error(cudaMalloc(&d_result, sizeof(int)), "Unable to accolate memory for result");
    check_for_error(cudaMemcpy(d_result, &h_result, sizeof(int), 
                               cudaMemcpyHostToDevice), "Unable to copy result to host");

    // Invoke the kernel
    firstOccurance_kernel<<<blocksPerGrid, threadsPerBlock>>>(v_ptr, n, target, d_result);
    cudaDeviceSynchronize();

    // Copy the result back
    check_for_error(cudaMemcpy(&h_result, d_result, sizeof(int), 
                               cudaMemcpyDeviceToHost), "Unable to copy back result");

    if (h_result != -1) {
        printf("First occurrence at index: %d\n", h_result);
    } else {
        printf("Element not found\n");
    }

    cudaFree(d_result);
    return h_result;
}

void createCSRFromPartitionedGraph(const int* d_parent, const int n, const int target, int numVertices, int numEdges, 
                                    int* offset, int* neighbour, const int neighbour_size, const int offset_size, 
                                    std::vector<int>& edge_u, std::vector<int>& where, int*& mapped_vert, 
                                    unweightedGraph& G1, unweightedGraph& G2, unweightedGraph& G3) {
    #ifdef DEBUG
        std::cout <<"Printing csr for main graph : \n";
        printCSRGraph(offset, neighbour, offset_size);
        std::cout <<"Neighbour array : \n"; print(neighbour, neighbour_size);
    #endif

    thrust::device_vector<int2> v(n);
	int pos = firstOccurance(d_parent, n, target, v);
	std::cout << "Sorted successfully." << std::endl;
    std::cout << "pos = " << pos << std::endl;

	std::vector <int> graph_0(pos), graph_1(n - pos);
	int *d_graph_0, *d_graph_1, *d_mapped_vert;
    
    mapped_vert = new int[n];

    check_for_error(cudaMalloc(&d_graph_0, pos * sizeof(int)), 
                    "Unable to allocate memory for graph_0");
    check_for_error(cudaMalloc(&d_graph_1, (n - pos) * sizeof(int)), 
                    "Unable to allocate memory for graph_1");
    check_for_error(cudaMalloc(&d_mapped_vert, n * sizeof(int)), 
                    "Unable to allocate memory for mapped_vert");

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    int2* v_ptr = thrust::raw_pointer_cast(v.data());
    init_graph<<<blocksPerGrid, threadsPerBlock>>>(d_graph_0, d_graph_1, d_mapped_vert, v_ptr, pos, n);
    cudaDeviceSynchronize();
    
    check_for_error(cudaMemcpy(mapped_vert, d_mapped_vert, n * sizeof(int), cudaMemcpyDeviceToHost), 
                    "Unable to copy back mapped_vert back");
	check_for_error(cudaMemcpy(graph_0.data(), d_graph_0, pos * sizeof(int), cudaMemcpyDeviceToHost), 
                    "Unable to copy back graph_0 back");
	check_for_error(cudaMemcpy(graph_1.data(), d_graph_1, (n - pos) * sizeof(int), cudaMemcpyDeviceToHost), 
                    "Unable to copy back graph_1 back");

    #ifdef DEBUG
        std::cout <<"\nGraph 0 size = "<< graph_0.size() << std::endl;
        std::cout <<"Graph 1 size = "<< graph_1.size() << std::endl;

        std::cout << "mapped vertices : \n";
        for(int i = 0; i < n; ++i)
            std::cout << i << " : " << mapped_vert[i] << std::endl;
        int jj = 0;
        std::cout <<"Graph 0 : \n";
        for(auto i : graph_0)
            std::cout << jj++ <<" : " << i << std::endl;
        std::cout << std::endl;

        jj = 0;
        std::cout <<"Graph 1 : \n";
        for(auto i : graph_1)
            std::cout << jj++ <<" : " << i << std::endl;
        std::cout << std::endl; 
    #endif
    
    std::vector<int> degree_0(pos);
    std::vector<int> degree_1(n - pos);

    std::vector<int> flag_0(2 * numEdges, 0);
    std::vector<int> flag_1(2 * numEdges, 0);  
    std::vector<int> flag_cross_edges(2 * numEdges, 0);

    //for graph_0
    #pragma omp parallel for
    for(uint i = 0; i < graph_0.size(); ++i)
    {
        //explore all neighbours of each vertex
        uint k = graph_0[i];
        for(int j = offset[k]; j < offset[k+1]; ++j)
        {
            uint ll = neighbour[j];
            
            if(where[k] == where[ll]) {
                degree_0[i]++;
                flag_0[j] = 1;
            }
            else
            {
                //it is a cross-edge
                flag_cross_edges[j] = true;
            }
        }
    }

    //for graph_1
    #pragma omp parallel for
    for(uint i = 0; i < graph_1.size(); ++i)
    {
        //explore all neighbours of each vertex
        uint k = graph_1[i];
        for(int j = offset[k]; j < offset[k+1]; ++j)
        {
            uint ll = neighbour[j];
            if(where[k] == where[ll])
            {
                degree_1[i]++;
                flag_1[j] = 1;
            }
            else
            {
                //it is a cross-edge
                flag_cross_edges[j] = true;
            }
        }
    }


    std::vector<int> pos_0(2 * numEdges, 0), pos_1(2 * numEdges, 0), pos_cross_edges(2 * numEdges, 0);
    int* graph_0_vertex_list = new int[pos + 1];
    int* graph_1_vertex_list = new int[n - pos + 1];

    G1.offset = graph_0_vertex_list;
    G2.offset = graph_1_vertex_list;

    //perform prefix sum for all
    inclusivePrefixSum(flag_0, pos_0);
    inclusivePrefixSum(flag_1, pos_1);
    inclusivePrefixSum(flag_cross_edges, pos_cross_edges);

    exclusivePrefixSum(degree_0, graph_0_vertex_list);
    exclusivePrefixSum(degree_1, graph_1_vertex_list);

	#ifdef DEBUG
	    std::cout << "graph_0_vertex_list : " ;print(graph_0_vertex_list, pos + 1);
	    std::cout << "graph_1_vertex_list : " ;print(graph_1_vertex_list, n - pos + 1);
	#endif

    int graph_0_vertex_list_size = pos + 1;
    int graph_1_vertex_list_size = n - pos + 1;

    int graph_0_edge_list_size = graph_0_vertex_list[graph_0_vertex_list_size - 1];
    int graph_1_edge_list_size = graph_1_vertex_list[graph_1_vertex_list_size - 1];

    int* graph_0_edge_list = new int[graph_0_edge_list_size];
    int* graph_1_edge_list = new int[graph_1_edge_list_size];

    G1.neighbour = graph_0_edge_list;
    G2.neighbour = graph_1_edge_list;

    std::vector<std::pair<int, int>> cross_edges(pos_cross_edges[pos_cross_edges.size() - 1]);

    G1.totalVertices = graph_0.size();
    G1.totalEdges = graph_0_edge_list_size;

    G2.totalVertices = graph_1.size();
    G2.totalEdges = graph_1_edge_list_size;

    #ifdef DEBUG
        std::cout << "graph_0_edge_list.size() = " << graph_0_edge_list_size << std::endl;
        std::cout << "graph_1_edge_list.size() = " << graph_1_edge_list_size << std::endl;
    #endif

    // G3.U = new int[cross_edges.size()];
    // G3.V = new int[cross_edges.size()];

    int g3_size = cross_edges.size();

    check_for_error(cudaMallocHost((void**)&(G3.U), sizeof(int) * (g3_size)), 
                    "Unable to allocate G3's U array");
    check_for_error(cudaMallocHost((void**)&(G3.V), sizeof(int) * (g3_size)), 
                    "Unable to allocate G3's V array");
    
    G3.totalVertices = numVertices;
    G3.totalEdges = cross_edges.size();
    G3.undirected_edge_list_size = cross_edges.size();

    check_for_error(cudaMallocHost((void**)&(G2.U), sizeof(int) * (G2.totalEdges)), 
                    "Unable to allocate G2's U array");
    check_for_error(cudaMallocHost((void**)&(G2.V), sizeof(int) * (G2.totalEdges)), 
                    "Unable to allocate G2's V array");

    // Populating edge list array
    #pragma omp parallel for
    for(uint i = 0; i < neighbour_size; ++i) 
    {
        if(flag_0[i])
            graph_0_edge_list[pos_0[i] - 1] = mapped_vert[neighbour[i]];
        if(flag_1[i]) 
        {
            graph_1_edge_list[pos_1[i] - 1] = mapped_vert[neighbour[i]];
            // std::cout << pos_1[i] - 1 <<" ";
            G2.U[pos_1[i] - 1] = mapped_vert[edge_u[i]];
            G2.V[pos_1[i] - 1] = mapped_vert[neighbour[i]];
        }
        if(flag_cross_edges[i])
        {
            int index = pos_cross_edges[i]; 
            G3.U[index - 1] = edge_u[i];
            G3.V[index - 1] = neighbour[i];

            #ifdef DEBUG
            	cross_edges[index - 1] = std::make_pair(edge_u[i], neighbour[i]);
            #endif
        }
    }

	#ifdef DEBUG
	    std::cout << "graph 0 edge list : "; print(graph_0_edge_list, graph_0_edge_list_size);
	    std::cout << "graph 1 edge list : "; print(graph_1_edge_list, graph_1_edge_list_size);
	    std::cout << "cross edges : \n";
	    for(auto i : cross_edges)
	        std::cout << i.first << ", " <<i.second <<"\n ";
	    std::cout << std::endl;
	#endif

    G2.root = 0;
    #ifdef DEBUG
        std::cout << "G2.totalEdges = " << G2.totalEdges << std::endl;
        std::cout <<"Pinned memory output : \n";
        print(G2.U, G2.totalEdges);
        print(G2.V, G2.totalEdges);
    #endif
   
    #ifdef VERIFY
        //************************************* Verification starts here *******************************************
        // v_ denotes verify

	    std::vector<int> v_degree_0(graph_0.size());
	    std::vector<int> v_degree_1(graph_1.size());

	    std::vector<int> v_flag_0(2 * numEdges, 0), v_flag_1(2 * numEdges, 0), v_flag_cross_edges(2 * numEdges, 0);
	    
	    //for graph_0
	    for(uint i = 0; i < graph_0.size(); ++i)
	    {
	        //explore all neighbours of each vertex
	        uint k = graph_0[i];
	        // std::cout << "k = " <<k <<" ";
	        for(int j = offset[k]; j < offset[k+1]; ++j)
	        {
	            // std::cout << "neighbour[j] = " <<neighbour[j] <<" ";
	            if(where[k] == where[neighbour[j]])
	            {
	                v_degree_0[i]++;
	                v_flag_0[j] = 1;
	            }
	            else
	            {
	                //it is a cross-edge
	                v_flag_cross_edges[j] = 1;
	            }
	        }
	    }
	    //for graph_1
	    for(uint i = 0; i < graph_1.size(); ++i)
	    {
	        //explore all neighbours of each vertex
	        uint k = graph_1[i];
	        for(int j = offset[k]; j < offset[k+1]; ++j)
	        {
	            if(where[k] == where[neighbour[j]])
	            {
	                v_degree_1[i]++;
	                v_flag_1[j] = 1;
	            }
	            else
	            {
	                //it is a cross-edge
	                v_flag_cross_edges[j] = 1;
	            }
	        }
	    }

	    // std::cout << "v flag_0 : ";           print(v_flag_0);
	    // std::cout << "v flag_1 : ";           print(v_flag_1);
	    // std::cout << "v flag_cross_edges : "; print(v_flag_cross_edges);
	    // std::cout << "v degree 0 : ";         print(v_degree_0);
	    // std::cout << "v degree 1 : ";         print(v_degree_1);

	    std::vector<int> v_pos_0(2 * numEdges, 0), v_pos_1(2 * numEdges, 0), v_pos_cross_edges(2 * numEdges, 0);
	    
	    std::vector<int> v_graph_0_vertex_list(graph_0.size() + 1);
	    std::vector<int> v_graph_1_vertex_list(graph_1.size() + 1);

	    // //perform prefix sum for all
	    inclusivePrefixSum(v_flag_0, v_pos_0);
	    inclusivePrefixSum(v_flag_1, v_pos_1);
	    inclusivePrefixSum(v_flag_cross_edges, v_pos_cross_edges);

	    exclusivePrefixSum(v_degree_0, v_graph_0_vertex_list.data());
	    exclusivePrefixSum(v_degree_1, v_graph_1_vertex_list.data());

	    // std::cout << "pos_0 : " ; print(v_pos_0);
	    // std::cout << "pos_1 : " ; print(v_pos_1);
	    // std::cout << "pos_cross_edge : " ; print(v_pos_cross_edges);

	    // std::cout << "graph_0_vertex_list : " ;print(v_graph_0_vertex_list);
	    // std::cout << "graph_1_vertex_list : " ;print(v_graph_1_vertex_list);

	    std::vector<int> v_graph_0_edge_list(v_graph_0_vertex_list[v_graph_0_vertex_list.size() - 1]);
	    std::vector<int> v_graph_1_edge_list(v_graph_1_vertex_list[v_graph_1_vertex_list.size() - 1]);
	    std::vector<std::pair<int, int>> v_cross_edges(v_pos_cross_edges[v_pos_cross_edges.size() - 1]);

	    // std::cout << "graph_0_edge_list.size() = " << v_graph_0_edge_list.size();
	    // std::cout << "graph_1_edge_list.size() = " << v_graph_1_edge_list.size();

	    for(uint i = 0; i < neighbour_size; ++i)
	    {
	        if(flag_0[i])
	            v_graph_0_edge_list[pos_0[i] - 1] = mapped_vert[neighbour[i]];
	        if(flag_1[i])
	            v_graph_1_edge_list[pos_1[i] - 1] = mapped_vert[neighbour[i]];
	        if(flag_cross_edges[i])
	        {
	            int index = pos_cross_edges[i];
	            v_cross_edges[index - 1] = std::make_pair(edge_u[i], neighbour[i]);
	        }
	    }
        // **************** //

	    // std::cout << "graph 0 edge list : "; print(v_graph_0_edge_list);
	    // std::cout << "graph 1 edge list : "; print(v_graph_1_edge_list);
	    // std::cout << "cross edges : \n";
	    // for(auto i : cross_edges)
	    //     std::cout << i.first << ", " <<i.second <<"\n ";
	    // std::cout << std::endl;


	    std::cout << "Flag 0 verification = " <<verify_vector(v_flag_0, flag_0) << std::endl;
	    std::cout << "Flag 1 verification = " <<verify_vector(v_flag_1, flag_1) << std::endl;
	    std::cout << "Flag cross_edges verification = " <<verify_vector(v_cross_edges, cross_edges) << std::endl;

	    std::cout << "Graph 0 edge list verification = " 
        <<verify_vector(graph_0_edge_list, v_graph_0_edge_list.data(), graph_0_edge_list_size) << std::endl;
	    std::cout << "Graph 1 edge list verification = " 
        <<verify_vector(graph_1_edge_list, v_graph_1_edge_list.data(), graph_1_edge_list_size) << std::endl;
	    std::cout << "Cross edges verification = "       
        <<verify_vector(cross_edges, v_cross_edges) << std::endl;

	#endif

    cudaFree(d_graph_0);
    cudaFree(d_graph_1);
    cudaFree(d_mapped_vert);
}