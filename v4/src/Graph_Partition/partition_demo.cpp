// This code demonstrates how to use the partitioning header.

#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <cuda.h>

#include "Common_Files/mytimer.h"
#include "Common_Files/Graph_CSR.h"
#include "Common_Files/error_checking.h"
#include "Graph_Partition/partition.h"
#include "Graph_Partition/euler.h"

int main(int argc, char **argv) {

	// input format  = (#(vertices in G) , #(edges in G) , G in CSR Format using two arrays , sequence of edges from G)
	// Dataset path should be the first argument
	// output format = (Vertex BCC number and Edge BCC number)
	// The vertices are numbered 0 to n-1
	// Second argument should be the batch size
	// Third argument should be the number of threads
	// Fourth arguement should be pipeline design 1 or 2 denoted by s1 or s2

	/************************** Initialization ********************************/

	#ifdef DEBUG
    		std::cout << "DEBUG mode is on!" << std::endl;
		#else
    		std::cout << "DEBUG mode is off!" << std::endl;
	#endif

	mytimer module_timer {};
	// Enable nested parallelism
    // omp_set_nested(1);

	// checking whether the dataset name is passed as an argument
	if (argc < 2) {
	    std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
	    return 1;
	}

	/********************** Reading Graph *************************/
	std::string filename = argv[1];
	std::ifstream inputGraph(filename);
	if(!inputGraph) {
		std::cerr <<"Unable to open file for reading.\n";
		return EXIT_FAILURE;
	}

	unweightedGraph G(inputGraph);
	std::cout << "Created unweightedGraph" << endl;

	module_timer.timetaken_reset("Reading Data and Creating CSR");

	#ifdef VERIFY
		auto result_G = G.isConnected();
		if(result_G.first) {
		    // The graph is connected
		    std::cout <<"Graph G is connected.\n";
		} else {
		    // The graph is not connected and has result.second connected components
		    std::cout <<"G is not connected. NumComponents are : " << result_G.second <<".\n";
		}
	#endif

	/******************************************************************/

	/*********************** Setting num of threds *********************/

	// cudaFree(0);

	mytimer Algo_timer{};

	/**************** Here we partition the graph ************************/
	
	unweightedGraph G1, G2, E3;
	vector<int> where(G.totalVertices);
	int *renumbering = nullptr;

	PartitionDataStructure partition_ds(G.totalVertices);

	module_timer.timetaken_reset("Paritioning data structures initialised");

	// EulerianPartition function takes input graph G as input and returns three graphs G1, G2 and cross_edges
	// and the partitioned array as output
	// input <- G
	// output <- G1, G2, E3, where, initial_renumbering

	mytimer Partition_timer {};
    EulerianPartition(G, partition_ds, G1, G2, E3, where, renumbering);
    Partition_timer.timetaken_reset("Paritioning took: ");
	/*****************************************************************************************/
    #ifdef DEBUG
		std::cout <<"G1.totalVertices = " << G1.totalVertices <<endl;
	    std::cout << "G1.totalEdges = " << G1.totalEdges <<"\n";
	    std::cout <<"G2.totalVertices = " << G2.totalVertices <<endl;
	    std::cout << "G2.totalEdges = " << G2.totalEdges <<"\n";

		G1.printCSR();
		G2.printCSR();
	#endif

	G.undirected_edge_list_size = G.totalEdges/2;
    G1.undirected_edge_list_size = G1.totalEdges/2;
	G2.undirected_edge_list_size = G2.totalEdges/2;

	#ifdef VERIFY
		auto result_G1 = G1.isConnected();
		if(result_G1.first) {
		    // The graph is connected
		    std::cout <<"G1 graph is connected.\n";
		} else {
		    // The graph is not connected and has result.second connected components
		    std::cout <<"G1 is not connected. NumComponents are : " << result_G1.second <<".\n";
		}

		auto result_G2 = G2.isConnected();
		if(result_G2.first) {
		    // The graph is connected
		    std::cout <<"G2 graph is connected.\n";
		} else {
		    // The graph is not connected and has result.second connected components
		    std::cout <<"G2 is not connected. NumComponents are : " << result_G2.second <<".\n";
		}

	#endif


    if(renumbering) {
    	std::cout <<"Freeing renumbering.\n";
    	delete[] renumbering;
    }
	std::cout << std::endl;

return EXIT_SUCCESS;
}
