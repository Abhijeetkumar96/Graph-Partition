#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

#include "Common_Files/Graph_CSR.h"
#include "Common_Files/error_checking.h"

/* ********************************************************************************************************* */

/* unweightedGraph Class:
 * =======================

 * Member Variables:
 * -----------------
 * - totalVertices: Represents the total number of vertices in the graph.
 * - totalEdges: Represents the 2 times the number of unique edges. For a graph with edges (A, B), both (A, B) and (B, A) are counted.
 * - undirected_edge_list_size: Size of the uni-directional edge list.
 * - offset: Pointer to an array indicating the starting point in the 'neighbour' array for each vertex.
 * - neighbour: Pointer to an array listing adjacent vertices for a given vertex.
 * - U, V: Pointers to arrays indicating the source and target vertices for each edge, respectively.
 * - root: Denotes the root vertex of the graph, useful for traversal or specific graph algorithms.
 * - degree: Pointer to an array indicating the degree (number of neighbors) for each vertex.
 * 
 * Member Functions:
 * -----------------
 * - unweightedGraph(): Default constructor. Initializes an empty graph.
 * - unweightedGraph(ifstream& edgeList): Constructor to initialize the graph from an edge list.
 * - printCSR(): Prints the graph in its Compressed Sparse Row (CSR) format, useful for debugging.

************************************************************************************************************* */

unweightedGraph::unweightedGraph() {
	totalVertices = 0;
	totalEdges = 0;
}

unweightedGraph::unweightedGraph(std::ifstream& edgeList) {

	edgeList >> totalVertices >> totalEdges;
	offset = new int[totalVertices + 1];
	degree = new int[totalVertices]();
	neighbour = new int[totalEdges];

	vertexlist.resize(totalEdges);

	// storing the directed edges for calculating the degree
	U = new int[totalEdges];
	V = new int[totalEdges];
	
	for (int i = 0; i < totalEdges; i++) {
		int u, v;
		edgeList >> u >> v;
		degree[u]++;
		U[i] = u;
		V[i] = v;
	}

	// updating the offset array
	std::vector<int> edgeCounter(totalVertices);
	offset[0] = 0;
	for (int i = 0; i < totalVertices; i++)
	{
		offset[i + 1] = degree[i] + offset[i];
		edgeCounter[i] = offset[i];
	}

	// updating the neighbour array
	for (int i = 0; i < totalEdges; i++) {
		int u, v;
		u = U[i];
		v = V[i];

		int currIndex = edgeCounter[u];
		edgeCounter[u]++;
		neighbour[currIndex] = v;
	}

	root = 0;
	int count = 0;
	int maxDegree = 0;

	for (int i = 0; i < totalVertices; i++) {
		int u = i;
		if (degree[i] > maxDegree) {
			maxDegree = degree[i];
			root = i;
		}
		for (int j = offset[i]; j < offset[i + 1]; j++) {
			vertexlist[j] = i;
			int v = neighbour[j];
			if (u < v) {
				U[count] = u;
				V[count] = v;
				count++;
			}
		}
	}
	delete[] U;
	delete[] V;
	U = V = nullptr;
}

void unweightedGraph::printCSR() {
	std::cout << "Total Edges = " << totalEdges << std::endl;
	std::cout << "Total Vertices = " << totalVertices << std::endl;
    
    std::cout << "Vertex -> Adjacents" << std::endl;
	for (int i = 0; i < totalVertices; i++) {
		std::cout << i << " -> ";
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            std::cout << neighbour[j] << " ";
        }
        std::cout << std::endl;
    }
}

// DFS to find connected components
void unweightedGraph::DFS_ConnectedComp(int node, std::vector<bool> &visited) {
    visited[node] = true;
    for (int i = offset[node]; i < offset[node + 1]; i++) {
        int v = neighbour[i];
        if (!visited[v]) {
            DFS_ConnectedComp(v, visited);
        }
    }
}

// Find number of connected components
int unweightedGraph::numberOfConnectedComponents() {
    int num_components = 0;
    std::vector<bool> visited(totalVertices, false);
    for (int node = 0; node < totalVertices; ++node) {
        if (!visited[node]) {
            DFS_ConnectedComp(node, visited);
            num_components++;
        }
    }
    return num_components;
}

std::pair<bool, int> unweightedGraph::isConnected() {
    int num_components = numberOfConnectedComponents();
    return {num_components == 1, num_components};
}

unweightedGraph::~unweightedGraph() {
	if(U) {
	    std::cout <<"\nFreeing U.";
	    check_for_error(cudaFreeHost(U), "Failed to free U");
	}

	if(V) {
	    std::cout <<"\nFreeing V.";
	    check_for_error(cudaFreeHost(V), "Failed to free V");
	}

	if(degree) {
	    std::cout <<"\nFree degree.";
	    delete[] degree;
	}

	if(offset) {
	    std::cout <<"\nFreeing offset.";
	    delete[] offset;
	}
	if(neighbour) {
	    std::cout << "\nFreeing neighbour.";
	    delete[] neighbour;
	}
}