/*
 * unweightedGraph Class:
 * This class models an unweighted graph using the Compressed Sparse Row (CSR) format.
 * It is designed to efficiently represent and manage large sparse graphs.
 */

#ifndef __GRAPH__
#define __GRAPH__

#include <fstream>
#include <vector>
#include <utility>

typedef struct edge
{
	int u, v;
} edge;

typedef struct query {
	int u, v;
} query;

class unweightedGraph {
private:
    void DFS_ConnectedComp(int node, std::vector<bool> &visited);

public:
    int totalVertices;
    // Represents the total number of bi-directional edges in the graph.
    // For a graph with edges (A, B), both (A, B) and (B, A) are considered, so it's twice the uni-directional edges.
    int totalEdges;
    // Size of the uni-directional edge list.
    int undirected_edge_list_size;
    int root;
    int *offset = nullptr;
    int *neighbour = nullptr;
    int *U = nullptr;
    int *V = nullptr;
    int *degree = nullptr;
    std::vector<int> vertexlist;

    // Constructors & Destructor
    unweightedGraph();
    unweightedGraph(std::ifstream& edgeList);
    ~unweightedGraph();

    // Member functions
    void printCSR();
    int numberOfConnectedComponents();
    std::pair<bool, int> isConnected();
};

#define degree(G, n) (G.offset[n+1] - G.offset[n])
#define out_degree(G, n) (G.offset[n+1] - G.offset[n])
#define out_vertices(G, n) (&G.neighbour[G.offset[n]])

#endif