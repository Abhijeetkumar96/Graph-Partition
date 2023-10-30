#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <string>
#include <thrust/host_vector.h>

void printCSRGraph(int* offset, int* neighbour, int offset_size);

#define PRINT(arr) displayVector(arr, #arr)
void print(const int* arr, int n);
void print(const std::vector<int>& arr);

void dfs(uint src, std::vector<std::vector<int>>& adjlist, std::vector<bool> visited);
std::vector <int> serial_sub_graph(std::vector<int>& parent, int root);

void displayVector(const std::vector<int>& arr, const std::string& name);
void printVector(const std::vector<int>& vec);

bool verify_vector(const std::vector<std::pair<int, int>>& vec1, const std::vector<std::pair<int, int>>& vec2);
bool verify_vector(const thrust::host_vector<int>& v1, const thrust::host_vector<int>& v2);
bool verify_vector(const std::vector<int>& v1, const std::vector<int>& v2);
bool verify_vector(const int* v1, const int* v2, const int v_size);

std::vector<int> serialPrefixSum(std::vector<int> arr);

#endif // GRAPH_EDGE_DELETIONS_H