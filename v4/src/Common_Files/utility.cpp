
/*___________________________________________________________ What's New : _____________________________________________________________
| 1. Enhanced vector printing feature: Display vector elements along with vector name with passing it explicitly.                       |
|                                                                                                                                       |
|_______________________________________________________________________________________________________________________________________*/

/*If you want to print the name of the vector without using a macro or passing it as an additional argument, 
it's not directly possible in C++. Variable names are not accessible at runtime, so you can't retrieve the name of a variable itself.

In C++, variable names are used by the compiler for identification and scoping during compilation. 
Once the code is compiled, the variable names are no longer available. */

#include "Common_Files/utility.h"

#include <iostream>
#include <vector>
#include <string>
#include <thrust/host_vector.h>


int count = 0;
std::vector<int> child_count;

void dfs(uint src, std::vector<std::vector<int>>& adjlist, std::vector<bool> visited) {
    visited[src] = true;
    int start = count++;
    // std::cout << src <<" ";
    for(auto i : adjlist[src])
    {
        if(!visited[i])
        {
            dfs(i, adjlist, visited);
        }
    }
    child_count[src] = count - start - 1;
    if(child_count[src] != 0)
        child_count[src]++;
}

std::vector <int> serial_sub_graph(std::vector<int>& parent, int root) {
    parent[root] = root;
    int n = parent.size();
    child_count.resize(n);
    std::vector<std::vector<int>> adjlist(n);
    std::vector<bool> visited(n, false);
    for(int i = 0; i < n; ++i) {
        if(i != parent[i])
        {
            int u = i;
            int v = parent[i];
            adjlist[u].push_back(v);
            adjlist[v].push_back(u);
        }
    }
    dfs(root, adjlist, visited);

    std::transform(child_count.begin(), child_count.end(), child_count.begin(), [](int num) 
        { 
            // if(num)
            //     return num - 1;
            // else
            //     return num; 
            return (num > 0) ? num - 1 : num;
        });
    return child_count;
}

void displayVector(const std::vector<int>& arr, const std::string& name) {
    std::cout <<"vector " << name << ": ";
    for(auto i : arr)
        std::cout << i <<" ";
    std::cout<<std::endl;
}

void printCSRGraph(int* offset, int* neighbour, int offset_size) {
    std::cout << "Vertex -> Adjacents" << std::endl;
    for (int i = 0; i < offset_size - 1; ++i) {
        std::cout << i << " -> ";
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            std::cout << neighbour[j] << " ";
        }
        std::cout << std::endl;
    }
}

void print(const int* arr, int n) {
    for(int i = 0; i < n; ++i)
        std::cout << arr[i] <<" ";
    std::cout << std::endl;
}

void print(const std::vector<int>& arr) {
    for(auto i : arr)
        std::cout << i <<" ";
    std::cout << std::endl;
}

bool verify_vector(const std::vector<int>& v1, const std::vector<int>& v2) {
    if(v1.size() != v2.size()) {
        std::cout<<"\nSize mismatching..\n";
        return false;
    }

    std::vector<int> error;

    for(uint i = 0; i< v1.size(); ++i) {
        if(v1[i] != v2[i])
            error.push_back(i);
    }
    if(!error.empty()) {
        std::cout <<"\nVerification failed.\n";
        return false;
        // PRINT(error);
    } else {
        std::cout<<"\tVerification Over. No error\n";
        return true;
    }
}

bool verify_vector(const thrust::host_vector<int>& v1, const thrust::host_vector<int>& v2) {
    if(v1.size() != v2.size())
    {
        std::cout << "\nSize mismatching..\n";
        return false;
    }
    
    std::vector<int> error;
    for(uint i = 0; i < v1.size(); ++i)
    {
        if(v1[i] != v2[i]) { 
            error.push_back(i);
            std::cout << "Failed for " << i << " v1 value = " << v1[i] << " v2 value = " << v2[i] << std::endl;
        }
    }
    
    if(!error.empty()) {
        // PRINT(error);
        std::cout <<"\nVerification failed.\n";
        return false;
    }
    else {
        std::cout << "\tVerification Over. No error\n";
        return true;
    }
}

bool verify_vector(const int* v1, const int* v2, const int v_size) {

    std::vector<int> error;

    for(int i = 0; i< v_size; ++i) {
        if(v1[i] != v2[i])
            error.push_back(i);
    }
    if(!error.empty()) {
        std::cout <<"\nVerification failed.\n";
        return false;
        // PRINT(error);
    } else {
        std::cout<<"\tVerification Over. No error\n";
        return true;
    }
}

bool verify_vector(const std::vector<std::pair<int, int>>& vec1, const std::vector<std::pair<int, int>>& vec2) {
    return vec1 == vec2;
}

std::vector<int> serialPrefixSum(std::vector<int> arr) {
    int n = arr.size();
    int sum = 0;
    std::vector<int> result_arr(n+1, 0);
    for(uint i = 0; i < arr.size(); ++i) {
        sum += arr[i];
        result_arr[i+1] = sum;
    }
    return result_arr;
}

