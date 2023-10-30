#include "Common_Files/serialScan.h"
#include <vector>

//exclusive scan int
void exclusivePrefixSum(std::vector<int> &arr, int* prefixSum)
{
    prefixSum[0] = 0;
    for (size_t i = 0; i < arr.size(); i++)
        prefixSum[i+1] = prefixSum[i] + arr[i];
}

//exclusive scan bool
void exclusivePrefixSum(std::vector<bool> &arr, int* prefixSum)
{
    prefixSum[0] = 0;
    for (size_t i = 0; i < arr.size(); i++)
        prefixSum[i+1] = prefixSum[i] + arr[i];
}

//inclusive scan int
void inclusivePrefixSum(std::vector<int> &arr, std::vector<int> &prefixSum)
{
    prefixSum[0] = arr[0];
    for (size_t i = 1; i < arr.size(); i++)
        prefixSum[i] = prefixSum[i - 1] + arr[i];
}

//inclusive scan bool
void inclusivePrefixSum(std::vector<bool> &arr, std::vector<int> &prefixSum)
{
    prefixSum[0] = arr[0];
    for (size_t i = 1; i < arr.size(); i++)
        prefixSum[i] = prefixSum[i - 1] + arr[i];
}
