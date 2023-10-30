#ifndef SERIALSCAN_H
#define SERIALSCAN_H

#include <vector>

//exclusive scan for int
void exclusivePrefixSum(std::vector<int> &arr, int* prefixSum);

//exclusive scan for bool
void exclusivePrefixSum(std::vector<bool> &arr, int* prefixSum);

//inclusive scan for int
void inclusivePrefixSum(std::vector<int> &arr, std::vector<int> &prefixSum);

//inclusive scan for bool
void inclusivePrefixSum(std::vector<bool> &arr, std::vector<int> &prefixSum);

#endif