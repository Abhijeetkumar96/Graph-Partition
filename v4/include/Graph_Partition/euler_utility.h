// array_util.h
#ifndef EULER_UTILITY_H
#define EULER_UTILITY_H

#include <vector>
#include <thrust/device_vector.h>

int getIndexClosestToHalfN(const std::vector<int>& values);
int getIndexinParallel(thrust::device_vector<int>& d_array, int target);

#endif // EULER_UTILITY_H
