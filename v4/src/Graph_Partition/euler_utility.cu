#include "Graph_Partition/euler_utility.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

int getIndexClosestToHalfN(const std::vector<int>& values) {
    int n = values.size();
    double target = 0.5 * n;

    int closestIndex = 0;
    double closestDifference = std::abs(values[0] - target);

    for (int i = 1; i < n; ++i) {
        double difference = std::abs(values[i] - target);
        if (difference < closestDifference) {
            closestDifference = difference;
            closestIndex = i;
        }
    }

    return closestIndex;
}

struct abs_diff
{
    int target;
    abs_diff(int target) : target(target) {}
    __device__
    int operator()(const int& x) const { return abs(x - target); }
};

int getIndexinParallel(thrust::device_vector<int>& d_array, int target) {
    //Apply transformation directly on the provided device_vector
    thrust::transform(d_array.begin(), d_array.end(), d_array.begin(), abs_diff(target));

    //find minimum element
    thrust::device_vector<int>::iterator iter = thrust::min_element(d_array.begin(), d_array.end());

    int min_value = *iter;

    // Print the result
    std::cout << "Minimum value is " << min_value + target << std::endl;
    //return the result
    return iter - d_array.begin();
}
