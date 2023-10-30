#ifndef __ERROR_CHECKING__
#define __ERROR_CHECKING__

#include <iostream>
#include <string>
#include <cuda_runtime.h> // Preferred for runtime APIs

inline void exitWithError(const std::string& message){
    std::cerr << message << std::endl;
    exit(-1);
}

inline void check_for_error(cudaError_t error, const std::string& message){
    if(error != cudaSuccess){
        std::cerr << message << std::endl;
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

#endif // ERROR_CHECKING_H
