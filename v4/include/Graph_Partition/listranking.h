#ifndef LISTRANKING_H
#define LISTRANKING_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

int* listRanking(int* d_next, int n, int last_edge);

#endif // LISTRANKING_H
