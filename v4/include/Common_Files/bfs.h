#ifndef __BFS_H__
#define __BFS_H__

#define THREAD_QUEUE_SIZE 1024

#include "Common_Files/Graph_CSR.h"
#include <iostream>

double bfs(unweightedGraph &G, int root, int* parents, int* levels);

inline void add_to_queue(int* thread_queue, int& thread_queue_size, 
                         int* queue_next, int& queue_size_next, int vert, int max_size);
inline void empty_queue(int* thread_queue, int& thread_queue_size, 
                        int* queue_next, int& queue_size_next, int max_size);

inline void add_to_queue(int* thread_queue, int& thread_queue_size, 
                         int* queue_next, int& queue_size_next, int vert, int max_size)
{
  thread_queue[thread_queue_size++] = vert;

  if (thread_queue_size == THREAD_QUEUE_SIZE)
    empty_queue(thread_queue, thread_queue_size, 
                queue_next, queue_size_next,max_size);
}

inline void empty_queue(int* thread_queue, int& thread_queue_size, 
                        int* queue_next, int& queue_size_next, int max_size)
{
	int start_offset;

	#pragma omp atomic capture
		start_offset = queue_size_next += thread_queue_size;

	if (start_offset >= max_size) {
		std::cout << "the next queue size limit is reached\n";
		exit(-1);
	}

	start_offset -= thread_queue_size;
	for (int i = 0; i < thread_queue_size; ++i)
		queue_next[start_offset + i] = thread_queue[i];
	thread_queue_size = 0;
}

#endif