#include "Common_Files/bfs.h"

#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include "Common_Files/Graph_CSR.h"
#include <iostream>

#define NOT_VISITED_MARKER -1

#define ALPHA 15.0
#define BETA 24.0
// #define debug

using namespace std;

double bfs(unweightedGraph &G, int root, int* parents, int* levels)
{
    int num_verts = G.totalVertices;
    double avg_out_degree = G.totalEdges/(double)G.totalVertices;

    int* queue = new int[num_verts];
    int* queue_next = new int[num_verts];
    int queue_size = 0;  
    int queue_size_next = 0;

    queue[0] = root;
    queue_size = 1;
    parents[root] = root;
    levels[root] = 0;

    int level = 1;
    int num_descs = 0;
    int local_num_descs = 0;
    bool use_hybrid = false;
    bool already_switched = false;

    double time = omp_get_wtime();

    #pragma omp parallel
    {
        int thread_queue[ THREAD_QUEUE_SIZE ];
        int thread_queue_size = 0;

        while (queue_size)
        {
            if (!use_hybrid)
            {
                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int i = 0; i < queue_size; ++i)
                {
                    int vert = queue[i];
                    unsigned out_degree = out_degree(G, vert);
                    int* outs = out_vertices(G, vert);
                    #ifdef debug
                        //cout << vert << " is selected\n";
                        //cout << out_degree << " is out degree\n";
                    #endif              
                    
                    for (unsigned j = 0; j < out_degree; ++j)
                    {      
                        int out = outs[j];
                        #ifdef debug
                        //cout << out << " is selected as out\n";
                        //cout << levels[out] << " is the level of out\n";
                        #endif  

                        if (levels[out] < 0)
                        {
                            levels[out] = level;
                            parents[out] = vert;
                            ++local_num_descs;
                            add_to_queue(thread_queue, thread_queue_size, 
                            queue_next, queue_size_next, out,num_verts);
                        }
                    }
                }
            }
            else
            {
                int prev_level = level - 1;

                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int vert = 0; vert < num_verts; ++vert)
                {
                    if (levels[vert] < 0)
                    {
                        unsigned out_degree = out_degree(G, vert);
                        int* outs = out_vertices(G, vert);
                        for (unsigned j = 0; j < out_degree; ++j)
                        {
                            int out = outs[j];
                            if (levels[out] == prev_level)
                            {
                                levels[vert] = level;
                                parents[vert] = out;
                                ++local_num_descs;
                                add_to_queue(thread_queue, thread_queue_size, 
                                queue_next, queue_size_next, vert, num_verts);
                                break;
                            }
                        }
                    }
                }
            }    

            #ifdef debug

                #pragma omp critical
                {
                    cout << "This is thread " << omp_get_thread_num() << " and in the level " << level << " and the hybrid flag is " << use_hybrid << "\n";
                    cout << "The vertices that have been found out by this thread are :- ";
                    for (int i = 0; i < thread_queue_size; ++i) cout << thread_queue[i] << " ";
                    cout << "\n";
                }

            #endif

            empty_queue(thread_queue, thread_queue_size, queue_next, queue_size_next,num_verts);
            #pragma omp barrier

            #pragma omp single
            { 
                num_descs += local_num_descs;

                if (!use_hybrid)
                {  
                    double edges_frontier = (double)local_num_descs * avg_out_degree;
                    double edges_remainder = (double)(num_verts - num_descs) * avg_out_degree;
                    if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0 && !already_switched)
                        use_hybrid = true;
                }
                else
                {
                    if ( ((double)num_verts / BETA) > local_num_descs  && !already_switched)
                    {
                        use_hybrid = false;
                        already_switched = true;
                    }
                }
                
                local_num_descs = 0;
                // ++num_levels;

                #ifdef debug
                    cout << "IN single - This is thread " << omp_get_thread_num() << " and in the level " << level << " and the hybrid flag is " << use_hybrid << "\n";
                    cout << "The vertices that have been found out in this round are :- ";
                    for (int i = 0; i < queue_size_next; ++i) cout << queue_next[i] << " ";
                    cout << "\n";
                #endif

                queue_size = queue_size_next;
                queue_size_next = 0;
                int* temp = queue;
                queue = queue_next;
                queue_next = temp;
                ++level;

            } // end single

        }
    } // end parallel
    double time2 = omp_get_wtime();

    delete [] queue;
    delete [] queue_next;

    return time2 - time;
}