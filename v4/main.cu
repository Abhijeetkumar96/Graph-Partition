#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "GPU_BCC/headers/mytimer.h"
#include "GPU_BCC/headers/error_checking.h"
#include "GPU_BCC/headers/Graph_CSR.h"

#include "GPU_BCG/headers/BCG_DS.h"
#include "GPU_BCG/headers/BCG_Functions.h"
#include "GPU_BCG/headers/parameters.h"

#include "verification_helper_functions.h"

//Added line 16 - 22
//***************************************************//
#include "Partitioning/openmp_partition.h"
#include "CPU_BCC/cpu_bcc.h"
#include "Partitioning/bfs.h"
//***************************************************//

using namespace std;

#define check 1
// #define DEBUG
#define print_info

// double bfs(unweightedGraph &G, int root, int* parents, int* levels);

int binary_search(int start,int end,int* source,int value){

	if(start > end){
		cout << "the start index is greater than the end index\n";
		return -1;
	}
	
	int mid = (start+end)/2;

	while(mid > start){

		if(source[mid] == value) 
			return mid;

		if(source[mid] < value) 
			start = mid;
		else 
			end = mid;

		mid = (start+end)/2;

	}

	if(source[start] == value) 
		return start;
	else if(source[end] == value) 
		return end;
	else 
		return -1;
}

int binary_search_starting(int start,int end,int* source,int value){

	if(start >= end){
		cout << "the start index is greater than or equal to the end index\n";
		return -1;
	}
	
	int mid = (start+end)/2;

	while(mid > start){

		if(source[mid] == value) 
			end = mid;
		else 
			start = mid;

		mid = (start+end)/2;

	}

	if(source[start] == value || source[end] != value) 
		return -1;
	else 
		return end;
}

int binary_search_ending(int start,int end,int* source,int value){

	if(start >= end){
		cout << "the start index is greater than or equal to the end index\n";
		return -1;
	}
	
	int mid = (start+end)/2;

	while(mid > start){

		if(source[mid] == value) 
			start = mid;
		else 
			end = mid;

		mid = (start+end)/2;

	}

	if(source[start] != value || source[end] == value) 
		return -1;
	else 
		return start;
}

void Extract_BCC_numbers(unweightedGraph G, GPU_BCG_DS g_bcg_ds,bool* cut_vertex_status,int *vertex_bcc_numbers,int *edge_bcc_numbers){

	int* main_mapping = vertex_bcc_numbers;
	int* pb_vertex_bcc_numbers = new int[g_bcg_ds.num_components];
	int* pb_edge_bcc_numbers = new int[g_bcg_ds.EDGES_SIZE];
	int* U = new int[g_bcg_ds.EDGES_SIZE];
	int* V = new int[g_bcg_ds.EDGES_SIZE];

	check_for_error( cudaMemcpy(main_mapping, g_bcg_ds.label, sizeof(int)* g_bcg_ds.num_vertices ,cudaMemcpyDeviceToHost) , "cannot copy label to cpu" );
	check_for_error( cudaMemcpy(pb_vertex_bcc_numbers,g_bcg_ds.d_componentNumber,sizeof(int)* g_bcg_ds.num_components ,cudaMemcpyDeviceToHost) , "cannot copy d_componentNumber to cpu" );
	check_for_error( cudaMemcpy(pb_edge_bcc_numbers,g_bcg_ds.d_edge_bcc_nums,sizeof(int)* g_bcg_ds.EDGES_SIZE ,cudaMemcpyDeviceToHost) , "cannot copy d_edge_bcc_nums to cpu" );
	check_for_error( cudaMemcpy(U, g_bcg_ds.d_U,sizeof(int)* g_bcg_ds.EDGES_SIZE ,cudaMemcpyDeviceToHost) , "cannot copy d_U to cpu" );
	check_for_error( cudaMemcpy(V, g_bcg_ds.d_V,sizeof(int)* g_bcg_ds.EDGES_SIZE ,cudaMemcpyDeviceToHost) , "cannot copy d_V to cpu" );
	check_for_error( cudaMemcpy(cut_vertex_status,g_bcg_ds.d_cut_vertex, sizeof(bool) * g_bcg_ds.num_components, cudaMemcpyDeviceToHost ), "cannot copy d_cut_vertex to cpu" );
	#ifdef DEBUG
		// cout << "Main mapping : \n"; 
		// for(int i = 0; i < g_bcg_ds.num_vertices; i++)
		// 	cout << main_mapping[i] << " ";

		cout << "\nEdge list from Extract_BCC_numbers: \n";
		for(int i = 0; i <g_bcg_ds.EDGES_SIZE; ++i)
			cout << i <<" -> " << U[i] <<" " <<V[i] << endl;
		cout << endl;

		// cout << "V Edge list from Extract_BCC_numbers: \n";
		// for(int i = 0; i <g_bcg_ds.EDGES_SIZE; ++i)
		// 	cout << i <<" -> " << V[i] << endl;
		// cout << endl;

	#endif

	#ifdef check
		vector<bool> is_seen(g_bcg_ds.num_vertices, false);
		int freq = 0;
		for(int i = 0; i < g_bcg_ds.EDGES_SIZE; ++i)
		{
			if(!is_seen[pb_edge_bcc_numbers[i]]) {
				freq++;
				is_seen[pb_edge_bcc_numbers[i]] = true;
			}
		}
		cout << "Number of bcc's in main graph : " << freq << endl;

	#endif
	// calculate the edge bcc numbers

	int num_ver_in_pb_graph = g_bcg_ds.num_components;
	int num_edges_in_pb_graph = g_bcg_ds.EDGES_SIZE;

	int sp_case = 0;

	mytimer par_timer{};

	#ifdef DEBUG
		cout << "G.maundirected_edge_list_size = " << G.undirected_edge_list_size << endl;
	#endif

	#pragma omp parallel for
	for(int i = 0; i < G.undirected_edge_list_size; ++i)
	{

		int remap_u = main_mapping[G.U[i]];
		int remap_v = main_mapping[G.V[i]];

		#ifdef check
				if( ( remap_u < 0 || remap_u >= num_ver_in_pb_graph ) || ( remap_v < 0 || remap_v >= num_ver_in_pb_graph ) ){
					cout << "After remapping the edges, the remapped vertex id is out of range\n";
					exit(-1);
				}
		#endif

		int b_ver = -1;
		if(cut_vertex_status[remap_u] == false) 
			b_ver = remap_u;
		else if(cut_vertex_status[remap_v] == false) 
			b_ver = remap_v;

		if(b_ver != -1)
		{
			edge_bcc_numbers[i] = pb_vertex_bcc_numbers[b_ver];
			#ifdef DEBUG
				cout <<"Edge : " << G.U[i] << " " << G.V[i] <<" <- " << edge_bcc_numbers[i] << endl;
			#endif
		}
		else
		{

			if(remap_u > remap_v)
			{
				int t = remap_v;
				remap_v = remap_u;
				remap_u = t;
			}

			++sp_case;
/*
			for(int l=0;l<num_edges_in_pb_graph;++l){

				if( (U[l] == remap_u) && (V[l] == remap_v) ){
					edge_bcc_numbers[i] = pb_edge_bcc_numbers[l];
					break;
				}
#ifdef check
				if(l == (num_edges_in_pb_graph-1)) exit_with_error("no match is found in pb graph for the special case");
#endif
			}
*/			
			int some_ind = binary_search(0, num_edges_in_pb_graph-1, U, remap_u);

#ifdef check
			if(some_ind == -1){
				cout << "no match is found in pb graph for vertex u " << remap_u << "\n";
				exit(-1);
			}
#endif

			int start;
			if(U[0] == remap_u) start = 0;
			else start = binary_search_starting(0,some_ind,U,remap_u);

			if(start == -1){
				cout << "no match is found in pb graph for starting index of vertex u " << remap_u << "\n";
				exit(-1);
			}

			int end;

			if(U[num_edges_in_pb_graph-1] == remap_u) end = num_edges_in_pb_graph-1;
			else end = binary_search_ending(some_ind,num_edges_in_pb_graph-1,U,remap_u);

			if(end == -1){
				cout << "no match is found in pb graph for ending index of vertex u " << remap_u << "\n";
				exit(-1);
			}
	
			int index = binary_search(start,end,V,remap_v);
			
			if(index == -1){
				cout << "no match is found in pb graph for vertex v " << remap_v << "\n";
				exit(-1);
			}

			edge_bcc_numbers[i] = pb_edge_bcc_numbers[index];

		}

	}

	par_timer.timetaken_reset("parallel loop");

	// For testing, same for loop without openmp 
#ifdef DEBUG
	for(int i=0;i<G.undirected_edge_list_size;++i){

		int remap_u = main_mapping[G.U[i]];
		int remap_v = main_mapping[G.V[i]];

#ifdef check
		if( ( remap_u < 0 || remap_u >= num_ver_in_pb_graph ) || ( remap_v < 0 || remap_v >= num_ver_in_pb_graph ) ){
			cout << "After remapping the edges, the remapped vertex id is out of range\n";
			exit(-1);
		}
#endif

		int b_ver = -1;
		if(cut_vertex_status[remap_u] == false) b_ver = remap_u;
		else if(cut_vertex_status[remap_v] == false) b_ver = remap_v;

		if(b_ver != -1){
			edge_bcc_numbers[i] = pb_vertex_bcc_numbers[b_ver];
		}else{

			if(remap_u > remap_v){
				int t = remap_v;
				remap_v = remap_u;
				remap_u = t;
			}

			++sp_case;
/*
			for(int l=0;l<num_edges_in_pb_graph;++l){

				if( (U[l] == remap_u) && (V[l] == remap_v) ){
					edge_bcc_numbers[i] = pb_edge_bcc_numbers[l];
					break;
				}
#ifdef check
				if(l == (num_edges_in_pb_graph-1)) exit_with_error("no match is found in pb graph for the special case");
#endif
			}
*/			
			int some_ind = binary_search(0,num_edges_in_pb_graph-1,U,remap_u);

#ifdef check
			if(some_ind == -1){
				cout << "no match is found in pb graph for vertex u " << remap_u << "\n";
				exit(-1);
			}
#endif

			int start;
			if(U[0] == remap_u) start = 0;
			else start = binary_search_starting(0,some_ind,U,remap_u);

#ifdef check
			if(start == -1){
				cout << "no match is found in pb graph for starting index of vertex u " << remap_u << "\n";
				exit(-1);
			}
#endif

			int end;

			if(U[num_edges_in_pb_graph-1] == remap_u) end = num_edges_in_pb_graph-1;
			else end = binary_search_ending(some_ind,num_edges_in_pb_graph-1,U,remap_u);

#ifdef check
			if(end == -1){
				cout << "no match is found in pb graph for ending index of vertex u " << remap_u << "\n";
				exit(-1);
			}
#endif	

			int index = binary_search(start,end,V,remap_v);

#ifdef check
			if(index == -1){
				cout << "no match is found in pb graph for vertex v " << remap_v << "\n";
				exit(-1);
			}
#endif

			edge_bcc_numbers[i] = pb_edge_bcc_numbers[index];

		}

	}

	par_timer.timetaken("for serial loop");
#endif

	#ifdef DEBUG
		cout << "Size of edge list - " << G.undirected_edge_list_size << " and Number of special case edges - " << sp_case << "\n";
	#endif

	delete[] pb_vertex_bcc_numbers;
	delete[] pb_edge_bcc_numbers;
	delete[] U;
	delete[] V;

}

int main(int argc, char **argv) {

	// input format  = (#(vertices in G) , #(edges in G) , G in CSR Format using two arrays , sequence of edges from G)
	// Dataset path should be the first argument
	// output format = (Vertex BCC number and Edge BCC number)
	// The vertices are numbered 0 to n-1
	// Second argument should be the batch size
	// Third argument should be the number of threads
	// Fourth arguement should be pipeline design 1 or 2 denoted by s1 or s2

	/************************** Initialization ********************************/

	#ifdef DEBUG
    	std::cout << "DEBUG mode is on!" << std::endl;
		#else
    		std::cout << "DEBUG mode is off!" << std::endl;
	#endif

	mytimer module_timer{};
	// Enable nested parallelism
    // omp_set_nested(1);

	// checking whether the dataset name is passed as an argument
	if (argc < 5) {
	    std::cerr << "Usage: " << argv[0] << " <dataset_path> <batch_size> <num_threads> <pipeline_design (s1 or s2)>" << std::endl;
	    return 1;
	}

	// check whether the value of pipeline design is either "s1" or "s2".
	std::string pipeline_design = argv[4];
	if (pipeline_design != "s1" && pipeline_design != "s2") {
	    std::cerr << "Error: Pipeline design should be either 's1' or 's2'." << std::endl;
	    return 1;
	}


	cout << "=========================================================\n\n";
	cout << "Heterogeneous BCC for the file " << argv[1] << " started\n\n";

	/********************** Reading Graph *************************/

	ifstream inputGraph;
	inputGraph.open(argv[1]);

	if(!inputGraph) {
		std::cerr <<"Unable to find/open the dataset file for reading.\n";
		return EXIT_FAILURE;
	}

	int batch_size = stoi(argv[2]);

	if(batch_size < 1) 
		std::cerr << "batch size cannot be negative or zero.\n";
	
	unweightedGraph G (inputGraph);
	std::cout << "Created unweightedGraph." << endl;

	G.undirected_edge_list_size = G.totalEdges/2;
	#ifdef DEBUG
    	cout <<"Printing after first initialisation : " << G.undirected_edge_list_size << endl;
    #endif

	module_timer.timetaken_reset("Reading Data and Creating CSR");

	/******************************************************************/

	/*********************** Setting num of threds *********************/

	if (argc < 4) {
		omp_set_num_threads(omp_get_num_procs());
	}
	else {
		int nt = stoi(argv[3]);
		if (nt < 0) {
			cout << "the number of threads given is invalid\n";
			exit(-1);
		}
		if (nt > omp_get_num_procs()) {
			nt = omp_get_num_procs();
			cout << "asking more than the number of processors - " << nt << "\n";
		}
		omp_set_num_threads(nt);
		cout << "num threads var is set to " << nt << "\n";
	}

	/***************************************************************/

	cudaFree(0);

	module_timer.timetaken_reset("dummy first kernel");

	mytimer Algo_timer{};

	/**************** Here we partition the graph ************************/
	
	unweightedGraph G1, G2, E3;
	vector<int> where(G.totalVertices);
	int *renumbering = nullptr;

	std::vector<int> parent(G.totalVertices);
	int* levels = new int[G.totalVertices];

	#pragma omp parallel for
	for (int i = 0; i < G.totalVertices; ++i) 
		levels[i] = -1;

	//partition is present in "openmp_partition.h"
	// partition function takes input graph G as input and returns three graphs G1, G2 and cross_edges
	// and the partitioned array as output
	// input <- G
	// output <- G1, G2, E3, where, initial_renumbering

    // partition(&G, &G1, &G2, &E3, parent, levels, where, renumbering);
    partition(G, G1, G2, E3, parent, levels, where, renumbering);

    delete[] levels;
}

	/*****************************************************************************************/
	std::cout <<"G1.totalVertices = " << G1.totalVertices <<endl;
    std::cout << "G1.totalEdges = " << G1.totalEdges <<"\n";
    std::cout <<"G2.totalVertices = " << G2.totalVertices <<endl;
    std::cout << "G2.totalEdges = " << G2.totalEdges <<"\n";

    #ifdef DEBUG
	    std::cout << "G1 csr = \n";
	    for (int v = 0; v < G1.totalVertices; v++) 
	    {
	        std::cout << "Vertex " << v << " neighbors: ";
	        for (int i = G1.offset[v]; i < G1.offset[v + 1]; i++) 
	        {
	           std::cout << G1.neighbour[i] << " ";
	        }
	        std::cout << "\n";
	    }

	    std::cout << "G2 csr = \n";
	    for (int v = 0; v < G2.totalVertices; v++) 
	    {
	        std::cout << "Vertex " << v << " neighbors: ";
	        for (int i = G2.offset[v]; i < G2.offset[v + 1]; i++) 
	        {
	            std::cout << G2.neighbour[i] << " ";
	        }
	        std::cout << "\n";
	    }
	#endif

	G2.undirected_edge_list_size = G2.totalEdges/2;
	GPU_BCG_DS g_bcg_ds(G.totalVertices, G.undirected_edge_list_size, batch_size);

	module_timer.timetaken_reset("allocating the necessary data");

	/************************* Finidng Spanning tree of G2 **********************************/

	// Finding the spanning tree of graph G2 ------------> Needs to be replaced with par BFS

	int num_ver = G2.totalVertices;
	vector <edge> mst(num_ver);
	vector<int> parent(num_ver);
	vector<int> levels(num_ver,-1);

	int root = 0;

	bfs(G2, root, parent.data(), levels.data());

	for(int i=0;i<num_ver;++i){

		int u = i;
		int v = parent[u];

		mst[i].u = u;
		mst[i].v = v;

	}

	module_timer.timetaken_reset("Spanning tree calculation");

	#ifdef check
		if (verify_spanning_tree(G2.totalVertices, 0, parent.data())) 
			cout << "\n\nspanning tree is verified\n";
		else 
			cout << "The parent array will not represent spanning tree\n";
	#endif

	parent.clear();
	levels.clear();

	#ifdef DEBUG

		cout << "edges in the spanning tree are :-\n";

		for(int i=0;i<mst.size();++i) cout << mst[i].u << " " << mst[i].v << "\n";
	#endif


	/****************************************************************************************/
	
	module_timer.reset();

	/************** Here we create another thread to handle GPU part and main thread will execute CPU part **********************/
	
	//Added here new code by Abhijeet
	int* cpu_U = nullptr; 
	int* cpu_V = nullptr;
	int list_len;
	int* bcc_numbers;
	
	// output -> cpu_U, cpu_V, list size and bcc_numbers
	//cpu_bcc is available in CPU_BCC folder
	int mode = 0;
	#pragma omp parallel sections
	{        	
		#pragma omp section
        {
			bcc_numbers = cpu_bcc(&G1, cpu_U, cpu_V, list_len);
			std::cout << "list_len from cpu function = " << list_len <<std::endl;
		}
      
    // GPU part ---------->
    	#pragma omp section
        {
			if(argc > 4){
				if(argv[4][0] == 's'){
					if(argv[4][1] == '1'){
						mode = 1;
		#ifdef print_info
						cout << "\nusing pipeline design 1\n";
		#endif
					}
					else if(argv[4][1] == '2'){
						mode = 2;
		#ifdef print_info
				cout << "\nusing pipeline design 2\n";
		#endif
					}
				}
			}	
			#ifdef DEBUG
				std::cout << "\nCalling gpu_bcg_creation now \n";
			#endif
			g_bcg_ds = gpu_bcg_creation(G2,g_bcg_ds,mst.data(),mst.size(),mode);
		}
	}

		/*****************************************************************************************/

		module_timer.timetaken_reset("PB graph generation");

		// std::cout << "printing initial remapping : \n";
		// for(int i = 0; i <G.totalVertices; ++i)
		// 	std::cout << renumbering[i] <<" ";

	/************* combine results ***********/

	
	// G1_map -> mapping array for G1 BCG
	combine_partial_results(G.totalVertices, list_len, cpu_U, cpu_V, G1.totalVertices, bcc_numbers, renumbering, where, g_bcg_ds);

	// Updating the values to reflect the combined PB graph
	// We offset the cpu BCG graph vertices id ( + num ver in GPU PB graph)

	g_bcg_ds.num_vertices = G.totalVertices;
	g_bcg_ds.num_edges = G.undirected_edge_list_size;
	

	/****************************************/

	/************** Process cross edges *****************/
	// E3.totalVertices = G.totalVertices;
	process_cross_edges(E3, g_bcg_ds, mode);
	module_timer.timetaken_reset("processing cross edges");
	g_bcg_ds = Repair_bcc_numbers(g_bcg_ds);

	/******************************************************/

	module_timer.timetaken_reset("BCC number repair");

	/**************** Calculate BCC Numbers *******************/

	int* edge_bcc_numbers = new int[G.undirected_edge_list_size]; // Replace this with G
	int* vertex_bcc_numbers = new int[G.totalVertices];
	bool* cut_vertex_status = new bool[g_bcg_ds.num_components];

	Extract_BCC_numbers(G, g_bcg_ds, cut_vertex_status, vertex_bcc_numbers, edge_bcc_numbers);

	/*********************************************************/

	module_timer.timetaken_reset("Extracting BCC Numbers");
	Algo_timer.timetaken("Entire Heterogeneous BCC Algorithm");

	/*********************** Printing for Verification *************************/

	print_for_verification(G, G.totalVertices, G.undirected_edge_list_size, cut_vertex_status,
							vertex_bcc_numbers, edge_bcc_numbers, argv[1], g_bcg_ds);

	/***************************************************************************/

	module_timer.timetaken_reset("For printing");

	delete[] edge_bcc_numbers;
	// delete[] vertex_bcc_numbers;
	delete[] cut_vertex_status;
	delete[] bcc_numbers;
	delete cpu_U;
    delete cpu_V;

	cout << "\nHeterogeneous bcc for the file " << argv[1] << " is completed\n";
	cout << "\n=========================================================\n";

	return 0;
}