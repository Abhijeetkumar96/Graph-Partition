# Compiler and Flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++17
NVCCFLAGS = -std=c++17
OMPFLAG = -fopenmp

# Directories
INCLUDE_DIR = include
BIN_DIR = bin
COMMON_FILES_DIR = src/Common_Files

# Files
PARTITION_OBJ = obj/euler.o obj/listranking.o obj/partition.o
COMMON_OBJ = obj/bfs.o obj/graph_csr.o

# Targets
all: partition_demo

partition_demo: obj/partition_demo.o $(PARTITION_OBJ) $(COMMON_OBJ)
	$(NVCC) $(NVCCFLAGS) -Xcompiler $(OMPFLAG) -I$(INCLUDE_DIR) obj/partition_demo.o obj/utility.o obj/serialScan.o obj/pointerJumping.o obj/euler_utility.o obj/partitionToCSR.o $(PARTITION_OBJ) $(COMMON_OBJ) -o $(BIN_DIR)/partition_demo -lcudart

# Object file rules
obj/bfs.o: $(COMMON_FILES_DIR)/bfs.cpp obj/graph_csr.o
	$(CXX) $(CXXFLAGS) $(OMPFLAG) -I$(INCLUDE_DIR) -c $(COMMON_FILES_DIR)/bfs.cpp -o obj/bfs.o

obj/graph_csr.o: $(COMMON_FILES_DIR)/Graph_CSR.cpp $(INCLUDE_DIR)/Common_Files/Graph_CSR.h
	$(CXX) $(CXXFLAGS) $(OMPFLAG) -I$(INCLUDE_DIR) -c $(COMMON_FILES_DIR)/Graph_CSR.cpp -o obj/graph_csr.o

obj/partition.o: include/Graph_Partition/partition.h \
                 obj/bfs.o obj/graph_csr.o obj/euler.o \
                 include/Common_Files/error_checking.h \
                 obj/partitionToCSR.o
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c src/Graph_Partition/partition.cu -o obj/partition.o 

obj/euler.o: src/Graph_Partition/euler.cu \
             include/Graph_Partition/euler.h \
             obj/listranking.o \
             include/Common_Files/error_checking.h \
             obj/utility.o \
             obj/pointerJumping.o \
             obj/euler_utility.o
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c src/Graph_Partition/euler.cu -o obj/euler.o

obj/pointerJumping.o: src/Graph_Partition/pointerJumping.cu include/Graph_Partition/pointerJumping.h
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c src/Graph_Partition/pointerJumping.cu -o obj/pointerJumping.o

obj/listranking.o: src/Graph_Partition/listranking.cu include/Graph_Partition/listranking.h
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c src/Graph_Partition/listranking.cu -o obj/listranking.o

obj/euler_utility.o: src/Graph_Partition/euler_utility.cu include/Graph_Partition/euler_utility.h
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -c src/Graph_Partition/euler_utility.cu -o obj/euler_utility.o

obj/utility.o: src/Common_Files/utility.cpp include/Common_Files/utility.h
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $(COMMON_FILES_DIR)/utility.cpp -o obj/utility.o

obj/partitionToCSR.o: src/Graph_Partition/partitionToCSR.cu include/Graph_Partition/partitionToCSR.h include/Common_Files/error_checking.h include/Common_Files/Graph_CSR.h obj/serialScan.o
	nvcc $(NVCCFLAGS) -Xcompiler -fopenmp -I$(INCLUDE_DIR) -c src/Graph_Partition/partitionToCSR.cu -o obj/partitionToCSR.o

obj/serialScan.o: src/Common_Files/serialScan.cpp include/Common_Files/serialScan.h
	$(CXX) -Wall -std=c++17 -O3 -I$(INCLUDE_DIR) -c $(COMMON_FILES_DIR)/serialScan.cpp -o obj/serialScan.o

obj/partition_demo.o: $(COMMON_OBJ) $(PARTITION_OBJ) src/Graph_Partition/partition_demo.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -I$(INCLUDE_DIR) -c src/Graph_Partition/partition_demo.cpp -o obj/partition_demo.o

# Debug and verify flags
debug: CXXFLAGS += -DDEBUG -DVERIFY
debug: NVCCFLAGS += -DDEBUG -DVERIFY
debug: all

verify: CXXFLAGS += -DVERIFY
verify: NVCCFLAGS += -DVERIFY
verify: all

# Clean rule
clean:
	rm -f $(BIN_DIR)/* obj/*

.PHONY: all clean partition_demo debug verify
