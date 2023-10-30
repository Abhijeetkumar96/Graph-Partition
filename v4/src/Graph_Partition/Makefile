#__*__ Graph_Partition Makefile __*__
CXX= c++
NVCC= nvcc
CXXFLAGS= -std=c++17 -Wall
OMPFLAGS= -fopenmp
NVCCFLAGS= -std=c++17

all: final

final: main.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) main.o obj/partition.o ../Common_Files/bfs.o ../Common_Files/Graph_CSR.o -o final

main.o: main.cpp src/partition.o ../Common_Files/bfs.o ../Common_Files/Graph_CSR.o ../Common_Files/headers/error_checking.h ../Common_Files/headers/mytimer.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c main.cpp

src/partition.o: src/partition.cpp src/partition.h 
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c src/partition.cpp -o obj/partition.o

../Common_Files/bfs.o: ../Common_Files/bfs.cpp ../Common_Files/headers/bfs.h ../Common_Files/Graph_CSR.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c ../Common_Files/bfs.cpp -o ../Common_Files/bfs.o

../Common_Files/Graph_CSR.o: ../Common_Files/Graph_CSR.cpp ../Common_Files/headers/Graph_CSR.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c ../Common_Files/Graph_CSR.cpp -o ../Common_Files/Graph_CSR.o

clean:
	rm -rf main.o src/partition.o ../Common_Files/Graph_CSR.o ../Common_Files/bfs.o

.PHONY: all clean