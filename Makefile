all: atmonu_osc

#!!! CUDA shared libs need to be specified AFTER the object files that use them
atmonu_osc: probGpu.o
	g++ -o atmonu_osc -Wall -O3 -std=c++14 -flto -march=native main.cpp probGpu.o -L/usr/local/cuda/lib64 -lcuda -lcudart

probGpu.o:
	nvcc -std=c++11 -arch=sm_30 -c probGpu.cu

.PHONY: clean
clean:
	rm -rf *.o atmonu_osc
