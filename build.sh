#!/bin/bash

nvcc -gencode arch=compute_30,code=sm_30 -Xcompiler -fPIC -c probGpu.cu

nvcc -Xcompiler -Wall,-O3,-std=c++11,-flto -o atmonu_osc main.cc probGpu.o
