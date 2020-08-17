# CUDA
CUDA = nvcc
INCS = $(wildcard *.cuh)
TARGET = $(patsubst %.cu, build/%_cuda.out, $(wildcard *.cu))
FLAGS = -std=c++11 -O3 -v
FLAGS += -I .
#FLAGS += -ptx

all:$(TARGET)

${TARGET}:build/%_cuda.out:%.cu
	        ${CUDA} $< ${FLAGS} -o $@

$(info ${CUDA_TARGET})