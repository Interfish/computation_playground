CXX = g++
MKLROOT = ~/miniconda3/envs/computation_playground/lib
TARGET = $(patsubst %.cpp, build/%.out, $(wildcard *.cpp))
FLAGS = -std=c++11 -O3 -DNDEBUG
MKL_FLAGS = -Wl,--start-group $(MKLROOT)/libmkl_intel_lp64.so $(MKLROOT)/libmkl_core.so $(MKLROOT)/libmkl_gnu_thread.so -Wl,--end-group -ldl -lpthread -fopenmp -lm
FLAGS += ${MKL_FLAGS}

all:$(TARGET)

${TARGET}:build/%.out:%.cpp
	        ${CXX} $< ${FLAGS} -o $@